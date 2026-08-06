import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, Dict
import timm
from huggingface_hub import hf_hub_download
import os
from src.models.base import FoundationModel
from src.models.registry import register_model


logger = logging.getLogger(__name__)


_RETFOUND_CACHED_WEIGHTS: Dict[str, Dict[str, torch.Tensor]] = {}


def _get_checkpoint_filename(repo_id: str) -> str:
    """Derive checkpoint filename from HuggingFace repo_id.

    Convention: YukunZhou/RETFound_mae_natureCFP → RETFound_mae_natureCFP.pth
    """
    return repo_id.split("/")[-1] + ".pth"


def _is_dinov2_checkpoint(repo_id: str) -> bool:
    """Detect if checkpoint is DINOv2-based (vs MAE-based)."""
    return "dinov2" in repo_id.lower()


def _get_timm_arch(repo_id: str) -> str:
    """Select timm model name based on checkpoint type."""
    if _is_dinov2_checkpoint(repo_id):
        return "vit_large_patch14_dinov2"
    return "vit_large_patch16_224"


def _adapt_state_dict_keys(state_dict: Dict) -> Dict:
    """Strip common prefixes from state dict keys for cross-architecture loading."""
    prefixes_to_strip = ["backbone.", "module.", "student.", "teacher.", "encoder."]
    for prefix in prefixes_to_strip:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            adapted = {}
            for k, v in state_dict.items():
                key = k[len(prefix):] if k.startswith(prefix) else k
                adapted[key] = v
            logger.info(f"Stripped '{prefix}' prefix from {len(adapted)} state dict keys")
            return adapted
    return state_dict


def _interpolate_pos_embed(model, state_dict):
    if "pos_embed" not in state_dict:
        return
    pos_embed = state_dict["pos_embed"]
    num_extra_tokens = model.pos_embed.shape[-2] - model.patch_embed.num_patches
    embed_dim = pos_embed.shape[-1]
    orig_grid_size = int(((pos_embed.shape[-2] - num_extra_tokens) ** 0.5))
    new_grid_size = int(model.patch_embed.num_patches ** 0.5)
    if orig_grid_size == new_grid_size:
        return
    cls_token = pos_embed[:, :num_extra_tokens]
    pos_grid = pos_embed[:, num_extra_tokens:] \
        .reshape(-1, orig_grid_size, orig_grid_size, embed_dim) \
        .permute(0, 3, 1, 2)
    pos_grid = torch.nn.functional.interpolate(
        pos_grid, size=(new_grid_size, new_grid_size),
        mode="bicubic", align_corners=False)
    state_dict["pos_embed"] = torch.cat(
        (cls_token, pos_grid.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)


@register_model("retfound")
class RETFound(FoundationModel):
    """RETFound: ViT-L/16 (MAE) or ViT-L/14 (DINOv2) pretrained on retinal fundus photos.

    Architecture is auto-detected from checkpoint name:
      - ``*dinov2*`` → ViT-L/14 DINOv2  (timm: ``vit_large_patch14_dinov2``)
      - otherwise   → ViT-L/16 MAE     (timm: ``vit_large_patch16_224``)
    """

    def __init__(self, num_classes=5, freeze_layers=20,
                 checkpoint="YukunZhou/RETFound_dinov2_meh",
                 image_size=224, drop_path=0.2,
                 normalize_features=True):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.drop_path = drop_path
        self.normalize_features = normalize_features
        self.backbone = None
        self.classifier = None
        self._timm_arch = _get_timm_arch(checkpoint)

    def load_weights(self, checkpoint_path=None):
        ckpt_path = checkpoint_path or self.checkpoint
        self._timm_arch = _get_timm_arch(ckpt_path)

        if _is_dinov2_checkpoint(ckpt_path):
            self.backbone = timm.create_model(
                "vit_large_patch14_dinov2",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
                img_size=self.image_size,
            )
        else:
            self.backbone = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
                drop_path_rate=self.drop_path,
                img_size=self.image_size,
            )
        self.classifier = nn.Linear(self.backbone.num_features, self.num_classes)

        if ckpt_path not in _RETFOUND_CACHED_WEIGHTS:
            if ckpt_path is not None and os.path.exists(ckpt_path):
                checkpoint_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            else:
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token is None:
                    raise RuntimeError(
                        "RETFound weights are gated. Set HF_TOKEN env var or run "
                        "`huggingface-cli login` before importing the model."
                    )
                os.environ["HF_TOKEN"] = hf_token
                filename = _get_checkpoint_filename(ckpt_path)
                hf_path = hf_hub_download(
                    repo_id=ckpt_path,
                    filename=filename,
                    cache_dir=os.environ.get("HF_HOME", "./cache"),
                )
                checkpoint_dict = torch.load(hf_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint_dict, dict) and "student" in checkpoint_dict:
                state_dict = checkpoint_dict["student"]
                logger.info("Extracted state_dict from DINOv2 'student' key")
            elif isinstance(checkpoint_dict, dict) and "teacher" in checkpoint_dict:
                state_dict = checkpoint_dict["teacher"]
                logger.info("Extracted state_dict from DINOv2 'teacher' key")
            elif isinstance(checkpoint_dict, dict) and "model" in checkpoint_dict:
                state_dict = checkpoint_dict["model"]
            else:
                state_dict = checkpoint_dict

            for k in list(state_dict.keys()):
                if k.startswith("decoder") or k == "mask_token":
                    state_dict.pop(k, None)
                if k in ("head.weight", "head.bias"):
                    state_dict.pop(k, None)

            state_dict = _adapt_state_dict_keys(state_dict)

            _RETFOUND_CACHED_WEIGHTS[ckpt_path] = state_dict
            del checkpoint_dict
            logger.info("RETFound checkpoint cached in CPU RAM")

        cached = _RETFOUND_CACHED_WEIGHTS[ckpt_path]
        state_dict = dict(cached)

        if "pos_embed" in cached:
            state_dict["pos_embed"] = cached["pos_embed"].clone()
            _interpolate_pos_embed(self.backbone, state_dict)

        load_msg = self.backbone.load_state_dict(state_dict, strict=False)
        del state_dict

        expected_missing = {"fc_norm.weight", "fc_norm.bias"}
        if _is_dinov2_checkpoint(ckpt_path):
            expected_missing.add("head.weight")
            expected_missing.add("head.bias")
        unexpected = set(load_msg.missing_keys) - expected_missing
        if unexpected:
            logger.warning(
                "Unexpected missing keys when loading RETFound: %s", unexpected
            )

        if self.freeze_layers > 0:
            self.backbone.patch_embed.requires_grad_(False)
            for i, block in enumerate(self.backbone.blocks):
                if i < self.freeze_layers:
                    block.requires_grad_(False)

        if self.normalize_features:
            logger.info("Feature normalization ENABLED in forward() (required for prototype classifier)")

    def forward(self, x):
        """Full forward pass: backbone (with avg pooling) -> classifier -> logits.

        The backbone internally averages all patch token outputs
        (global_pool="avg") and returns a single vector per image.

        Args:
            x: Input tensor of shape (B, 3, H, W), normalized with ImageNet stats
        Returns:
            Logits tensor of shape (B, ``num_classes``)
        """
        features = self.backbone(x)
        if self.normalize_features:
            features = F.normalize(features, dim=1)
        return self.classifier(features)

    def extract_embedding(self, x):
        """Extract [CLS] token embedding for shift detection.

        Uses ``forward_features`` (architecture-agnostic) to get all tokens,
        then returns the CLS token at index 0.

        NOTE: Does NOT apply L2 normalization. Shift detection operates on
        raw embedding space where feature norms carry distribution information.
        """
        x = self.backbone.forward_features(x)
        return x[:, 0]

    def get_num_classes(self):
        return self.num_classes

    def get_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    @property
    def model_embed_dim(self):
        return 1024 if self.backbone is None else self.backbone.num_features
