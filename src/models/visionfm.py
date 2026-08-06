"""VisionFM: ViT-Base/16 foundation model for ophthalmic imaging.

Pretrained with iBOT (DINO + BEiT) on 3.4M fundus images.
Architecture: ViT-Base (768-dim, 12 blocks, 12 heads, patch16, 224x224).
Feature extraction: CLS tokens from last 4 blocks → 3072-dim.
Normalization: Fundus-specific stats (mean=[0.4237, 0.2609, 0.1284],
                std=[0.2948, 0.2017, 0.1367]).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator, Dict, Optional
import os
from src.models.base import FoundationModel
from src.models.registry import register_model
from src.models.visionfm_vit import VisionTransformer

logger = logging.getLogger(__name__)

_VFM_CACHED_WEIGHTS: Dict[str, Dict[str, torch.Tensor]] = {}

# Fundus normalization statistics from VisionFM paper
FUNDUS_MEAN = [0.4237, 0.2609, 0.1284]
FUNDUS_STD = [0.2948, 0.2017, 0.1367]


def _adapt_state_dict_keys(state_dict: Dict) -> Dict:
    """Strip common prefixes from state dict keys."""
    prefixes_to_strip = ["backbone.", "module.", "student.", "teacher.", "encoder."]
    for prefix in prefixes_to_strip:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            adapted = {}
            for k, v in state_dict.items():
                key = k[len(prefix):] if k.startswith(prefix) else k
                adapted[key] = v
            logger.info("Stripped '%s' prefix from %d state dict keys", prefix, len(adapted))
            return adapted
    return state_dict


def _interpolate_pos_embed(model: VisionTransformer, state_dict: Dict):
    """Interpolate position embeddings if resolution differs."""
    if "pos_embed" not in state_dict:
        return
    pos_embed = state_dict["pos_embed"]
    num_extra_tokens = 1  # CLS token
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


@register_model("visionfm")
class VisionFM(FoundationModel):
    """VisionFM: ViT-Base/16 pretrained with iBOT on retinal fundus images.

    Architecture: 768-dim, 12 transformer blocks, 12 heads, patch16, 224x224.
    Feature extraction: CLS tokens from last 4 blocks → 3072-dim.
    Classification head: nn.Linear(3072, num_classes).
    """

    def __init__(self, num_classes=5, freeze_layers=8,
                 checkpoint="fundus", image_size=224, drop_path=0.1,
                 normalize_features=True):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.drop_path = drop_path
        self.normalize_features = normalize_features
        self.backbone: Optional[VisionTransformer] = None
        self.classifier: Optional[nn.Linear] = None

    def load_weights(self, checkpoint_path=None):
        """Load VisionFM pretrained weights.

        Args:
            checkpoint_path: Path to .pth file. If None, downloads from
                Google Drive using visionfm_download.py.
        """
        if checkpoint_path is None:
            from src.models.visionfm_download import download_visionfm_weights
            checkpoint_path = download_visionfm_weights()

        # Instantiate backbone
        self.backbone = VisionTransformer(
            img_size=self.image_size,
            patch_size=16,
            in_chans=3,
            num_classes=0,  # no head
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=self.drop_path,
            return_all_tokens=True,
            use_mean_pooling=False,
        )

        # Classification head: 3072-dim (4 blocks × 768 CLS tokens) → num_classes
        self.classifier = nn.Linear(768 * 4, self.num_classes)

        # Load checkpoint
        if checkpoint_path not in _VFM_CACHED_WEIGHTS:
            logger.info("Loading VisionFM checkpoint from %s", checkpoint_path)
            checkpoint_dict = torch.load(checkpoint_path, map_location="cpu",
                                         weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint_dict, dict):
                if "teacher" in checkpoint_dict:
                    state_dict = checkpoint_dict["teacher"]
                    logger.info("Extracted state_dict from 'teacher' key")
                elif "student" in checkpoint_dict:
                    state_dict = checkpoint_dict["student"]
                    logger.info("Extracted state_dict from 'student' key")
                elif "model" in checkpoint_dict:
                    state_dict = checkpoint_dict["model"]
                    logger.info("Extracted state_dict from 'model' key")
                elif "visionfm_state_dict" in checkpoint_dict:
                    state_dict = checkpoint_dict["visionfm_state_dict"]
                    logger.info("Extracted state_dict from 'visionfm_state_dict' key")
                else:
                    state_dict = checkpoint_dict
            else:
                state_dict = checkpoint_dict

            # Strip prefixes
            state_dict = _adapt_state_dict_keys(state_dict)

            # Remove decoder/mask tokens and head weights
            for k in list(state_dict.keys()):
                if k.startswith("decoder") or k == "mask_token":
                    state_dict.pop(k, None)
                if k in ("head.weight", "head.bias"):
                    state_dict.pop(k, None)

            _VFM_CACHED_WEIGHTS[checkpoint_path] = state_dict
            del checkpoint_dict
            logger.info("VisionFM checkpoint cached in CPU RAM")

        cached = _VFM_CACHED_WEIGHTS[checkpoint_path]
        state_dict = dict(cached)

        # Interpolate position embeddings if needed
        if "pos_embed" in cached:
            state_dict["pos_embed"] = cached["pos_embed"].clone()
            _interpolate_pos_embed(self.backbone, state_dict)

        load_msg = self.backbone.load_state_dict(state_dict, strict=False)
        del state_dict

        expected_missing = {"fc_norm.weight", "fc_norm.bias", "norm.weight", "norm.bias"}
        unexpected = set(load_msg.missing_keys) - expected_missing
        if unexpected:
            logger.warning("Unexpected missing keys when loading VisionFM: %s", unexpected)

        # Freeze first N transformer blocks
        if self.freeze_layers > 0:
            self.backbone.patch_embed.requires_grad_(False)
            self.backbone.cls_token.requires_grad_(False)
            self.backbone.pos_embed.requires_grad_(False)
            for i, block in enumerate(self.backbone.blocks):
                if i < self.freeze_layers:
                    block.requires_grad_(False)
            logger.info("Frozen first %d of 12 transformer blocks", self.freeze_layers)

        if self.normalize_features:
            logger.info("Feature normalization ENABLED in forward()")

    def forward(self, x):
        """Forward pass: backbone (last 4 blocks CLS concat) → classifier → logits.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        intermediate = self.backbone.get_intermediate_layers(x, n=4)
        # Concatenate CLS tokens from last 4 blocks: (B, 768*4) = (B, 3072)
        features = torch.cat([block[:, 0] for block in intermediate], dim=-1)
        if self.normalize_features:
            features = F.normalize(features, dim=1)
        return self.classifier(features)

    def extract_embedding(self, x):
        """Extract CLS token embedding from last block for shift detection.

        Returns 768-dim raw embedding (no L2 normalization).
        """
        # backbone has return_all_tokens=True, so forward returns all tokens
        all_tokens = self.backbone(x)
        return all_tokens[:, 0]

    def extract_features(self, x):
        """Extract 3072-dim features (CLS concat from last 4 blocks).

        Used by the runner for prototype initialization and evaluation.
        No L2 normalization applied.
        """
        intermediate = self.backbone.get_intermediate_layers(x, n=4)
        return torch.cat([block[:, 0] for block in intermediate], dim=-1)

    def get_num_classes(self):
        return self.num_classes

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.backbone.parameters() if p.requires_grad)

    @property
    def model_embed_dim(self):
        return 768 * 4 if self.backbone is None else 768 * 4  # 3072 for classifier
