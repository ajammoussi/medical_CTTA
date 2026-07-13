import torch
import torch.nn as nn
from typing import Iterator
import timm
from huggingface_hub import hf_hub_download
import os
from src.models.base import FoundationModel
from src.models.registry import register_model


def _interpolate_pos_embed(model, state_dict):
    """Resize position embeddings to match a non-default image size.

    ViT models embed each image patch at a specific position in a 2D grid.
    The position embeddings (pos_embed) are learned parameters with one
    vector per grid cell. When the input image differs from the pretraining
    size (224px), the grid changes (e.g. 14x14 for 224px vs 28x28 for 448px)
    and the stored pos_embed has the wrong number of entries.

    This function:
    1. Separates special tokens (e.g. [CLS]) from spatial grid tokens
    2. Reshapes the 1D grid-token sequence into a 2D grid
    3. Bicubic-interpolates the 2D grid from the old size to the new size
    4. Reassembles: special tokens + interpolated grid tokens

    Args:
        model: The timm ViT model instance (provides expected grid dimensions)
        state_dict: The pretrained checkpoint's state dict (modified in-place)
    """
    if "pos_embed" not in state_dict:
        return
    # pos_embed shape: (1, num_tokens, embed_dim), e.g. (1, 197, 1024) for 224px
    # where num_tokens = 1 [CLS] token + 196 patch tokens (14x14 grid)
    pos_embed = state_dict["pos_embed"]

    # Special tokens are the difference between total sequence length and patch count
    # Typically just the [CLS] token (1 extra), but some ViT variants have more
    num_extra_tokens = model.pos_embed.shape[-2] - model.patch_embed.num_patches
    embed_dim = pos_embed.shape[-1]

    # Original grid: subtract special tokens, then take sqrt to get grid dimension
    # e.g. (197 - 1 = 196) -> sqrt(196) = 14
    orig_grid_size = int(((pos_embed.shape[-2] - num_extra_tokens) ** 0.5))
    # New grid: based on the model's expected patch count (depends on image_size)
    # e.g. for 448px: (448/16)^2 = 784 patches -> sqrt(784) = 28
    new_grid_size = int(model.patch_embed.num_patches ** 0.5)

    if orig_grid_size == new_grid_size:
        return

    # Extract special tokens (e.g. [CLS]) as-is — they are not spatial and don't need interpolation
    # Shape: (1, num_extra_tokens, embed_dim)
    cls_token = pos_embed[:, :num_extra_tokens]

    # Reshape grid tokens to 2D: (1, grid_tokens, embed_dim) -> (1, H, W, embed_dim)
    # Then permute to (1, embed_dim, H, W) for PyTorch's interpolate (expects channels-first)
    # Shape: (1, embed_dim, orig_grid_size, orig_grid_size)
    pos_grid = pos_embed[:, num_extra_tokens:] \
        .reshape(-1, orig_grid_size, orig_grid_size, embed_dim) \
        .permute(0, 3, 1, 2)

    # Bicubic interpolation treats each embedding dimension as an independent channel
    # and resamples the 2D grid to the target resolution
    # Shape: (1, embed_dim, new_grid_size, new_grid_size)
    pos_grid = torch.nn.functional.interpolate(
        pos_grid, size=(new_grid_size, new_grid_size),
        mode="bicubic", align_corners=False)

    # Reverse the permutation+flatten and concatenate special tokens back
    # (1, embed_dim, H, W) -> (1, H, W, embed_dim) -> (1, H*W, embed_dim)
    # Then concatenate with [CLS]: (1, num_extra_tokens + H*W, embed_dim)
    state_dict["pos_embed"] = torch.cat(
        (cls_token, pos_grid.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)


@register_model("retfound")
class RETFound(FoundationModel):
    """RETFound: ViT-L/16 MAE-pretrained on retinal fundus photos."""

    def __init__(self, num_classes=5, freeze_layers=20,
                 checkpoint="YukunZhou/RETFound_mae_natureCFP",
                 image_size=224, drop_path=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.drop_path = drop_path
        self.backbone = None
        self.classifier = None

    def load_weights(self, checkpoint_path=None):
        """Build the ViT backbone, attach a classifier, load pretrained weights, and freeze layers.

        This method is called once at initialization. It:
        1. Creates a timm ViT-Large/16 backbone (no classification head, avg pooling)
        2. Attaches a fresh Linear(1024, num_classes) classifier
        3. Loads RETFound MAE-pretrained weights (from local file or HuggingFace Hub)
        4. Strips MAE decoder + mask token + old head from the checkpoint
        5. Interpolates position embeddings if image_size differs from 224
        6. Loads into the model, tolerating expected key mismatches
        7. Freezes early layers to save memory and preserve general features
        """
        # --- Build the backbone ---
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,                # load RETFound's retinal weights, not ImageNet
            num_classes=0,                   # removes the default classification head — we add our own
            global_pool="avg",               # averages patch tokens instead of using only [CLS]
            drop_path_rate=self.drop_path,   # regularization (0.2 is default for ViT-L)
            img_size=self.image_size,        # input image resolution (224px by default, supports 448px via interpolation)
        )
        # 1024 is ViT-L's hidden dimension; output is 5 DR grades (0-4)
        self.classifier = nn.Linear(self.backbone.num_features, self.num_classes)

        # --- Locate and load the pretrained checkpoint ---
        ckpt_path = checkpoint_path or self.checkpoint
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
            hf_path = hf_hub_download(
                repo_id=ckpt_path if ckpt_path else self.checkpoint,
                filename="RETFound_mae_natureCFP.pth",
                cache_dir=os.environ.get("HF_HOME", "./cache"),
            )
            checkpoint_dict = torch.load(hf_path, map_location="cpu", weights_only=False)

        # --- Extract and clean the state dict ---
        # Checkpoint may be {'model': state_dict, 'optimizer': ...} (training ckpt)
        # or just the raw state dict (weights-only). Handle both.
        state_dict = (
            checkpoint_dict["model"]
            if isinstance(checkpoint_dict, dict) and "model" in checkpoint_dict
            else checkpoint_dict
        )
        
        for k in list(state_dict.keys()):
            # Remove MAE decoder weights — we only need the encoder (backbone)
            # Remove mask_token — a learned MAE token not used in classification
            if k.startswith("decoder") or k == "mask_token":
                state_dict.pop(k, None)
            
            # Remove old head weights — we have our own classifier
            if k in ("head.weight", "head.bias"):
                state_dict.pop(k, None)

        # --- Handle image size mismatch via position embedding interpolation ---
        # If image_size != 224, the patch grid is a different size (e.g. 14x14 -> 28x28).
        _interpolate_pos_embed(self.backbone, state_dict)

        # --- Load weights (non-strict because of known key mismatch) ---
        load_msg = self.backbone.load_state_dict(state_dict, strict=False)
        expected_missing = {"fc_norm.weight", "fc_norm.bias"}
        unexpected = set(load_msg.missing_keys) - expected_missing
        if unexpected:
            import logging
            logging.getLogger(__name__).warning(
                "Unexpected missing keys when loading RETFound: %s", unexpected
            )

        # --- Freeze early layers to preserve general features + save memory ---
        # patch_embed learns low-level features (edges, textures) — always freeze it.
        # The first freeze_layers transformer blocks are also frozen.
        # Only the last (24 - freeze_layers) blocks and the classifier head are trainable.
        # This prevents overfitting on small datasets and saves ~250M param gradients.
        if self.freeze_layers > 0:
            self.backbone.patch_embed.requires_grad_(False)
            for i, block in enumerate(self.backbone.blocks):
                if i < self.freeze_layers:
                    block.requires_grad_(False)

    def forward(self, x):
        """Full forward pass: backbone (with avg pooling) -> classifier -> logits.

        Used for training and evaluation. The backbone internally averages
        all 196 patch token outputs (global_pool="avg"), applies fc_norm,
        and returns a single 1024-dim vector per image.

        Args:
            x: Input tensor of shape (B, 3, H, W), normalized with ImageNet stats
        Returns:
            Logits tensor of shape (B, 5) for DR grades 0-4
        """
        features = self.backbone(x)
        return self.classifier(features)

    def extract_embedding(self, x):
        """Extract the raw [CLS] token embedding for shift detection.

        Unlike forward(), this method manually runs the ViT forward pass
        and returns ONLY the [CLS] token, bypassing the average pooling
        that timm's backbone applies internally.

        Why the [CLS] token instead of avg-pooled features?
        - The [CLS] token learns via self-attention to aggregate the most
          discriminative patch information — it's more sensitive to subtle
          distribution changes in specific image regions.
        - Average pooling blends all 196 patches equally, which can dilute
          localized domain shifts (e.g. a brightness difference only in
          the upper-left quadrant).
        - The 1024-dim [CLS] embedding preserves more information than the
          5-dim logits, making it a better signal for drift detection.

        This is called by EmbeddingDriftDetector during CTTA to track
          whether the feature distribution is shifting away from the source.

        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            [CLS] token embeddings of shape (B, 1024) — no avg pooling applied
        """
        # Step 1: Patch embedding — split image into 16x16 patches, project to 1024-dim
        # Input:  (B, 3, 224, 224)
        # Output: (B, 196, 1024)  where 196 = (224/16)^2 patches
        x = self.backbone.patch_embed(x)

        # Step 2: Prepend learned [CLS] token — one per sample in the batch
        # self.backbone.cls_token has shape (1, 1, 1024), expand to (B, 1, 1024)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)

        # Step 3: Concatenate: sequence = [CLS, patch_1, patch_2, ..., patch_196]
        # Shape: (B, 197, 1024)
        x = torch.cat([cls_token, x], dim=1)

        # Step 4: Add position embeddings to encode spatial location of each patch
        # Without this, the transformer is permutation-invariant (bag of patches)
        x = x + self.backbone.pos_embed

        # Step 5: Dropout on position-encoded tokens (regularization, train only)
        x = self.backbone.pos_drop(x)

        # Step 6: Forward through all 24 transformer blocks
        # Each block: LayerNorm -> Multi-Head Self-Attention -> Residual ->
        #             LayerNorm -> MLP (Linear+GELU+Linear) -> Residual
        # After all 24 blocks, each token has contextualized information
        # from all other tokens via attention
        for block in self.backbone.blocks:
            x = block(x)

        # Step 7: Final LayerNorm (stabilizes the output distribution)
        x = self.backbone.norm(x)

        # Step 8: Extract only the [CLS] token (index 0 in the sequence)
        # Shape: (B, 1024)
        # Note: NO average pooling — this is the raw attention-aggregated
        # representation, more sensitive to domain shifts than the smoothed
        # pooled features used in forward()
        return x[:, 0]

    def get_num_classes(self):
        return self.num_classes

    def get_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    @property
    def model_embed_dim(self):
        return 1024 if self.backbone is None else self.backbone.num_features
