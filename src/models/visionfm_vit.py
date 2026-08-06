"""VisionFM ViT-Base backbone.

Standalone implementation matching VisionFM's models/vision_transformer.py.
ViT-Base/16: embed_dim=768, depth=12, num_heads=12, patch_size=16, img_size=224.
Pretrained with iBOT (DINO + BEiT) on 3.4M ophthalmic images.
"""

import torch
import torch.nn as nn
from functools import partial


class PatchEmbed(nn.Module):
    """Image to Patch Embedding via Conv2d."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention with optional Flash Attention support."""

    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None,
                 attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop_rate=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm (LayerNorm -> Attention -> residual,
    LayerNorm -> MLP -> residual)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden,
                       act_layer=act_layer, drop_rate=drop_rate)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class VisionTransformer(nn.Module):
    """Vision Transformer backbone matching VisionFM architecture.

    Args:
        img_size: Input image size (default 224).
        patch_size: Patch size (default 16).
        in_chans: Number of input channels (default 3).
        num_classes: Number of classes (0 = no head, return features).
        embed_dim: Embedding dimension (default 768).
        depth: Number of transformer blocks (default 12).
        num_heads: Number of attention heads (default 12).
        mlp_ratio: MLP hidden dim ratio (default 4.0).
        qkv_bias: Use QKV bias (default True).
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        return_all_tokens: If True, forward() returns all tokens (not just CLS).
        use_mean_pooling: If True, average patch tokens instead of CLS.
        init_values: Layer scale init values (None = no layer scale).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=None, return_all_tokens=False,
                 use_mean_pooling=False, init_values=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_mean_pooling:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
            )
            for i in range(depth)
        ])

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, pos_embed):
        """Handle position embedding interpolation for different resolutions."""
        num_patch_tokens = self.patch_embed.num_patches
        num_pos_tokens = pos_embed.shape[1] - 1
        if num_pos_tokens == num_patch_tokens:
            return pos_embed
        cls_token = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        orig_grid = int(num_pos_tokens ** 0.5)
        new_grid = int(num_patch_tokens ** 0.5)
        patch_pos = patch_pos.reshape(-1, orig_grid, orig_grid, x.shape[-1]).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat([cls_token, patch_pos], dim=1)

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + self.interpolate_pos_encoding(x, self.pos_embed)
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        if self.use_mean_pooling:
            x = self.fc_norm(x[:, 1:].mean(dim=1))
        else:
            x = self.norm(x)
        if self.return_all_tokens:
            return x
        return x[:, 0]

    def get_intermediate_layers(self, x, n=1):
        """Get outputs from the last n blocks.

        Args:
            x: Input tensor (B, C, H, W).
            n: Number of last blocks to extract.

        Returns:
            List of n tensors, each (B, num_patches+1, embed_dim).
        """
        x = self.prepare_tokens(x)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= len(self.blocks) - n:
                features.append(x)
        return features

    def get_last_selfattention(self, x):
        """Get attention map from the last block."""
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk.attn(x)
        return None
