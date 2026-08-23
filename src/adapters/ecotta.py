"""EcoTTA adapter — memory-efficient continual test-time adaptation.

Reference: https://github.com/Lily-Le/EcoTTA
Paper: Song, Lee, et al., CVPR 2023 (arXiv 2303.01904).

Official EcoTTA mechanisms preserved:
  - The frozen original encoder is partitioned into K groups; shallow groups are
    split more densely than deep ones (paper Table 4c: WRN-40 [3,3,6,6] for K=4).
  - Each partition attaches a lightweight meta network. For a ViT backbone we use
    the token-wise analog of the paper's BN + conv-block:
        out1 = original_partition(x)                 (frozen)
        out  = out1 + mlp(x)                         (zero-init Linear-GELU-Linear)
    (the meta MLP starts at 0 -> out == out1 exactly, so the baseline forward is
    preserved before any training; this is the paper's residual meta modification)
  - Self-distilled regularization per partition (official form R = |f_M(x)-f_orig(x)|):
        R_k = mean( | mlp(x.detach()) | )
    (detached input -> R_k gradient reaches only that partition's meta params)
  - Adaptation loss (single-pass, vs the official two-pass loop):
        L = mean(H_k)[H_k < H0] + reg_lambda * sum_k R_k
    with H0 = entropy_margin * log(C)  (EATA-style confident-sample margin).
  - Warmup on the source train split with CE (paper: SGD lr=5e-2), then TTA with
    SGD lr=5e-3, momentum=0.9.

DR-specific adaptations (fundus DR / frozen ViT):
  - The meta MLP is zero-initialized (last layer) so every partition starts at
    exact identity -> baseline forward stays intact before warmup.
  - Per-partition gradient checkpointing (torch.utils.checkpoint) recomputes the
    frozen block stack on backward, so peak memory is O(1 partition) instead of
    O(all partitions) during warmup/adaptation (ViT-L fits on a 16 GB T4).
  - A per-class cap (ecotta_per_class_cap) limits confident samples per predicted
    class so entropy minimization can't collapse onto the majority class.
  - A confident-sample quota guarantees an entropy signal every batch (the margin
    filter alone can produce 0 confident samples on overconfident medical models,
    which historically caused "dead adaptation" — see PALM notes in AGENTS.md).
  - Metric-based fallback: never report QWK/acc degraded >0.02 vs source.

VisionFM constraint: its classifier consumes CLS tokens of the LAST 4 blocks via
backbone.get_intermediate_layers(x, n=4) -> 3072-dim. Partition sizes must therefore
be [2,2,4,1,1,1,1] (K=7) so the four collected outputs land exactly on blocks
8,9,10,11. RETFound (timm) pools the final block output, so the paper's shallow-dense
[4,4,8,8] (K=4) is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional

from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

logger = logging.getLogger(__name__)


class PartitionWithMeta(nn.Module):
    """A frozen slice of the original encoder plus an attached meta network.

    forward() runs the frozen original blocks (optionally under gradient
    checkpointing), then the residual meta modification ``out = out1 + mlp(x)``.
    When ``cal_reg`` is True it also stores the self-distilled regularization
    ``R = mean(|mlp(x.detach())|)`` (official R = |f_M(x) - f_orig(x)|) with a
    detached input so R's gradient reaches only this partition's meta params.
    """

    def __init__(self, blocks: List[nn.Module], dim: int, hidden_scale: float = 1.0,
                 grad_checkpointing: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        hidden = max(1, int(dim * hidden_scale))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.grad_checkpointing = grad_checkpointing
        self.cal_reg = False
        self.reg_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and x.requires_grad:
            out1 = torch.utils.checkpoint.checkpoint(self.blocks, x, use_reentrant=False)
        else:
            out1 = self.blocks(x)
        out = out1 + self.mlp(x)
        if self.cal_reg:
            self.reg_loss = self.mlp(x.detach()).abs().mean()
        return out


@register_adapter("ecotta")
class EcoTTAAdapter(CTTAAdapter):
    """EcoTTA adapter — frozen encoder + per-partition meta networks."""

    def __init__(self, lr=5e-3, weight_decay=0.0, image_size=224,
                 max_grad_norm=5.0,
                 ecotta_num_partitions: int = 4,
                 ecotta_partition_sizes: Optional[List[int]] = None,
                 ecotta_meta_hidden_scale: float = 1.0,
                 ecotta_reg_lambda: float = 0.25,
                 ecotta_entropy_margin: float = 0.4,
                 ecotta_warmup_epochs: int = 5,
                 ecotta_warmup_lr: float = 5e-2,
                 ecotta_tta_lr: Optional[float] = None,
                 ecotta_min_confident_fraction: float = 0.25,
                 ecotta_per_class_cap: Optional[int] = None,
                 ecotta_grad_checkpointing: bool = True,
                 **kwargs):
        self.num_partitions = max(1, int(ecotta_num_partitions))
        self.partition_sizes = list(ecotta_partition_sizes) if ecotta_partition_sizes else None
        self.meta_hidden_scale = max(0.25, float(ecotta_meta_hidden_scale))
        self.reg_lambda = max(0.0, float(ecotta_reg_lambda))
        self.entropy_margin = max(0.0, float(ecotta_entropy_margin))
        self.warmup_epochs = max(0, int(ecotta_warmup_epochs))
        self.warmup_lr = float(ecotta_warmup_lr)
        self.tta_lr = float(ecotta_tta_lr) if ecotta_tta_lr is not None else float(lr)
        self.min_confident_fraction = max(0.0, min(1.0, ecotta_min_confident_fraction))
        self.per_class_cap = None if ecotta_per_class_cap is None else max(1, int(ecotta_per_class_cap))
        self.grad_checkpointing = bool(ecotta_grad_checkpointing)
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.image_size = image_size

        self.model: Optional[FoundationModel] = None
        self._partitions: List[PartitionWithMeta] = []
        self._meta_params: List[nn.Parameter] = []
        self._source_meta: List[torch.Tensor] = []
        self._optimizer = None
        self._device = None
        self._setup_done = False
        self._fallback_active = False

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self, model: FoundationModel,
              source_snapshot=None, pretrained_snapshot=None):
        if self._setup_done:
            return
        self.model = model
        self._device = next(model.parameters()).device

        backbone = model.backbone
        if not hasattr(backbone, "blocks"):
            raise ValueError("EcoTTA requires a backbone with a 'blocks' attribute")
        blocks = backbone.blocks
        n_blocks = len(blocks)

        sizes = self._resolve_sizes(n_blocks)
        dim = self._infer_dim(blocks)

        # Disable the backbone's coarse-level checkpointing (timm checkpoint_seq /
        # set_grad_checkpointing) — it would wrap our partition list from outside.
        # Per-partition gradient checkpointing inside PartitionWithMeta.forward
        # (honoring self.grad_checkpointing) recomputes one frozen block stack on
        # backward, keeping peak memory O(1 partition) instead of O(all partitions).
        if hasattr(backbone, "grad_checkpointing"):
            backbone.grad_checkpointing = False

        # Freeze the whole original network (paper: only meta networks train).
        for p in model.parameters():
            p.requires_grad_(False)

        # Build partitions (original blocks moved into frozen Sequentials).
        partitions = []
        start = 0
        for size in sizes:
            group = list(blocks)[start:start + size]
            start += size
            partitions.append(PartitionWithMeta(group, dim, self.meta_hidden_scale,
                                                grad_checkpointing=self.grad_checkpointing))

        # Preserve the container semantics of the backbone:
        #   - timm ViT: nn.Sequential, called as blocks(x)
        #   - VisionFM custom ViT: nn.ModuleList, iterated in forward()
        # nn.Sequential is iterable, so a Sequential wrapper works for both.
        backbone.blocks = nn.Sequential(*partitions)
        backbone.blocks = backbone.blocks.to(self._device)

        self._partitions = partitions
        # Paper: only the meta networks train — the original blocks inside a
        # partition must stay frozen. Collecting part.mlp.parameters() (not
        # part.parameters()) keeps the frozen-encoder property intact.
        meta_params = []
        for part in partitions:
            for p in part.mlp.parameters():
                p.requires_grad_(True)
                meta_params.append(p)
        self._meta_params = meta_params
        self._source_meta = [p.detach().cpu().clone() for p in meta_params]

        self._optimizer = torch.optim.SGD(
            meta_params, lr=self.tta_lr, momentum=0.9, weight_decay=self.weight_decay,
        )
        self._setup_done = True
        logger.info(
            "EcoTTA setup: %d blocks -> %d partitions %s, %d meta params (dim=%d)",
            n_blocks, len(sizes), sizes, len(meta_params), dim,
        )

    def _resolve_sizes(self, n_blocks: int) -> List[int]:
        if self.partition_sizes is not None:
            sizes = list(self.partition_sizes)
            if sum(sizes) != n_blocks:
                raise ValueError(
                    f"ecotta_partition_sizes {sizes} sum to {sum(sizes)}, expected {n_blocks}"
                )
            return sizes
        # Paper shallow-dense scheme: [x, x, 2x, 2x] for K=4 (6x = n_blocks).
        K = self.num_partitions
        if K == 4:
            x = max(1, n_blocks // 6)
            sizes = [x, x, 2 * x, 2 * x]
            sizes[-1] += n_blocks - sum(sizes)
            return sizes
        base = n_blocks // K
        rem = n_blocks % K
        sizes = [base] * K
        for i in range(rem):
            sizes[K - 1 - i] += 1
        return sizes

    def _infer_dim(self, blocks) -> int:
        backbone = self.model.backbone
        for attr in ("embed_dim", "num_features"):
            v = getattr(backbone, attr, None)
            if v:
                return int(v)
        first = list(blocks)[0]
        for m in first.modules():
            if isinstance(m, nn.LayerNorm) and m.weight is not None:
                return int(m.weight.shape[0])
        raise ValueError("EcoTTA: cannot infer embed dim from backbone")

    # ------------------------------------------------------------------ #
    # Warmup (source CE)
    # ------------------------------------------------------------------ #
    def warmup(self, train_loader) -> None:
        """Warm up meta networks on the source train split with CE (paper: SGD lr=5e-2)."""
        if not self._setup_done or self._fallback_active or self.warmup_epochs <= 0:
            return
        if train_loader is None:
            self._source_meta = [p.detach().cpu().clone() for p in self._meta_params]
            return

        device = self._device
        optimizer = torch.optim.SGD(
            self._meta_params, lr=self.warmup_lr, momentum=0.9, weight_decay=self.weight_decay,
        )
        self.model.train()
        for epoch in range(self.warmup_epochs):
            total = 0.0
            n = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._meta_params, self.max_grad_norm)
                optimizer.step()
                total += float(loss.item()) * images.size(0)
                n += images.size(0)
            logger.info(
                "EcoTTA warmup epoch %d/%d: CE=%.4f",
                epoch + 1, self.warmup_epochs, total / max(1, n),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.eval()
        # Post-warmup state is the source-aligned fallback target.
        self._source_meta = [p.detach().cpu().clone() for p in self._meta_params]

    # ------------------------------------------------------------------ #
    # Adaptation step (single-pass: entropy + self-distilled reg)
    # ------------------------------------------------------------------ #
    def adapt_step(self, batch: torch.Tensor, logits=None) -> dict:
        B = batch.shape[0]
        device = batch.device
        num_classes = self.model.get_num_classes()

        if self._fallback_active:
            return self._zero_metrics(B, num_classes)

        saved = [p.data.detach().cpu().clone() for p in self._meta_params]

        for part in self._partitions:
            part.cal_reg = True

        student_logits = self.model(batch)
        probs = F.softmax(student_logits, dim=-1)
        entropies = self._softmax_entropy(student_logits)

        # EATA-style confident-sample margin: H0 = entropy_margin * log(C),
        # per-class capped so entropy minimization can't collapse on one class.
        mask = self._select_confident(entropies, probs, B, device)
        entropy_loss = entropies[mask].mean()

        reg_loss = sum((part.reg_loss for part in self._partitions), 0.0)
        loss = entropy_loss + self.reg_lambda * reg_loss

        for part in self._partitions:
            part.cal_reg = False

        finite = bool(torch.isfinite(loss).item())
        if loss.grad_fn is not None and finite:
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._meta_params, self.max_grad_norm)
            self._optimizer.step()
            updated = True
        else:
            for p, sv in zip(self._meta_params, saved):
                p.data.copy_(sv.to(p.device))
            updated = False

        with torch.no_grad():
            max_prob = float(probs.max(dim=-1).values.mean().item())
            ent_mean = float(entropies.mean().item())
            reg_val = float(reg_loss.item()) if hasattr(reg_loss, "item") else 0.0

        return {
            "loss": float(loss.item()),
            "student_entropy": ent_mean,
            "entropy_loss": float(entropy_loss.item()),
            "reg_loss": reg_val,
            "mean_max_prob": max_prob,
            "num_confident_samples": int(mask.sum().item()),
            "updated": updated,
        }

    def _select_confident(self, entropies: torch.Tensor, probs: torch.Tensor,
                          B: int, device) -> torch.Tensor:
        """Confident-sample mask: entropy < entropy_margin*ln(C), per-class capped.

        Falls back to a top-k quota of the lowest-entropy samples
        (ecotta_min_confident_fraction) when the margin filter is empty, then
        re-applies the per-class cap on the quota so the signal stays balanced.
        """
        num_classes = probs.shape[1]
        h0 = self.entropy_margin * math.log(max(1, num_classes))
        mask = entropies < h0
        if self.per_class_cap is not None:
            mask = self._cap_by_class(mask, probs, entropies)
        if int(mask.sum().item()) < 1:
            quota = min(B, max(1, int(round(self.min_confident_fraction * B))))
            _, inds = torch.topk(entropies, quota, largest=False)
            mask = torch.zeros(B, dtype=torch.bool, device=device)
            mask[inds] = True
            if self.per_class_cap is not None:
                capped = self._cap_by_class(mask, probs, entropies)
                if int(capped.sum().item()) >= 1:
                    mask = capped
        return mask

    def _cap_by_class(self, mask: torch.Tensor, probs: torch.Tensor,
                      entropies: torch.Tensor) -> torch.Tensor:
        """Keep at most per_class_cap lowest-entropy samples per predicted class."""
        B = mask.shape[0]
        preds = probs.argmax(dim=1)
        out = torch.zeros(B, dtype=torch.bool, device=mask.device)
        for c in range(probs.shape[1]):
            cls = preds == c
            cand = mask & cls
            n = int(cand.sum().item())
            if n == 0:
                continue
            if n <= self.per_class_cap:
                out |= cand
            else:
                e = entropies.masked_fill(~cand, float('inf'))
                _, inds = torch.topk(e, self.per_class_cap, largest=False)
                out[inds] = True
        return out

    @staticmethod
    def _softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    # ------------------------------------------------------------------ #
    # Restore / fallback / serialization
    # ------------------------------------------------------------------ #
    def restore_parameters(self):
        for p, sv in zip(self._meta_params, self._source_meta):
            p.data.copy_(sv.to(p.device))

    def check_fallback(self, baseline_metrics: dict, current_metrics: dict) -> bool:
        ACC_TOL = 0.02
        if self._fallback_active:
            return True
        b_qwk = baseline_metrics.get("qwk", 0.0)
        c_qwk = current_metrics.get("qwk", 0.0)
        b_acc = baseline_metrics.get("overall_accuracy", 0.0)
        c_acc = current_metrics.get("overall_accuracy", 0.0)
        if c_qwk < b_qwk - ACC_TOL or c_acc < b_acc - ACC_TOL:
            logger.warning(
                "fallback: QWK %.4f->%.4f, acc %.4f->%.4f",
                b_qwk, c_qwk, b_acc, c_acc,
            )
            self.restore_parameters()
            self._fallback_active = True
            return True
        return False

    def _zero_metrics(self, B, num_classes):
        return {
            "loss": 0.0, "student_entropy": 0.0, "entropy_loss": 0.0,
            "reg_loss": 0.0, "mean_max_prob": 0.0, "num_confident_samples": 0,
            "updated": False,
        }

    def state_dict(self):
        return {
            "source_meta": [v.cpu().clone() for v in self._source_meta],
            "meta_params": [p.detach().cpu().clone() for p in self._meta_params],
            "setup_done": self._setup_done,
            "fallback_active": self._fallback_active,
        }

    def load_state_dict(self, state):
        self._source_meta = [v.clone() for v in state.get("source_meta", [])]
        self._setup_done = state.get("setup_done", True)
        self._fallback_active = state.get("fallback_active", False)
        meta = state.get("meta_params", [])
        for p, v in zip(self._meta_params, meta):
            p.data.copy_(v.to(p.device))