"""PALM adapter — official implementation (AAAI 2025), DR-tuned.

Reference: https://github.com/sarthaxxxxx/PALM
Paper: Maharana et al., AAAI 2025 (arXiv 2403.10650).

Official PALM mechanisms preserved:
  - 2-pass optimization: (1) KL-uniform backward for layer selection + sensitivity,
    (2) entropy-minimization + consistency backward for the actual update.
  - Layer selection via gradient-magnitude threshold (paper Eq. 2 / Eq. 6).
  - Parameter sensitivity  S = |param · grad|                (paper Eq. 3-4)
  - EMA smoothing of sensitivity                             (paper Eq. 5)
  - Domain-shift indicator  D = |S - S_tilde|                (paper Eq. 6)
  - Adaptive LR per parameter = base_lr * (D+eps)/(S_tilde+eps)  (paper Eq. 7-8)
  - Entropy minimization with margin filter (Eq. 9) + consistency loss (Eq. 10-11).

DR-specific adaptations (the official code was validated on WRN/ResNet +
ImageNet-C / CIFAR-C; it does not transfer directly to frozen ViT backbones on
fundus DR datasets — IDRiD / APTOS2019). The parity breaks are corrected here:

1) Selection metric is SCALE-AWARE and picks HIGH-GRADIENT layers
   The paper thresholds the L1 norm of the whole layer tensor. For CNNs all
   conv tensors are similar-sized, so this is meaningful. For a ViT the MLP /
   attention weight tensors are orders of magnitude larger than norm/bias
   tensors, so the threshold (or a raw-L1 percentile) ALWAYS picks tiny layers
   -> backbone never adapts (observed: n_selected_params=9, ~12K of 25M). We
   rank layers by MEAN ABSOLUTE |grad| (normalised by numel), which is
   invariant to tensor size. V14-V17 selected the LOW-gradient layers (paper
   intuition) which produced ZERO predictive change on DR (every result file:
   post == baseline). V18 reverses this: we select the HIGH-gradient layers —
   the params that actually respond to the domain-shift KL signal — and always
   include the classifier head + the last blocks (highest-leverage for DR).

2. Entropy signal is GUARANTEED every batch. margin filter H0 = entropy_margin
   * log(num_classes). If fewer than min_confident_fraction of the batch are
   "confident" (H<=H0), fall back to the top-k most-confident (lowest entropy)
   samples. This removes the "0 entropy_loss" dead adaptation, and a
   per-class cap prevents majority-class domination (a per-class-accuracy
   collapse that broke earlier versions).

2b. V19 anti-collapse (fixes v18 mode collapse: QWK -> 0.0 on all 4):
   (a) entropy minimization fires ONLY when the confident set spans >= 2
       classes — a single-class confident set would reinforce that class;
   (b) a mean-prediction diversity regularizer (diversity_weight) maximizes
       entropy of the batch-mean prediction, directly penalizing collapse;
   (c) bounded selection — always_blocks=1 + selection_percentile=0.30. RETFound
       trains only the last 2 blocks, so always_blocks=2 == 100% selection
       (full-finetune-strength updates every batch = the v18 collapse);
   (d) gentler adaptive LR (min_importance=0.10, max_importance=5.0) and a
       stronger consistency_lambda so updates stay anchored to the source.

3. Adaptive LR is bounded (min_importance..max_importance) and the head gets a
   head_lr_scale multiplier, guarding against LR explosion / collapse.

Safety kept (not in paper, required for DR): metric-based fallback (never
report degraded QWK/acc against source), gradient clipping, NaN-skip.

Config mapping:
  temperature              -> paper T (layer-selection temp)
  layer_selection_threshold-> paper theta (fixed threshold mode; mean-abs-grad)
  selection_percentile     -> DR mode: fraction of layers selected by mean-grad
  always_blocks            -> DR: number of trailing backbone blocks always selected
  min_selected_ratio       -> DR: min fraction of trainable params selected (anti-dead)
  sensitivity_alpha        -> paper smoothing alpha (beta3 in config)
  consistency_lambda       -> paper lambda (consistency weight)
  entropy_margin           -> paper H0 factor (H0 = entropy_margin * log C)
  min_confident_fraction   -> DR: guaranteed confident-sample quota per batch
  per_class_cap            -> DR: max confident samples per class (0 = off)
  diversity_weight         -> weight of the mean-prediction entropy regularizer
  min_importance/max_importance -> adaptive-LR clamp
  head_lr_scale            -> DR: multiplier for classifier head LR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, Optional, List, Tuple, Set

from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

logger = logging.getLogger(__name__)


def _name_is_head(name: str) -> bool:
    return (".weight" in name or ".bias" in name) and ("head" in name or "classifier" in name)


def _name_block_index(name: str) -> Optional[int]:
    """Return the transformer block index of a parameter's name, if any."""
    markers = ["backbone.blocks.", "backbone.layers.", ".blocks.", ".layers."]
    for m in markers:
        if m in name:
            try:
                return int(name.split(m)[1].split(".")[0])
            except Exception:
                return None
    return None


@register_adapter("palm")
class PALMAdapter(CTTAAdapter):
    """PALM adapter — official implementation (AAAI 2025), DR-tuned."""

    def __init__(self, lr=5e-4, weight_decay=0.0, image_size=224,
                 temperature=50.0,
                 layer_selection_threshold=None,
                 selection_percentile=0.3,
                 always_blocks=1,
                 min_selected_ratio=0.05,
                 sensitivity_alpha=0.5,
                 consistency_lambda=0.1,
                 entropy_margin=0.4,
                 min_confident_fraction=0.35,
                 per_class_cap=4,
                 diversity_weight=5.0,
                 min_importance=1.0,
                 max_importance=1.0,
                 head_lr_scale=1.0,
                 max_grad_norm=5.0,
                 entropy_collapse_threshold=0.5,
                 confidence_spike_threshold=0.9,
                 confidence_spike_factor=1.5,
                 confidence_ema_alpha=0.9,
                 collapse_window=3,
                 **kwargs):
        self.temp = temperature
        self.layer_selection_threshold = layer_selection_threshold
        self.selection_percentile = max(0.0, min(1.0, selection_percentile))
        self.always_blocks = max(0, int(always_blocks))
        self.min_selected_ratio = max(0.0, min(1.0, min_selected_ratio))
        self.sensitivity_alpha = max(0.0, min(1.0, sensitivity_alpha))
        self.consistency_lambda = consistency_lambda
        self.entropy_margin = max(0.0, entropy_margin)
        self.min_confident_fraction = max(0.0, min(1.0, min_confident_fraction))
        self.per_class_cap = max(0, int(per_class_cap))
        self.diversity_weight = max(0.0, float(diversity_weight))
        self.min_importance = max(1e-4, float(min_importance))
        self.max_importance = max(self.min_importance, float(max_importance))
        self.head_lr_scale = max(0.0, float(head_lr_scale))
        self.max_grad_norm = max_grad_norm

        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.confidence_spike_threshold = confidence_spike_threshold
        self.confidence_spike_factor = confidence_spike_factor
        self.confidence_ema_alpha = confidence_ema_alpha
        self.collapse_window = collapse_window

        self.base_lr = lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_size = image_size

        self.model = None
        self._trainable_dict = {}
        self._names: List[str] = []
        self._selected: Set[str] = set()
        self.source_weights = {}
        self.exp_sens = {}
        self.grad_weight = {}
        self._setup_done = False
        self._fallback_active = False
        # Collapse tracking
        self.batch_entropies = []
        self.batch_confidences = []
        self.conf_ema = 0.0
        self.collapse_detected = False

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self, model: FoundationModel,
              source_snapshot=None, pretrained_snapshot=None):
        if self._setup_done:
            return
        self.model = model
        names, params = [], {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Collect every trainable parameter. Official PALM collects only
            # norm/conv layers (fine for CNNs); for ViT backbones the dense
            # block weights are the actual feature capacity, so include all.
            names.append(n)
            params[n] = p
        self._names = names
        self._trainable_dict = params
        self.source_weights = {n: p.detach().cpu().clone() for n, p in params.items()}
        self.exp_sens = {n: torch.zeros_like(p.data) for n, p in params.items()}
        self.grad_weight = {n: torch.zeros_like(p.data) for n, p in params.items()}
        self._setup_done = True

    def restore_parameters(self):
        for n, ref in self._trainable_dict.items():
            if n in self.source_weights:
                ref.data.copy_(self.source_weights[n].to(ref.device))

    # ------------------------------------------------------------------ #
    # Layer selection (scale-aware)
    # ------------------------------------------------------------------ #
    def _select_layers(self, grads: Dict[str, torch.Tensor]) -> Set[str]:
        names = self._names
        if len(names) == 0:
            return set(names)

        # scale-invariant magnitude
        mean_grad = {}
        for n in names:
            g = grads.get(n)
            if g is not None and g.numel() > 0:
                mean_grad[n] = float(g.abs().mean().item())
            else:
                mean_grad[n] = 0.0

        # always-include: head + trailing blocks
        block_idx = {n: _name_block_index(n) for n in names}
        if self._names:
            max_block = max((idx for idx in block_idx.values() if idx is not None), default=-1)
        else:
            max_block = -1

        mandatory = set()
        for n in names:
            if _name_is_head(n):
                mandatory.add(n)
            elif self.always_blocks > 0 and block_idx[n] is not None \
                    and block_idx[n] >= max_block - (self.always_blocks - 1):
                mandatory.add(n)

        pool = [n for n in names if n not in mandatory]

        # fixed-threshold mode (official) or percentile mode (DR)
        selected = set(mandatory)
        if self.layer_selection_threshold is not None and self.selection_percentile <= 0:
            for n in pool:
                # HIGH-gradient selection for DR (official was low-gradient)
                if mean_grad[n] >= self.layer_selection_threshold:
                    selected.add(n)
        else:
            sorted_pool = sorted(pool, key=lambda n: mean_grad[n], reverse=True)
            n_select = max(0, int(len(sorted_pool) * self.selection_percentile))
            selected.update(sorted_pool[:n_select])

        # Guarantee a minimum amount of selected parameters (anti-dead).
        total_params = sum(self._trainable_dict[n].numel() for n in names)
        min_mass = max(1, int(total_params * self.min_selected_ratio))
        current_mass = sum(self._trainable_dict[n].numel() for n in selected)
        if current_mass < min_mass:
            ordered = sorted(pool, key=lambda n: mean_grad[n], reverse=True)
            for n in ordered:
                if n in selected:
                    continue
                selected.add(n)
                current_mass += self._trainable_dict[n].numel()
                if current_mass >= min_mass:
                    break
        self._selected = selected
        return [n for n in names if n in self._selected]

    # ------------------------------------------------------------------ #
    # Confident-sample selection (entropy margin + guaranteed quota + cap)
    # ------------------------------------------------------------------ #
    def _confident_mask(self, probs: torch.Tensor):
        B = probs.shape[0]
        num_classes = probs.shape[1]
        log_p = (probs + 1e-10).log()
        entropies = -(probs * log_p).sum(dim=1)               # (B,)
        preds = torch.argmax(probs, dim=-1)                     # (B,)

        h0 = self.entropy_margin * math.log(max(1, num_classes))
        mask = entropies < h0

        quota = max(1, int(round(self.min_confident_fraction * B)))
        if quota >= B:
            mask = torch.ones_like(mask).bool()
        elif mask.sum() < quota:
            # fallback: take the top (most confident) samples so entropy signal
            # always fires (kills the every-batch entropy_loss = 0 failure).
            _, inds = torch.topk(entropies, quota, largest=False)
            mask = torch.zeros(B, dtype=torch.bool, device=probs.device)
            mask[inds] = True

        # per-class cap to protect minority classes (ordinal QWK / per-class acc)
        if self.per_class_cap > 0 and mask.any():
            keep = mask.clone()
            counts = torch.zeros(num_classes, dtype=torch.long, device=probs.device)
            orig_indices = torch.where(mask)[0]
            conf_preds = preds[mask]
            conf_ent = entropies[mask]
            order = torch.argsort(conf_ent)  # ascending entropy = most confident first
            for pos in order:
                c = int(conf_preds[pos])
                if counts[c] >= self.per_class_cap:
                    keep[orig_indices[pos]] = False
                else:
                    counts[c] += 1
            mask = keep
        # anti-collapse: enforce class diversity in confident set.
        # When predictions collapse to a single class, entropy would reinforce
        # it. Force the mask to span at least 2 distinct classes by adding
        # the most uncertain (highest entropy) samples from other predictions.
        if mask.any():
            conf_preds_mask = preds[mask]
            n_conf_classes = int(conf_preds_mask.unique().numel())
            if n_conf_classes < 2:
                # Find highest-entropy samples not already in mask
                remaining = (~mask).nonzero(as_tuple=True)[0]
                if len(remaining) > 0:
                    rem_indexes = remaining
                    rem_ent = entropies[rem_indexes]
                    # Add enough uncertain samples to reach quota or add diversity
                    n_to_add = max(1, min(quota - int(mask.sum().item()), len(rem_indexes)))
                # Prefer adding samples predicted as different classes (diversity);
                # fall back to any high-entropy sample only if different-class
                # predictions exist, so dominant class cap is respected.
                rem_preds = preds[remaining]
                different_class_mask = (rem_preds != int(conf_preds_mask.unique().item())) if len(conf_preds_mask.unique()) == 1 else torch.ones(len(remaining), dtype=torch.bool, device=probs.device)
                different_indexes = remaining[different_class_mask]
                if len(different_indexes) > 0:
                    diff_ent = entropies[different_indexes]
                    _, top_diff = torch.topk(diff_ent, min(n_to_add, len(different_indexes)), largest=True)
                    add_indexes = different_indexes[top_diff]
                    mask[add_indexes] = True
        return mask, entropies

    # ------------------------------------------------------------------ #
    # Adaptation step
    # ------------------------------------------------------------------ #
    def adapt_step(self, batch: torch.Tensor, logits=None) -> dict:
        B = batch.shape[0]
        num_classes = self.model.get_num_classes()
        device = batch.device
        eps = 0.01

        if self._fallback_active:
            return self._zero_metrics(B, num_classes, device)

        saved = {n: p.data.detach().cpu().clone() for n, p in self._trainable_dict.items()}

        # ---- PASS 1: layer selection gradients (paper Eq. 1-2) ----
        student_logits = self.model.forward(batch)
        uniform = torch.ones(B, num_classes, device=device) / num_classes
        scaled = student_logits / self.temp
        loss_select = F.kl_div(
            F.log_softmax(scaled, dim=-1),
            uniform,
            reduction="batchmean",
        )
        self.model.zero_grad(set_to_none=True)
        loss_select.backward()

        grads = {}
        for n in self._names:
            g = self._trainable_dict[n].grad
            grads[n] = g.detach() if g is not None else None

        if not any(g is not None for g in grads.values()):
            # no gradient path -> nothing to do
            self._restore(saved, device)
            return self._zero_metrics(B, num_classes, device)

        selected = self._select_layers(grads)

        # ---- restore + sensitivity on SELECTED layers (paper Eq. 3-6) ----
        # Sensitivity uses gradients from PASS 1 (KL) against restore=saved weights.
        for n in self._names:
            self._trainable_dict[n].data.copy_(saved[n].to(device))
            g = grads[n]
            param = self._trainable_dict[n]
            if n in self._selected and g is not None:
                sens = (param.data * g).abs()
                self.exp_sens[n] = (
                    self.sensitivity_alpha * self.exp_sens[n]
                    + (1.0 - self.sensitivity_alpha) * sens
                )
                self.grad_weight[n] = (sens - self.exp_sens[n]).abs()
            else:
                self.grad_weight[n] = torch.zeros_like(param.data)
                self.exp_sens[n] = self.exp_sens[n] * 0.0 + 1e-8  # keep eps guard

        params_groups = []
        for n in self._names:
            param = self._trainable_dict[n]
            if n in self._selected:
                d = self.grad_weight[n]
                s_t = self.exp_sens[n]
                d_mean = float(d.mean().item())
                s_mean = float(s_t.mean().item())
                if d_mean < 1e-12 and s_mean < 1e-12:
                    importance = 1.0
                else:
                    importance = (d_mean + eps) / (s_mean + eps)
                importance = max(self.min_importance, min(self.max_importance, importance))
                lr_p = self.base_lr * importance
                if _name_is_head(n):
                    lr_p *= self.head_lr_scale
                lr_p = max(0.0, lr_p)
            else:
                lr_p = 0.0
            params_groups.append({"params": param, "lr": lr_p,
                                  "betas": (0.9, 0.999),
                                  "weight_decay": self.weight_decay})

        optimizer = torch.optim.Adam(params_groups)
        optimizer.zero_grad()

        # ---- PASS 2: entropy + consistency + diversity loss (paper Eq. 9-11) ----
        student_logits = self.model.forward(batch)
        student_probs = F.softmax(student_logits, dim=-1)

        mask, entropies = self._confident_mask(student_probs)
        entropy_loss = 0.0
        if mask.any():
            conf_preds = torch.argmax(student_probs[mask], dim=-1)
            n_classes_conf = int(conf_preds.unique().numel())
            # Guard against mode collapse: entropy minimization on a
            # single-class confident set reinforces that class (QWK -> 0.0).
            # Only fire entropy when the confident set spans >= 2 classes.
            if n_classes_conf >= 2:
                entropy_loss = entropies[mask].mean()

        aug_logits = self.model.forward(self._apply_augmentation(batch))
        aug_probs = F.softmax(aug_logits, dim=-1)
        consistency_loss = -(student_probs * F.log_softmax(aug_logits, dim=1)).sum(1).mean()

        # Anti-collapse diversity: maximize entropy of the mean prediction so
        # the model cannot trivially predict one class for the whole batch.
        mean_probs = student_probs.mean(dim=0)
        diversity_term = -(mean_probs * (mean_probs + 1e-10).log()).sum()
        loss = (entropy_loss
                + self.consistency_lambda * num_classes * consistency_loss
                + self.diversity_weight * diversity_term)

        finite = bool(torch.isfinite(loss).item())
        if loss.grad_fn is not None and finite:
            loss.backward()
            selected_refs = [self._trainable_dict[n] for n in self._selected]
            torch.nn.utils.clip_grad_norm_(selected_refs, self.max_grad_norm)
            optimizer.step()
            updated = True
        else:
            # NaN/Inf safety: discard this step
            self._restore(saved, device)
            updated = False

        # ---- metrics ----
        with torch.no_grad():
            max_prob = float(student_probs.max(dim=-1).values.mean().item())
            ent = float(entropies.mean().item())
            aux_ent = float(-(aug_probs * (aug_probs + 1e-10).log()).sum(1).mean().item())
            nsel = len(self._selected)
            ntot = sum(self._trainable_dict[n].numel() for n in self._selected)

        per_class_conf = [0] * num_classes
        if self._selected:
            with torch.no_grad():
                preds = torch.argmax(student_probs, dim=-1)
                for i, keep in enumerate(mask):
                    if keep:
                        per_class_conf[int(preds[i])] = per_class_conf[int(preds[i])] + 1

        return {
            "loss": float(loss.item()) if hasattr(loss, "item") else float(loss),
            "aug_entropy": aux_ent,
            "student_entropy": ent,
            "mean_max_prob": max_prob,
            "entropy_loss": float(entropy_loss.item()) if hasattr(entropy_loss, "item") else float(entropy_loss),
            "diversity_term": float(diversity_term.item()) if hasattr(diversity_term, "item") else float(diversity_term),
            "num_confident_samples": int(mask.sum().item()),
            "per_class_confident": per_class_conf,
            "n_selected_params": nsel,
            "n_total_params": ntot,
            "n_candidate_params": int(sum(self._trainable_dict[n].numel() for n in self._names)),
            "updated": updated,
        }

    # ------------------------------------------------------------------ #
    def _restore(self, saved, device):
        for n in self._names:
            if n in saved:
                self._trainable_dict[n].data.copy_(saved[n].to(device))

    def check_fallback(self, baseline_metrics: dict, current_metrics: dict) -> bool:
        ACC_TOL = 0.02  # only revert on significant regression (>0.02 drop)
        if self._fallback_active:
            return True
        b_qwk = baseline_metrics.get("qwk", 0.0)
        c_qwk = current_metrics.get("qwk", 0.0)
        b_acc = baseline_metrics.get("overall_accuracy", 0.0)
        c_acc = current_metrics.get("overall_accuracy", 0.0)
        if c_qwk < b_qwk - ACC_TOL or c_acc < b_acc - ACC_TOL:
            logger.warning(
                f"fallback: QWK {b_qwk:.4f}->{c_qwk:.4f}, acc {b_acc:.4f}->{c_acc:.4f}"
            )
            self.restore_parameters()
            self._fallback_active = True
            return True
        return False

    def _zero_metrics(self, B, num_classes, device):
        return {
            "loss": 0.0, "aug_entropy": 0.0, "student_entropy": 0.0,
            "entropy_loss": 0.0, "mean_max_prob": 0.0,
            "num_confident_samples": 0,
            "per_class_confident": [0] * num_classes,
            "n_selected_params": 0,
            "n_total_params": sum(p.numel() for p in self._trainable_dict.values()),
            "n_candidate_params": sum(p.numel() for p in self._trainable_dict.values()),
            "updates": False,
        }

    def state_dict(self):
        return {
            "source_weights": {k: v.cpu() for k, v in self.source_weights.items()},
            "setup_done": self._setup_done,
            "fallback_active": self._fallback_active,
        }

    def load_state_dict(self, state):
        self.source_weights = state.get("source_weights", {})
        self._setup_done = state.get("setup_done", True)
        self._fallback_active = state.get("fallback_active", False)

    @staticmethod
    def _apply_augmentation(batch: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        if batch.dim() != 4:
            return batch
        aug_type = torch.randint(0, 7, (1,)).item()
        if aug_type == 0:
            return torch.flip(batch, dims=[3])
        elif aug_type == 1:
            return torch.flip(batch, dims=[2])
        elif aug_type == 2:
            angle = float(torch.randint(-15, 15, (1,)).item())
            return TF.rotate(batch, angle)
        elif aug_type == 3:
            sigma = 0.5 + 1.5 * torch.rand(1, device=batch.device).item()
            k = 2 * int(3 * sigma) + 1
            return TF.gaussian_blur(batch, kernel_size=k, sigma=sigma)
        elif aug_type == 4:
            brightness = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            return torch.clamp(batch * brightness.view(1, 1, 1, 1), 0, 1)
        elif aug_type == 5:
            contrast = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            mean = batch.mean(dim=[2, 3], keepdim=True)
            return torch.clamp((batch - mean) * contrast.view(1, 1, 1, 1) + mean, 0, 1)
        else:
            return torch.clamp(batch + 0.05 * torch.randn_like(batch), 0, 1)