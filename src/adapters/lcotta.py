"""LCoTTA adapter — lifelong continual TTA via online learning in a tracked
low-dimensional subspace.

Reference: https://github.com/ThunderDavid/LCoTTA
Paper: Duan et al., "Lifelong Test-Time Adaptation via Online Learning in
Tracked Low-Dimensional Subspace", NeurIPS 2025.

Official LCoTTA mechanisms preserved (official ``methods/subspace_plus.py``):
  - Only the affine parameters of normalization layers train (LayerNorm /
    GroupNorm / BatchNorm weight+bias; <1% of params for ViTs).
  - Entropy-based sample filtering (ESF, EATA-style): keep samples with
    H < e_margin, down-weight by coeff = exp(e_margin - H).
  - Cosine-redundancy filter vs an EMA of the running mean prediction
    (|cos| < threshold removes redundant samples), as in the official code.
  - Online subspace tracking: every ``lcotta_sample_interval`` batches the raw
    gradient vector of the norm params is appended to a FIFO queue of length
    ``lcotta_queue_length``; PCA (SVD) of the centered queue gives an r-dim
    principal subspace P; the update uses the projected gradient
    g_tilde = P^T (P g), suppressing scattered entropy-deceptive (ED)
    gradients while keeping the correlated entropy-truthful (ET) directions.
  - Until the queue holds >= r gradients, updates fall back to plain SGD
    (official behavior during the first batches).

DR-specific adaptations (fundus DR / ViT foundation models):
  - e_margin = lcotta_entropy_margin * log(C) (same convention as PALM/E-CoTTA
    configs; official hardcodes 0.4*ln(1000) for ImageNet).
  - A confident-sample quota (lcotta_min_confident_fraction) guarantees an
    entropy signal every batch — the margin filter alone can produce 0
    confident samples on overconfident medical models ("dead adaptation",
    see AGENTS.md); the official code has no such fallback.
  - Optional per-class cap (lcotta_per_class_cap) prevents collapse onto the
    majority DR grade.
  - Metric-based fallback (check_fallback): never report QWK/acc degraded
    >0.02 vs source (same safety net as PALM/E-CoTTA).
  - Gradient clipping on the projected update (max_grad_norm) — the official
    code does not clip, but DR runs share the framework-wide safety net.
  - Optional classifier-head adaptation (lcotta_adapt_head +
    lcotta_head_lr_multiplier, separate SGD param group): the prototype head
    drives QWK directly and CoTTA/PALM/E-CoTTA all adapt it; the official
    LCoTTA updates norm layers only.
  - The confident-sample quota (lcotta_min_confident_fraction) is enforced on
    EVERY batch (margin set UNION top-quota lowest-entropy), not only as an
    empty-mask fallback: overconfident medical models can sit entirely below
    the margin with ~zero entropy, giving vanishing gradients and zero
    adaptation (observed in first DR runs: loss flat at 0.63-0.66 across all
    batches).
  - Optional Adam optimizer (lcotta_optimizer="adam"): SGD+momentum on LN
    affine params makes near-zero progress at DR's tiny gradient scales in
    the 7-23 single steps a test set allows; Adam normalizes step sizes.
  - Optional prior alignment (lcotta_prior_alignment_weight, SAR-style APU):
    maximizes the entropy of the batch mean prediction. V3 DR runs showed
    entropy minimization drains borderline class-2 (moderate DR) mass into
    adjacent grades — QWK rises while class-2 accuracy collapses. This term
    stabilizes the marginal class distribution.
  - The confident-set quota top-up is CLASS-BALANCED (round-robin over
    predicted classes by ascending entropy) so minority grades always anchor
    themselves instead of being crowded out by majority-grade samples.
  - Optional head trust region (lcotta_head_anchor_weight): after every
    optimizer step, head weights are pulled a fixed fraction (the weight,
    clamped to 0.9) back toward their setup-time prototype values.
    V4 showed prior alignment cannot protect class 2 (marginal prediction
    entropy is already ~max, so its gradient is ~0) — the erosion comes from
    head boundary sharpening, which Adam makes immune to L2 loss penalties.
    The post-step geometric pull bounds drift for any optimizer so gains
    flow through LN feature adaptation while per-class accuracies stay
    protected.
  - Queue sampling interval / length / rank are tuned per dataset x model in
    configs/lcotta/*.yaml: DR test sets are tens of batches (not ImageNet-C's
    millions), so the official interval=50..100 would never fill the queue,
    and rank/queue must be sized to the actual batch count of each dataset.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

logger = logging.getLogger(__name__)


@register_adapter("lcotta")
class LCoTTAAdapter(CTTAAdapter):
    """LCoTTA adapter — norm-layer entropy minimization with online
    subspace-projected updates."""

    def __init__(self, lr=1.5e-4, weight_decay=0.0, image_size=224,
                 max_grad_norm=5.0,
                 lcotta_subspace_dim: int = 10,
                 lcotta_queue_length: int = 30,
                 lcotta_sample_interval: int = 2,
                 lcotta_entropy_margin: float = 0.4,
                 lcotta_momentum: float = 0.9,
                 lcotta_cosine_filter: bool = True,
                 lcotta_cosine_threshold: float = 0.05,
                 lcotta_prob_ema: float = 0.9,
                 lcotta_min_confident_fraction: float = 0.25,
                 lcotta_per_class_cap: Optional[int] = None,
                 lcotta_adapt_head: bool = False,
                 lcotta_head_lr_multiplier: float = 10.0,
                 lcotta_optimizer: str = "sgd",
                 lcotta_prior_alignment_weight: float = 0.0,
                 lcotta_head_anchor_weight: float = 0.0,
                 **kwargs):
        self.subspace_dim = max(1, int(lcotta_subspace_dim))
        self.queue_length = max(self.subspace_dim, int(lcotta_queue_length))
        self.sample_interval = max(1, int(lcotta_sample_interval))
        self.entropy_margin = max(0.0, float(lcotta_entropy_margin))
        self.momentum = float(lcotta_momentum)
        self.cosine_filter = bool(lcotta_cosine_filter)
        self.cosine_threshold = float(lcotta_cosine_threshold)
        self.prob_ema = min(1.0, max(0.0, float(lcotta_prob_ema)))
        self.min_confident_fraction = max(0.0, min(1.0, lcotta_min_confident_fraction))
        self.per_class_cap = None if lcotta_per_class_cap is None else max(1, int(lcotta_per_class_cap))
        self.adapt_head = bool(lcotta_adapt_head)
        self.head_lr_multiplier = max(1.0, float(lcotta_head_lr_multiplier))
        self.optimizer_name = str(lcotta_optimizer).lower()
        self.prior_align_weight = max(0.0, float(lcotta_prior_alignment_weight))
        self.head_anchor_weight = max(0.0, float(lcotta_head_anchor_weight))
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.lr = float(lr)
        # base lr is passed by the runner via `lr`

        self.model: Optional[FoundationModel] = None
        self._params: List[nn.Parameter] = []
        self._param_shapes: List[torch.Size] = []
        self._numel = 0
        self._source_weights: List[torch.Tensor] = []
        self._head_params: List[nn.Parameter] = []      # tail of _params when adapt_head
        self._head_source: List[torch.Tensor] = []       # device copies for the anchor
        self._optimizer = None
        self._device = None
        self._setup_done = False
        self._fallback_active = False

        self._grad_queue: List[torch.Tensor] = []   # CPU float32 vectors (n,)
        self._batch_num = 0
        self._current_model_probs: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self, model: FoundationModel,
              source_snapshot=None, pretrained_snapshot=None):
        if self._setup_done:
            return
        self.model = model
        self._device = next(model.parameters()).device

        # Freeze everything, then re-enable only normalization affine params
        # (official collect_params/configure_model).
        for p in model.parameters():
            p.requires_grad_(False)

        params = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for pn, p in m.named_parameters():
                    if pn in ("weight", "bias"):
                        p.requires_grad_(True)
                        params.append(p)
        if not params:
            raise ValueError("LCoTTA: no normalization layers found in the model")

        # Optional classifier-head adaptation (DR-specific): the prototype head
        # drives QWK directly; CoTTA/PALM/E-CoTTA all adapt it. Appended AFTER
        # the norm params so the gradient-vector layout stays deterministic.
        n_norm = len(params)
        if self.adapt_head:
            clf = getattr(model, "classifier", None)
            if clf is not None:
                for p in clf.parameters():
                    p.requires_grad_(True)
                    params.append(p)

        self._params = params
        self._param_shapes = [p.shape for p in params]
        self._numel = sum(p.numel() for p in params)
        self._source_weights = [p.detach().cpu().clone() for p in params]

        param_groups = [{"params": params[:n_norm], "lr": self.lr}]
        if self.adapt_head and len(params) > n_norm:
            param_groups.append(
                {"params": params[n_norm:], "lr": self.lr * self.head_lr_multiplier})
            if self.head_anchor_weight > 0:
                self._head_params = params[n_norm:]
                self._head_source = [p.detach().clone() for p in params[n_norm:]]
        if self.optimizer_name == "adam":
            # Adam normalizes the per-parameter step: entropy gradients on
            # overconfident medical models are tiny and SGD+momentum makes
            # near-zero progress in the few steps a DR test set allows.
            self._optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        else:
            self._optimizer = torch.optim.SGD(
                param_groups, lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self._setup_done = True
        logger.info(
            "LCoTTA setup: %d norm-layer tensors (+%d head tensors), %d trainable params, "
            "subspace r=%d, queue=%d, interval=%d",
            n_norm, len(params) - n_norm, self._numel, self.subspace_dim,
            self.queue_length, self.sample_interval,
        )

    # ------------------------------------------------------------------ #
    # Adaptation step
    # ------------------------------------------------------------------ #
    def adapt_step(self, batch: torch.Tensor, logits=None) -> dict:
        B = batch.shape[0]
        num_classes = self.model.get_num_classes()

        if self._fallback_active:
            return self._zero_metrics(B)

        saved = [p.data.detach().cpu().clone() for p in self._params]

        student_logits = self.model(batch)
        probs = F.softmax(student_logits, dim=-1)
        entropies = self._softmax_entropy(student_logits)

        base_mask = self._select_confident(entropies, probs, B)
        mask = base_mask

        # Cosine-redundancy filter vs running mean prediction (official ESF).
        # DR safety net: if the redundancy filter empties the batch (happens
        # with near-uniform medical-model predictions), keep the margin set so
        # the entropy signal never dies.
        filtered_probs = probs[mask]
        if self.cosine_filter and self._current_model_probs is not None \
                and filtered_probs.size(0) > 0:
            ref = self._current_model_probs.to(filtered_probs.device)
            cos = F.cosine_similarity(
                ref.unsqueeze(0), filtered_probs, dim=1)
            keep = cos.abs() < self.cosine_threshold
            new_mask = torch.zeros_like(mask)
            new_mask[mask] = keep
            if int(new_mask.sum().item()) >= 1:
                mask = new_mask
                filtered_probs = probs[mask]

        updated_probs = self._update_model_probs(filtered_probs)

        h0 = self.entropy_margin * math.log(max(1, num_classes))
        ent = entropies[mask]
        if ent.size(0) == 0:
            self._batch_num += 1
            self._current_model_probs = updated_probs
            return self._zero_metrics(B, entropies.mean().item())

        coeff = torch.exp(h0 - ent.detach())
        entropy_loss = (ent * coeff).mean()

        # Prior alignment (SAR-style APU): maximize the entropy of the batch
        # mean prediction. Keeps the marginal class distribution stable.
        # NOTE (V4 finding): marginals are already near-max entropy on DR, so
        # this alone cannot stop class-2 erosion — see the head trust region.
        prior_loss_val = 0.0
        if self.prior_align_weight > 0:
            mean_p = probs.mean(dim=0)
            prior_H = -(mean_p * mean_p.clamp_min(1e-12).log()).sum()
            loss = entropy_loss - self.prior_align_weight * prior_H
            prior_loss_val = float(prior_H.item())
        else:
            loss = entropy_loss

        finite = bool(torch.isfinite(loss).item())
        if not finite or loss.grad_fn is None:
            for p, sv in zip(self._params, saved):
                p.data.copy_(sv.to(p.device))
            self._batch_num += 1
            self._current_model_probs = updated_probs
            return self._zero_metrics(B, entropies.mean().item())

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # ---- subspace tracking (official moving_W queue) ----
        queue_appended = False
        if self._batch_num > 0 and self._batch_num % self.sample_interval == 0:
            gvec = self._get_grad_vec().detach().cpu()
            self._grad_queue.append(gvec)
            if len(self._grad_queue) > self.queue_length:
                self._grad_queue.pop(0)
            queue_appended = True

        subspace_active = False
        proj_ratio = -1.0
        if len(self._grad_queue) >= self.subspace_dim:
            P = self._compute_subspace()
            if P is not None:
                gk = self._get_grad_vec()
                proj = torch.mm(P.t(), torch.mm(P, gk.unsqueeze(1)))
                proj_ratio = float(
                    (proj.norm() / (gk.norm() + 1e-12)).item())
                self._set_grad_vec(proj.squeeze(1).to(gk.device))
                subspace_active = True

        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
        self._optimizer.step()

        # Head trust region (DR): pull the classifier a fixed fraction of the
        # way back toward its setup-time prototype weights after every step.
        # Implemented post-step (NOT as a loss term) because Adam renormalizes
        # per-parameter steps — an L2 loss term is almost a no-op under Adam,
        # while this geometric pull bounds drift for any optimizer. Head
        # boundary sharpening is what erodes the ambiguous middle grade
        # (class 2); the trust region keeps gains flowing through LN feature
        # adaptation instead.
        head_pull_applied = False
        if self.head_anchor_weight > 0 and self._head_params:
            lam = min(0.9, self.head_anchor_weight)
            with torch.no_grad():
                for p, p0 in zip(self._head_params, self._head_source):
                    p.data.lerp_(p0.to(p.device), lam)
            head_pull_applied = True

        self.model.zero_grad(set_to_none=True)

        self._current_model_probs = updated_probs
        self._batch_num += 1

        with torch.no_grad():
            max_prob = float(probs.max(dim=-1).values.mean().item())

        return {
            "loss": float(loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "prior_loss": prior_loss_val,
            "head_pull": head_pull_applied,
            "student_entropy": float(entropies.mean().item()),
            "mean_max_prob": max_prob,
            "num_confident_samples": int(mask.sum().item()),
            "queue_len": len(self._grad_queue),
            "queue_appended": queue_appended,
            "subspace_active": subspace_active,
            "proj_norm_ratio": proj_ratio,
            "batch_num": self._batch_num,
            "updated": True,
        }

    # ------------------------------------------------------------------ #
    # Sample filtering
    # ------------------------------------------------------------------ #
    def _select_confident(self, entropies: torch.Tensor, probs: torch.Tensor,
                          B: int) -> torch.Tensor:
        """Confident-sample selection: entropy margin H < entropy_margin*ln(C)
        UNION a top-k quota (lcotta_min_confident_fraction) of the lowest-
        entropy samples, then optional per-class cap.

        The quota is enforced on EVERY batch (unlike the official code, which
        only margin-filters): an overconfident medical model can sit entirely
        below the margin with near-zero entropies -> vanishing gradient and
        dead adaptation. Topping up to the quota keeps the signal alive.

        The top-up is CLASS-BALANCED (round-robin over predicted classes by
        ascending entropy) instead of a global top-k: a pure global top-k can
        select every sample from the majority grade, leaving hard minority
        grades with no gradient to anchor them.
        """
        device = entropies.device
        num_classes = probs.shape[1]
        h0 = self.entropy_margin * math.log(max(1, num_classes))
        mask = entropies < h0
        quota = max(1, int(round(self.min_confident_fraction * B)))
        if int(mask.sum().item()) < quota:
            preds = probs.argmax(dim=1)
            chosen = self._balanced_quota_indices(entropies, preds, quota)
            topup = torch.zeros(B, dtype=torch.bool, device=device)
            if chosen:
                topup[torch.tensor(chosen, dtype=torch.long, device=device)] = True
            mask = mask | topup
        if self.per_class_cap is not None:
            capped = self._cap_by_class(mask, probs, entropies)
            if int(capped.sum().item()) >= 1:
                mask = capped
        return mask

    @staticmethod
    def _balanced_quota_indices(entropies: torch.Tensor, preds: torch.Tensor,
                                quota: int) -> list:
        """Round-robin over predicted classes, each class contributing its
        lowest-entropy unchosen samples, until `quota` indices are filled."""
        C = int(preds.max().item()) + 1
        per_class = {c: [] for c in range(C)}
        for i in torch.argsort(entropies).tolist():
            per_class[int(preds[i])].append(i)
        ptr = {c: 0 for c in per_class}
        chosen = []
        progressed = True
        while len(chosen) < quota and progressed:
            progressed = False
            for c in sorted(per_class):
                if len(chosen) >= quota:
                    break
                if ptr[c] < len(per_class[c]):
                    chosen.append(per_class[c][ptr[c]])
                    ptr[c] += 1
                    progressed = True
        return chosen

    def _cap_by_class(self, mask: torch.Tensor, probs: torch.Tensor,
                      entropies: torch.Tensor) -> torch.Tensor:
        preds = probs.argmax(dim=1)
        out = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        for c in range(probs.shape[1]):
            cand = mask & (preds == c)
            n = int(cand.sum().item())
            if n == 0:
                continue
            if n <= self.per_class_cap:
                out |= cand
            else:
                e = entropies.masked_fill(~cand, float("inf"))
                _, inds = torch.topk(e, self.per_class_cap, largest=False)
                out[inds] = True
        return out

    def _update_model_probs(self, new_probs: torch.Tensor) -> Optional[torch.Tensor]:
        """EMA of the mean prediction (official update_model_probs)."""
        if new_probs.size(0) == 0:
            return self._current_model_probs
        with torch.no_grad():
            mean = new_probs.mean(0)
            if self._current_model_probs is None:
                return mean.cpu()
            return (
                self.prob_ema * self._current_model_probs.to(mean.device)
                + (1.0 - self.prob_ema) * mean
            ).cpu()

    @staticmethod
    def _softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    # ------------------------------------------------------------------ #
    # Gradient vector <-> parameter mapping, subspace
    # ------------------------------------------------------------------ #
    def _get_grad_vec(self) -> torch.Tensor:
        vecs = []
        for p in self._params:
            g = p.grad
            vecs.append(g.detach().reshape(-1) if g is not None
                        else torch.zeros(p.numel(), device=p.device))
        return torch.cat(vecs, 0)

    def _set_grad_vec(self, grad_vec: torch.Tensor) -> None:
        idx = 0
        for p, shape in zip(self._params, self._param_shapes):
            size = p.numel()
            p.grad = grad_vec[idx:idx + size].reshape(shape).clone()
            idx += size

    def _compute_subspace(self) -> Optional[torch.Tensor]:
        """PCA (SVD) of the centered gradient queue -> (r, n) orthonormal rows."""
        try:
            W = torch.stack(self._grad_queue, dim=0)          # (k, n) cpu f32
            W = W - W.mean(dim=0, keepdim=True)
            # SVD of the small-side matrix; rows of Vt are the principal dirs.
            _, _, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.subspace_dim, Vt.shape[0])
            return Vt[:r].to(self._device)
        except Exception as exc:  # numerical issues must never kill adaptation
            logger.warning("LCoTTA: subspace computation failed (%s); "
                           "falling back to plain step", exc)
            return None

    # ------------------------------------------------------------------ #
    # Restore / fallback / serialization
    # ------------------------------------------------------------------ #
    def restore_parameters(self):
        for p, sv in zip(self._params, self._source_weights):
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
                "LCoTTA fallback: QWK %.4f->%.4f, acc %.4f->%.4f",
                b_qwk, c_qwk, b_acc, c_acc,
            )
            self.restore_parameters()
            self._fallback_active = True
            return True
        return False

    def _zero_metrics(self, B, ent_mean: float = 0.0) -> dict:
        return {
            "loss": 0.0, "entropy_loss": 0.0, "student_entropy": ent_mean,
            "mean_max_prob": 0.0, "num_confident_samples": 0,
            "queue_len": len(self._grad_queue), "queue_appended": False,
            "subspace_active": False, "proj_norm_ratio": -1.0,
            "batch_num": self._batch_num, "updated": False,
        }

    def state_dict(self):
        return {
            "source_weights": [v.cpu().clone() for v in self._source_weights],
            "params": [p.detach().cpu().clone() for p in self._params],
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "grad_queue": [v.clone() for v in self._grad_queue],
            "batch_num": self._batch_num,
            "current_model_probs": (None if self._current_model_probs is None
                                    else self._current_model_probs.cpu().clone()),
            "setup_done": self._setup_done,
            "fallback_active": self._fallback_active,
        }

    def load_state_dict(self, state):
        self._source_weights = [v.clone() for v in state.get("source_weights", [])]
        self._setup_done = state.get("setup_done", True)
        self._fallback_active = state.get("fallback_active", False)
        self._batch_num = state.get("batch_num", 0)
        self._grad_queue = [v.clone() for v in state.get("grad_queue", [])]
        probs = state.get("current_model_probs")
        self._current_model_probs = None if probs is None else probs.clone()
        params = state.get("params", [])
        for p, v in zip(self._params, params):
            p.data.copy_(v.to(p.device))
        opt_state = state.get("optimizer")
        if opt_state is not None and self._optimizer is not None:
            self._optimizer.load_state_dict(opt_state)
