"""PALM adapter — adaptive LR via prediction uncertainty + parameter sensitivity.

Reference: "PALM: Pushing Adaptive Learning Rate Mechanisms for
            Continual Test-Time Adaptation" (AAAI 2025)

Two stages per batch:
  1. Layer selection: compute KL(softmax(ℓ/T) || uniform) gradients,
     select per-tensor parameters with small L2 gradient norms.
  2. Adaptive LR: compute parameter sensitivity S = |θ·∇θ|,
     maintain EMA Ŝ, compute importance i = |S - Ŝ| / Ŝ,
     set per-tensor LR = base_lr * i for selected tensors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

# ImageNet-normalized values live in roughly [-2.5, 2.5];
# use a wider clamp than [0,1] for augmentations.
_AUG_CLAMP = (-3.0, 3.0)


@register_adapter("palm")
class PALMAdapter(CTTAAdapter):
    """PALM adapter — selective layer adaptation with per-tensor adaptive LR."""

    def __init__(self, lr=1e-4, weight_decay=0.0, image_size=224,
                 temperature=10.0, layer_selection_threshold=None,
                 selection_percentile=0.3, sensitivity_alpha=0.5,
                 consistency_lambda=0.01, entropy_threshold_factor=0.8,
                 num_augmentations=1):
        self.lr = lr
        self.base_lr = lr
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.temperature = temperature
        self.layer_selection_threshold = layer_selection_threshold
        self.selection_percentile = selection_percentile
        self.sensitivity_alpha = sensitivity_alpha
        self.consistency_lambda = consistency_lambda
        self.entropy_threshold_factor = entropy_threshold_factor
        self.num_augmentations = num_augmentations

        self.model = None
        self.param_names: List[str] = []
        self.param_refs: List[nn.Parameter] = []
        self.param_shapes: Dict[str, torch.Size] = {}
        self.optimizer = None
        self.running_sensitivity: Dict[str, torch.Tensor] = {}
        self.num_classes: int = 0
        self.entropy_threshold: float = 0.0
        self._setup_done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, model: FoundationModel,
              source_snapshot: Optional[Dict[str, torch.Tensor]] = None) -> None:
        if self._setup_done:
            return
        self.model = model
        self.num_classes = model.get_num_classes()
        self.entropy_threshold = self.entropy_threshold_factor * _log(self.num_classes)

        self.param_names, self.param_refs = self._collect_trainable_params(model)
        self.param_shapes = {n: p.shape for n, p in zip(self.param_names, self.param_refs)}

        param_groups = [{'params': [p]} for p in self.param_refs]
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=self.base_lr, weight_decay=self.weight_decay,
        )

        self.running_sensitivity = {
            name: torch.zeros_like(p.detach().cpu())
            for name, p in zip(self.param_names, self.param_refs)
        }
        self._setup_done = True

    def adapt_step(self, batch: torch.Tensor, logits: torch.Tensor = None) -> Dict[str, float]:
        B = batch.shape[0]
        C = self.num_classes
        device = batch.device
        eps = 1e-8

        # ---- Stage 1: KL divergence backward for gradient information ----
        self.optimizer.zero_grad(set_to_none=True)
        logits_kl = self.model.forward(batch)
        L_kl = self._kl_div_uniform_loss(logits_kl)
        L_kl.backward()

        param_norms = self._compute_param_grad_norms()
        selected_param_names = self._select_layers(param_norms)

        # ---- Stage 2: Compute sensitivity & importance (correct order) ----
        # Paper Eq. 4: S = |θ · ∇θ|
        # Eq. 5: Ŝ_new = α·S + (1-α)·Ŝ_prev   [UPDATE FIRST]
        # Eq. 6: D̂ = |S - Ŝ_new|
        # Eq. 7: i = D̂ / Ŝ_new
        param_importance = {}
        for name in selected_param_names:
            idx = self.param_names.index(name)
            param = self.param_refs[idx]
            if param.grad is None:
                continue

            S = torch.abs(param.detach() * param.grad.detach()).cpu()

            S_hat_prev = self.running_sensitivity[name].to(S.device)

            S_hat_new = (self.sensitivity_alpha * S
                         + (1.0 - self.sensitivity_alpha) * S_hat_prev)
            self.running_sensitivity[name] = S_hat_new.cpu()

            D_hat = torch.abs(S - S_hat_new)
            importance = (D_hat + eps) / (S_hat_new + eps)
            param_importance[name] = importance

        # ---- Stage 3: Per-tensor adaptive LR via optimizer param groups ----
        for idx, name in enumerate(self.param_names):
            if name in param_importance:
                imp_scalar = float(param_importance[name].mean().item())
                self.optimizer.param_groups[idx]['lr'] = self.base_lr * imp_scalar
            else:
                self.optimizer.param_groups[idx]['lr'] = 0.0

        # ---- Stage 4: Forward for optimization loss (entropy + consistency) ----
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model.forward(batch)

        H = _entropy(logits)
        confident = H <= self.entropy_threshold
        n_confident = int(confident.sum().item())
        mean_entropy = float(H.mean().item())

        L_total = torch.tensor(0.0, device=device)
        if n_confident > 0:
            L_total = H[confident].mean()

        if self.consistency_lambda > 0 and self.num_augmentations > 0:
            L_const = torch.tensor(0.0, device=device)
            with torch.no_grad():
                target_probs = F.softmax(logits, dim=-1)
            for _ in range(self.num_augmentations):
                aug = self._apply_augmentation(batch)
                logits_aug = self.model.forward(aug)
                L_const += F.kl_div(
                    F.log_softmax(logits_aug, dim=-1),
                    target_probs,
                    reduction="batchmean",
                )
            L_const = L_const / self.num_augmentations
            L_total = L_total + self.consistency_lambda * L_const

        if L_total.requires_grad and L_total.grad_fn is not None:
            L_total.backward()

            n_selected = len(param_importance)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            n_selected = 0
            self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            final_logits = self.model.forward(batch)
            final_probs = F.softmax(final_logits, dim=-1)
            final_ent = -(final_probs * (final_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()

        max_prob = float(final_probs.max(dim=-1).values.mean().item())

        return {
            "loss": float(L_total.item()) if isinstance(L_total, torch.Tensor) else 0.0,
            "kl_loss": float(L_kl.item()),
            "entropy": mean_entropy,
            "post_adapt_entropy": final_ent,
            "num_confident_samples": n_confident,
            "mean_max_prob": max_prob,
            "n_selected_params": n_selected,
            "n_total_params": len(self.param_names),
            "selected_fraction": n_selected / max(len(self.param_names), 1),
        }

    def restore_parameters(self) -> None:
        """PALM does not use stochastic restore; this is a no-op."""
        pass

    def state_dict(self) -> Dict:
        return {
            "running_sensitivity": self.running_sensitivity,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "setup_done": self._setup_done,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.running_sensitivity = state.get("running_sensitivity", {})
        self._setup_done = state.get("setup_done", True)
        if state.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(state["optimizer"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_trainable_params(self, model: FoundationModel) -> Tuple[List[str], List[nn.Parameter]]:
        names, params = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)
        return names, params

    def _kl_div_uniform_loss(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits / self.temperature, dim=-1)
        log_uniform = -math.log(self.num_classes)
        kl_per_sample = (p * (p.clamp_min(1e-10).log() - log_uniform)).sum(dim=-1)
        return kl_per_sample.mean()

    def _compute_param_grad_norms(self) -> Dict[str, float]:
        norms = {}
        for name, param in zip(self.param_names, self.param_refs):
            if param.grad is None:
                continue
            norms[name] = float(param.grad.detach().norm(p=2).item())
        return norms

    def _select_layers(self, param_norms: Dict[str, float]) -> set:
        if not param_norms:
            return set()
        if self.layer_selection_threshold is not None:
            return {
                name for name, score in param_norms.items()
                if score <= self.layer_selection_threshold
            }
        scores = list(param_norms.values())
        k = max(1, int(len(scores) * self.selection_percentile))
        sorted_params = sorted(param_norms, key=param_norms.get)
        return set(sorted_params[:k])

    @staticmethod
    def _apply_augmentation(batch: torch.Tensor) -> torch.Tensor:
        """Apply one of 8 retinal-specific augmentations on GPU.

        Input is ImageNet-normalized (mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]), so values span roughly [-2.5, 2.5].
        Do NOT clamp to [0,1]; use _AUG_CLAMP instead.
        """
        import torchvision.transforms.functional as TF
        aug_type = torch.randint(0, 8, (1,)).item()
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
            return torch.clamp(
                batch * brightness.view(1, 1, 1, 1), *_AUG_CLAMP)
        elif aug_type == 5:
            contrast = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            mean = batch.mean(dim=[2, 3], keepdim=True)
            return torch.clamp(
                (batch - mean) * contrast.view(1, 1, 1, 1) + mean,
                *_AUG_CLAMP)
        elif aug_type == 6:
            return torch.clamp(batch + 0.05 * torch.randn_like(batch),
                               *_AUG_CLAMP)
        else:
            perm = torch.randperm(3, device=batch.device)
            return batch[:, perm, :, :]


def _log(x: float) -> float:
    return math.log(x)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * (probs.clamp_min(1e-10)).log()).sum(dim=-1)
