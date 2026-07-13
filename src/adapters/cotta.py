"""CoTTA adapter with single-model-copy design for T4 VRAM efficiency.

Design:
  - Single model copy (no deepcopy); teacher maintained as CPU dict of LN params.
  - LayerNorm-only adaptation (~1% of 307M ViT-L params).
  - Per-neuron stochastic restoration (paper-correct).
  - KATANA-style augmentations applied on GPU.
  - No source-confidence fallback.
  - Uses the `logits` argument instead of recomputing teacher forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel


@register_adapter("cotta")
class CoTTAAdapter(CTTAAdapter):
    """CoTTA adapter — single-model-copy, LayerNorm-only, per-neuron restore."""

    def __init__(self, ema_alpha=0.999, restore_prob=0.01,
                 num_augmentations=8, confidence_threshold=0.5,
                 lr=1e-4, weight_decay=0.0, image_size=224):
        self.ema_alpha = ema_alpha
        self.restore_prob = restore_prob
        self.num_augmentations = num_augmentations
        self.confidence_threshold = confidence_threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.model = None
        self._param_names: List[str] = []
        self._param_refs: List[nn.Parameter] = []
        self.source_weights: Dict[str, torch.Tensor] = {}
        self.teacher_weights: Dict[str, torch.Tensor] = {}
        self.optimizer = None
        self._restoration_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, model: FoundationModel,
              source_snapshot: Optional[Dict[str, torch.Tensor]] = None) -> None:
        self.model = model
        params, self._param_names = self._collect_ln_params(model)
        self._param_refs = params

        self.optimizer = torch.optim.SGD(
            params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay,
        )

        self.source_weights = {}
        for name, p in zip(self._param_names, params):
            self.source_weights[name] = p.detach().cpu().clone()

        self.teacher_weights = {k: v.clone() for k, v in self.source_weights.items()}

    def adapt_step(self, batch: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
        """One adaptation step.

        Args:
            batch: (B, C, H, W) tensor on GPU.
            logits: model(batch) output *before* this step (used for metrics).

        Returns:
            Dict of scalar metrics.
        """
        B = batch.shape[0]
        num_classes = self.model.get_num_classes()

        # ---- Save current LN weights ----
        saved = {}
        for name, ref in zip(self._param_names, self._param_refs):
            saved[name] = ref.data.clone()

        # ---- Load teacher weights for pseudo-label computation ----
        self._load_weights(self.teacher_weights)

        # ---- Augmentation-averaged pseudo-labels ----
        avg_probs = torch.zeros(B, num_classes, device=batch.device)
        with torch.no_grad():
            for _ in range(self.num_augmentations):
                aug = self._apply_augmentation(batch)
                out = self.model.forward(aug)
                avg_probs += F.softmax(out, dim=-1)
        avg_probs /= self.num_augmentations
        max_probs, pseudo_labels = avg_probs.max(dim=-1)

        # ---- Restore student LN weights ----
        for name, ref in zip(self._param_names, self._param_refs):
            ref.data.copy_(saved[name])

        # ---- Confidence filtering: only use high-confidence predictions ----
        confident_mask = max_probs >= self.confidence_threshold
        confident_samples = int(confident_mask.sum().item())
        mean_max_prob = float(max_probs.mean().item())
        if confident_samples == 0:
            return {
                "loss": 0.0,
                "teacher_entropy": 0.0,
                "student_entropy": 0.0,
                "teacher_student_divergence": 0.0,
                "restoration_count": self._restoration_count,
                "confident_samples": 0,
                "mean_max_prob": mean_max_prob,
            }

        # ---- Student forward + backward + step (only on confident samples) ----
        student_logits = self.model.forward(batch)
        loss = F.cross_entropy(
            student_logits[confident_mask],
            pseudo_labels[confident_mask],
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # ---- Teacher EMA (CPU) ----
        self._ema_update()

        # ---- Stochastic restore to source ----
        self._restore_parameters()

        # ---- Metrics ----
        with torch.no_grad():
            t_probs = F.softmax(logits, dim=-1)
            s_probs = F.softmax(student_logits, dim=-1)
            t_ent = -(t_probs * (t_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()
            s_ent = -(s_probs * (s_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()
            div = F.kl_div(s_probs.log(), t_probs, reduction="batchmean").item()

        return {
            "loss": loss.item(),
            "teacher_entropy": t_ent,
            "student_entropy": s_ent,
            "teacher_student_divergence": div,
            "restoration_count": self._restoration_count,
            "num_confident_samples": confident_samples,
            "mean_max_prob": mean_max_prob,
        }

    def restore_parameters(self) -> None:
        """Hard restore to source weights."""
        for name, ref in zip(self._param_names, self._param_refs):
            if name in self.source_weights:
                ref.data.copy_(self.source_weights[name].to(ref.device))

    def state_dict(self) -> Dict:
        return {
            "source_weights": self.source_weights,
            "teacher_weights": self.teacher_weights,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.source_weights = state.get("source_weights", {})
        self.teacher_weights = state.get("teacher_weights", {})
        if state.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(state["optimizer"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_ln_params(self, model: FoundationModel):
        """Return (list_of_params, list_of_names) for all LayerNorm layers."""
        names, params = [], []
        for m_name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                for pname, p in module.named_parameters(recurse=False):
                    names.append(f"{m_name}.{pname}")
                    params.append(p)
        return params, names

    def _load_weights(self, weights_dict: Dict[str, torch.Tensor]):
        for name, ref in zip(self._param_names, self._param_refs):
            if name in weights_dict:
                ref.data.copy_(weights_dict[name].to(ref.device, non_blocking=True))

    def _ema_update(self):
        for name, ref in zip(self._param_names, self._param_refs):
            tw = self.teacher_weights[name]
            tw.mul_(self.ema_alpha).add_(ref.detach().cpu(), alpha=1.0 - self.ema_alpha)

    def _restore_parameters(self):
        count = 0
        for name, ref in zip(self._param_names, self._param_refs):
            if name not in self.source_weights:
                continue
            src = self.source_weights[name]
            if ref.dim() >= 2:
                n_rows = ref.shape[0]
                row_mask = torch.rand(n_rows, device=ref.device) < self.restore_prob
                ref.data[row_mask] = src.to(ref.device)[row_mask]
                count += int(row_mask.sum().item())
            else:
                mask = torch.rand_like(ref.data) < self.restore_prob
                ref.data[mask] = src.to(ref.device)[mask]
                count += int(mask.sum().item())
        self._restoration_count = count
        return count

    @staticmethod
    def _apply_augmentation(batch: torch.Tensor) -> torch.Tensor:
        """Apply one of 8 retinal-specific augmentations on GPU."""
        import torchvision.transforms.functional as TF
        aug_type = torch.randint(0, 8, (1,)).item()
        if aug_type == 0:
            # Horizontal flip (retinal symmetry)
            return torch.flip(batch, dims=[3])
        elif aug_type == 1:
            # Vertical flip (inverted capture)
            return torch.flip(batch, dims=[2])
        elif aug_type == 2:
            # Random rotation ±15° (camera angle variation)
            angle = float(torch.randint(-15, 15, (1,)).item())
            return TF.rotate(batch, angle)
        elif aug_type == 3:
            # Gaussian blur (out-of-focus / motion blur)
            sigma = 0.5 + 1.5 * torch.rand(1, device=batch.device).item()
            k = 2 * int(3 * sigma) + 1
            blur = TF.gaussian_blur(batch, kernel_size=k, sigma=sigma)
            return blur
        elif aug_type == 4:
            # Brightness adjustment (flash / exposure differences)
            brightness = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            return torch.clamp(batch * brightness.view(1, 1, 1, 1), 0, 1)
        elif aug_type == 5:
            # Contrast adjustment (different camera sensors)
            contrast = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            mean = batch.mean(dim=[2, 3], keepdim=True)
            return torch.clamp((batch - mean) * contrast.view(1, 1, 1, 1) + mean, 0, 1)
        elif aug_type == 6:
            # Gaussian noise (sensor noise variation)
            return torch.clamp(batch + 0.05 * torch.randn_like(batch), 0, 1)
        else:
            # Channel shuffle (color calibration differences)
            perm = torch.randperm(3, device=batch.device)
            return torch.clamp(batch[:, perm, :, :], 0, 1)
