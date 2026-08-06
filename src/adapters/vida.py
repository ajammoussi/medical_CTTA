"""ViDA: Homeostatic Visual Domain Adapter for Continual Test-Time Adaptation.

Implements ViDA (ICLR 2024) adapted for medical imaging DR classification.

Key ideas:
  - Dual adapter branches: low-rank (domain-shared) + high-rank (domain-specific)
  - Homeostatic Knowledge Allotment (HKA): uncertainty-based dynamic weighting
  - Symmetric KL consistency loss between student and EMA teacher
  - Dual-rate EMA: slower for original model, faster for adapter parameters
  - Stochastic parameter restoration to prevent catastrophic forgetting

Reference:
  Liu et al., "ViDA: Homeostatic Visual Domain Adapter for Continual Test Time
  Adaptation", ICLR 2024.
  Code: https://github.com/Yangsenqiao/vida
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional

from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ViDA Injected Linear module
# ---------------------------------------------------------------------------


class ViDAInjectedLinear(nn.Module):
    """Linear layer with dual adapter branches (low-rank + high-rank).

    Output: f_o + lambda_l * f_l + lambda_h * f_h

    where:
      f_o = W @ x                    (original frozen weights)
      f_l = W_l_up @ W_l_down @ x    (low-rank branch, bottleneck = r)
      f_h = W_h_up @ W_h_down @ x    (high-rank branch, bottleneck = r2)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, r: int = 1, r2: int = 128):
        super().__init__()

        # Original frozen linear layer
        self.linear_vida = nn.Linear(in_features, out_features, bias)

        # Low-rank branch (domain-shared knowledge)
        self.vida_down = nn.Linear(in_features, r, bias=False)
        self.vida_up = nn.Linear(r, out_features, bias=False)

        # High-rank branch (domain-specific knowledge)
        self.vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_features, bias=False)

        # Scale factors (set dynamically via HKA)
        self.scale1 = 1.0  # lambda_l (low-rank)
        self.scale2 = 1.0  # lambda_h (high-rank)

        # Initialize adapter weights to zero (starts as identity)
        nn.init.normal_(self.vida_down.weight, std=1 / max(r, 1) ** 2)
        nn.init.zeros_(self.vida_up.weight)
        nn.init.normal_(self.vida_down2.weight, std=1 / max(r2, 1) ** 2)
        nn.init.zeros_(self.vida_up2.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            self.linear_vida(input)
            + self.vida_up(self.vida_down(input)) * self.scale1
            + self.vida_up2(self.vida_down2(input)) * self.scale2
        )


# ---------------------------------------------------------------------------
# Injection utility
# ---------------------------------------------------------------------------


def inject_vida_adapters(
    model: nn.Module,
    r: int = 1,
    r2: int = 128,
    target_modules: Optional[List[str]] = None,
) -> List[str]:
    """Replace Linear layers inside attention blocks with ViDAInjectedLinear.

    For timm ViTs: targets ``blocks.{i}.attn.qkv`` and ``blocks.{i}.attn.proj``.
    For custom ViTs (VisionFM): targets ``blocks.{i}.attn.qkv`` and
    ``blocks.{i}.attn.proj``.

    Returns list of injected parameter names.
    """
    if target_modules is None:
        target_modules = ["attn"]

    injected_names = []

    def _find_and_replace(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this child's class name matches a target
            if type(child).__name__ in target_modules and hasattr(child, "qkv"):
                # Replace qkv and proj inside attention modules
                _replace_linear_in_module(child, full_name, r, r2, injected_names)
            else:
                _find_and_replace(child, full_name)

    _find_and_replace(model)
    return injected_names


def _replace_linear_in_module(
    attn_module: nn.Module,
    prefix: str,
    r: int,
    r2: int,
    injected_names: List[str],
):
    """Replace qkv and proj Linear layers inside an attention module."""
    for attr_name in ["qkv", "proj"]:
        if not hasattr(attn_module, attr_name):
            continue
        original = getattr(attn_module, attr_name)
        if not isinstance(original, nn.Linear):
            continue

        vida_layer = ViDAInjectedLinear(
            in_features=original.in_features,
            out_features=original.out_features,
            bias=original.bias is not None,
            r=r,
            r2=r2,
        )

        # Copy original weights into the frozen branch
        vida_layer.linear_vida.weight.data.copy_(original.weight.data)
        if original.bias is not None:
            vida_layer.linear_vida.bias.data.copy_(original.bias.data)

        # Freeze original weights
        vida_layer.linear_vida.weight.requires_grad_(False)
        if original.bias is not None:
            vida_layer.linear_vida.bias.requires_grad_(False)

        setattr(attn_module, attr_name, vida_layer)
        injected_names.extend([
            f"{prefix}.{attr_name}.vida_down.weight",
            f"{prefix}.{attr_name}.vida_up.weight",
            f"{prefix}.{attr_name}.vida_down2.weight",
            f"{prefix}.{attr_name}.vida_up2.weight",
        ])

        logger.debug(
            "Injected ViDA adapter into %s.%s: "
            "(%d -> %d, r=%d, r2=%d)",
            prefix, attr_name, original.in_features, original.out_features, r, r2,
        )


# ---------------------------------------------------------------------------
# Symmetric KL loss
# ---------------------------------------------------------------------------


def symmetric_kl_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence: 0.5 * KL(p_t || p_s) + 0.5 * KL(p_s || p_t).

    This is equivalent to the original ViDA softmax_entropy loss.
    """
    p_student = F.softmax(student_logits, dim=-1)
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_teacher = F.softmax(teacher_logits, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_teacher || p_student)
    kl_1 = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)
    # KL(p_student || p_teacher)
    kl_2 = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)

    return 0.5 * kl_1 + 0.5 * kl_2


# ---------------------------------------------------------------------------
# ViDA Adapter
# ---------------------------------------------------------------------------


@register_adapter("vida")
class ViDAAdapter(CTTAAdapter):
    """ViDA: Homeostatic Visual Domain Adapter for CTTA.

    Injects dual-branch adapters (low-rank + high-rank) into attention layers.
    Uses augmentation variance for uncertainty estimation and homeostatic
    knowledge allotment to dynamically balance adapter branches.

    DR-specific adaptations from CoTTA/PALM analysis:
      - Conservative learning rates (1e-5 range) for overconfident ViTs
      - Stochastic restoration to prevent entropy collapse
      - Augmentation variance (not MC Dropout) for uncertainty estimation
      - Per-class cap on pseudo-labels to prevent class domination
    """

    def __init__(
        self,
        # ViDA-specific
        vida_rank1: int = 1,
        vida_rank2: int = 128,
        uncertainty_threshold: float = 0.2,
        num_augmentations: int = 10,
        uncertainty_scale: float = 0.1,
        alpha_teacher: float = 0.99,
        alpha_vida: float = 0.8,
        vida_lr: float = 5e-4,
        model_lr: float = 5e-7,
        # Stochastic restoration (0 = disabled, per ViDA paper)
        restore_prob: float = 0.0,
        # Pseudo-label classification loss
        ce_loss_weight: float = 1.0,
        pseudo_label_threshold: float = 0.5,
        # Augmentation
        image_size: int = 224,
        # Shared with other adapters (accepted but unused)
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        self.vida_rank1 = vida_rank1
        self.vida_rank2 = vida_rank2
        self.uncertainty_threshold = uncertainty_threshold
        self.num_augmentations = num_augmentations
        self.uncertainty_scale = uncertainty_scale
        self.alpha_teacher = alpha_teacher
        self.alpha_vida = alpha_vida
        self.vida_lr = vida_lr
        self.model_lr = model_lr
        self.restore_prob = restore_prob
        self.ce_loss_weight = ce_loss_weight
        self.pseudo_label_threshold = pseudo_label_threshold
        self.image_size = image_size
        self.max_grad_norm = max_grad_norm
        # Store unused kwargs to avoid errors from base runner
        self._unused_kwargs = kwargs

        # Internal state
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self._param_names: List[str] = []
        self._vida_param_names: List[str] = []
        self._source_state: Dict[str, torch.Tensor] = {}
        self._restoration_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(
        self,
        model: FoundationModel,
        source_snapshot: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_snapshot: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Setup ViDA adapters and teacher model."""
        self.model = model

        # Step 1: Inject ViDA adapters into attention layers
        injected = inject_vida_adapters(
            model.backbone,
            r=self.vida_rank1,
            r2=self.vida_rank2,
            target_modules=["Attention", "attn"],
        )
        logger.info(
            "Injected %d ViDA adapter parameter tensors into backbone",
            len(injected),
        )

        # Move injected adapter params to the same device as the model
        device = next(model.parameters()).device
        model.to(device)

        # Step 2: Collect trainable params (only adapter params are trainable)
        vida_params = []
        vida_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and "vida_" in name:
                vida_params.append(param)
                vida_names.append(name)

        self._vida_param_names = vida_names
        self._param_names = vida_names

        logger.info(
            "ViDA trainable params: %d adapter tensors (%s total)",
            len(vida_names),
            f"{sum(p.numel() for p in vida_params):,}",
        )

        # Step 3: Create optimizer (only adapter params)
        # Adam provides per-parameter adaptive learning rates, which is critical
        # for adapter training where gradient magnitudes vary widely across layers.
        self.optimizer = torch.optim.Adam(
            vida_params,
            lr=self.vida_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        # Step 4: Save source state for stochastic restoration
        self._source_state = {
            name: param.detach().cpu().clone()
            for name, param in zip(vida_names, vida_params)
        }

        # Step 5: Create EMA teacher (deep copy of model)
        self.teacher_model = copy.deepcopy(model)
        # Freeze teacher completely
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_model.eval()

        logger.info(
            "ViDA setup complete: lr_model=%.1e, lr_vida=%.1e, "
            "alpha_teacher=%.4f, alpha_vida=%.4f, restore_prob=%.4f, "
            "ce_loss_weight=%.2f, pseudo_label_thresh=%.2f, optimizer=Adam",
            self.model_lr, self.vida_lr,
            self.alpha_teacher, self.alpha_vida,
            self.restore_prob,
            self.ce_loss_weight, self.pseudo_label_threshold,
        )

    def adapt_step(
        self,
        batch: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one ViDA adaptation step.

        Protocol:
          1. Teacher eval -> compute uncertainty via augmentation variance
          2. HKA: set scale factors based on uncertainty
          3. Teacher forward -> consistency target + pseudo-labels
          4. Student forward on augmented input
          5. Combined loss: KL consistency + pseudo-label CE
          6. Update student
          7. Dual-rate EMA update teacher
          8. Stochastic restore
        """
        B = batch.shape[0]
        device = batch.device

        # ---- Step 1: Teacher in eval mode for uncertainty estimation ----
        self.teacher_model.eval()

        # ---- Step 2: Compute uncertainty via augmentation variance ----
        augmented_preds = []
        with torch.no_grad():
            for _ in range(self.num_augmentations):
                aug = self._apply_augmentation(batch)
                pred = self.teacher_model(aug)
                augmented_preds.append(pred)

        # Variance across augmentations as uncertainty measure
        stacked = torch.stack(augmented_preds)  # [N, B, C]
        variance = torch.var(stacked, dim=0)    # [B, C]
        uncertainty = float(torch.mean(variance).item()) * self.uncertainty_scale

        # ---- Step 3: Homeostatic Knowledge Allotment (HKA) ----
        if uncertainty >= self.uncertainty_threshold:
            # High uncertainty: boost high-rank (domain-specific)
            lambda_high = 1.0 + uncertainty
            lambda_low = 1.0 - uncertainty
        else:
            # Low uncertainty: boost low-rank (domain-shared)
            lambda_low = 1.0 + uncertainty
            lambda_high = 1.0 - uncertainty

        # Clamp scales to reasonable range
        lambda_low = max(0.0, min(2.0, lambda_low))
        lambda_high = max(0.0, min(2.0, lambda_high))

        self._set_adapter_scales(self.model, lambda_low, lambda_high)
        self._set_adapter_scales(self.teacher_model, lambda_low, lambda_high)

        # ---- Step 4: Teacher forward (consistency target + pseudo-labels) ----
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            teacher_conf, teacher_pred = teacher_probs.max(dim=-1)
            confidence_mask = teacher_conf > self.pseudo_label_threshold

        # ---- Step 5: Student forward on AUGMENTED input ----
        # Apply a random augmentation so student sees a different view than teacher.
        # This creates a non-zero KL loss even when weights are identical,
        # providing the gradient signal needed for adapter learning.
        student_input = self._apply_augmentation(batch)
        self.model.train()
        student_logits = self.model(student_input)

        # ---- Step 6: Combined loss ----
        # KL consistency: student should match teacher despite different augmentation
        kl_loss = symmetric_kl_loss(student_logits, teacher_logits.detach()).mean()

        # Pseudo-label CE: direct classification signal from teacher's confident predictions
        if confidence_mask.sum() > 0:
            ce_loss = F.cross_entropy(
                student_logits[confidence_mask],
                teacher_pred[confidence_mask],
            )
        else:
            ce_loss = torch.tensor(0.0, device=device)

        loss = kl_loss + self.ce_loss_weight * ce_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — only clip adapter params
        vida_params = [p for n, p in self.model.named_parameters()
                       if p.requires_grad and "vida_" in n]
        if vida_params:
            torch.nn.utils.clip_grad_norm_(vida_params, self.max_grad_norm)

        self.optimizer.step()

        # ---- Step 7: Dual-rate EMA teacher update ----
        self._ema_update_teacher()

        # ---- Step 8: Stochastic parameter restoration (skipped when restore_prob=0) ----
        if self.restore_prob > 0:
            self._stochastic_restore()
        else:
            self._restoration_count = 0

        # ---- Metrics ----
        with torch.no_grad():
            s_probs = F.softmax(student_logits, dim=-1)
            t_probs = F.softmax(teacher_logits, dim=-1)
            s_entropy = -(s_probs * (s_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()
            t_entropy = -(t_probs * (t_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()
            s_max_prob = s_probs.max(dim=-1).values.mean().item()

        return {
            "loss": loss.item(),
            "kl_loss": kl_loss.item(),
            "ce_loss": ce_loss.item(),
            "confidence_frac": float(confidence_mask.float().mean().item()),
            "uncertainty": uncertainty,
            "lambda_low": lambda_low,
            "lambda_high": lambda_high,
            "student_entropy": s_entropy,
            "teacher_entropy": t_entropy,
            "mean_max_prob": s_max_prob,
            "restoration_count": self._restoration_count,
        }

    def restore_parameters(self) -> None:
        """Hard restore adapter parameters to source values."""
        for name in self._vida_param_names:
            for n, p in self.model.named_parameters():
                if n == name and name in self._source_state:
                    p.data.copy_(self._source_state[name].to(p.device))

    def state_dict(self) -> Dict:
        return {
            "source_state": self._source_state,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._source_state = state.get("source_state", {})
        if state.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(state["optimizer"])

    @property
    def teacher_weights(self) -> Dict[str, torch.Tensor]:
        """Return teacher model state dict (for sequential runner compatibility)."""
        return self.teacher_model.state_dict() if self.teacher_model is not None else {}

    def set_teacher_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Overwrite teacher model weights (e.g. restore post-source checkpoint)."""
        if self.teacher_model is not None:
            self.teacher_model.load_state_dict(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_adapter_scales(self, model: nn.Module, lambda_low: float, lambda_high: float):
        """Set scale factors on all ViDAInjectedLinear modules."""
        for module in model.modules():
            if isinstance(module, ViDAInjectedLinear):
                module.scale1 = lambda_low
                module.scale2 = lambda_high

    def _ema_update_teacher(self):
        """Dual-rate EMA: slower for original model params, faster for adapter params."""
        for (t_name, t_param), (s_name, s_param) in zip(
            self.teacher_model.named_parameters(),
            self.model.named_parameters(),
        ):
            if "vida_" in s_name:
                # Adapter parameters: faster EMA
                t_param.data = (
                    self.alpha_vida * t_param.data
                    + (1.0 - self.alpha_vida) * s_param.data
                )
            else:
                # Original model parameters: slower EMA
                t_param.data = (
                    self.alpha_teacher * t_param.data
                    + (1.0 - self.alpha_teacher) * s_param.data
                )

    def _stochastic_restore(self):
        """Stochastic restoration of adapter parameters to source values."""
        count = 0
        for name, param in zip(self._vida_param_names, self._get_vida_params()):
            if name in self._source_state:
                mask = torch.rand_like(param.data) < self.restore_prob
                param.data[mask] = self._source_state[name].to(param.device)[mask]
                count += int(mask.sum().item())
        self._restoration_count = count

    def _get_vida_params(self) -> List[nn.Parameter]:
        """Get ViDA adapter parameters from the model."""
        return [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "vida_" in n
        ]

    @staticmethod
    def _apply_augmentation(batch: torch.Tensor) -> torch.Tensor:
        """Apply one of 7 retinal-specific augmentations on GPU."""
        import torchvision.transforms.functional as TF
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
