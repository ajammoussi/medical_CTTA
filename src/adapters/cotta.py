import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List
from src.adapters.base import CTTAAdapter
from src.adapters.registry import register_adapter
from src.models.base import FoundationModel

logger = logging.getLogger(__name__)


@register_adapter("cotta")
class CoTTAAdapter(CTTAAdapter):
    """CoTTA adapter — restricted-param training, single-gate KL + conditional diversity.

    Round 6c changes (DINOv2 diverse pseudo-labels):
      - confidence_threshold=0.50 (was 0.70 — starved all non-class-0 samples)
      - per_class_cap=2 (was 4 — forces class diversity in KL signal)
      - diversity_weight=0.01 (was 0.05 — stops fighting KL when model is confident)
      - kl_label_smoothing=0.0 (was 0.1 — unnecessary for well-calibrated DINOv2)

    Round 6b changes (DINOv2 balanced calibration):
      - lr=5e-5, head_lr_multiplier=5.0, blocks=4, restore_prob=0.002

    Round 5 changes:
      - head_lr_multiplier default 100->10 (was too aggressive with noisy pseudo-labels)
      - per_class_cap: caps confident samples per class per batch (prevents class-3 hijack)
      - kl_label_smoothing: smooths KL target to reduce wrong pseudo-label gradient
      - class_weights: per-class KL reliability weighting (baseline accuracy)

    Teacher ensemble:
      - teacher_ensemble_weight: blend pretrained weights into teacher for pseudo-label generation
      - Prevents teacher from drifting too far from pretrained features in cross-domain setting
      - Useful when source adaptation biases teacher against target domain classes
    """

    def __init__(self, ema_alpha=0.999, restore_prob=0.01,
                 num_augmentations=6, confidence_threshold=0.65,
                 lr=5e-4, weight_decay=0.0, image_size=224,
                 max_grad_norm=1.0, teacher_temperature=0.5,
                 entropy_weight=0.1, diversity_weight=0.05,
                 adapt_layernorm=True, adapt_last_n_blocks=6,
                 head_lr_multiplier=5.0,
                 per_class_cap=0,
                 kl_label_smoothing=0.0,
                 class_weights=None,
                 confidence_gated_restore=False,
                 teacher_ensemble_weight=0.0,
                 use_class_specific_thresholds=False,
                  class_prior_alignment_weight=0.0,
                  class_prior_alignment_temperature=0.1,
                  class_forcing_threshold=0):
        self.ema_alpha = ema_alpha
        self.restore_prob = restore_prob
        self.num_augmentations = num_augmentations
        self.confidence_threshold = confidence_threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.max_grad_norm = max_grad_norm
        self.teacher_temperature = teacher_temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.adapt_layernorm = adapt_layernorm
        self.adapt_last_n_blocks = adapt_last_n_blocks
        self.head_lr_multiplier = head_lr_multiplier
        self.per_class_cap = per_class_cap            # int: 0=off, >0 = max per class per batch
        self.kl_label_smoothing = kl_label_smoothing  # float: 0.0=off, e.g. 0.1
        self.class_weights = class_weights            # Optional[List[float]]
        self.confidence_gated_restore = confidence_gated_restore  # bool: skip restore when teacher unconfident
        self.teacher_ensemble_weight = teacher_ensemble_weight  # float: 0=off, blend pretrained into teacher for pseudo-labels
        self.use_class_specific_thresholds = use_class_specific_thresholds  # bool: dynamic per-class thresholds from PLF paper
        self.class_prior_alignment_weight = class_prior_alignment_weight  # float: 0=off, CPA loss weight from PLF paper
        self.class_prior_alignment_temperature = class_prior_alignment_temperature  # float: temperature for CPA soft label smoothing
        self.class_forcing_threshold = class_forcing_threshold  # int: 0=off; N=force class after N batches with 0 pseudo-labels
        self.model = None
        self._param_names: List[str] = []
        self._param_refs: List[nn.Parameter] = []
        self.source_weights: Dict[str, torch.Tensor] = {}
        self.teacher_weights: Dict[str, torch.Tensor] = {}
        self.pretrained_weights: Optional[Dict[str, torch.Tensor]] = None
        self.optimizer = None
        self._restoration_count = 0
        self._class_starvation_counts: Dict[int, int] = {}  #consecutive batches with 0 pseudo-labels per class
        self._class_forcing_active = False  # whether forcing was triggered this step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, model: FoundationModel,
              source_snapshot: Optional[Dict[str, torch.Tensor]] = None,
              pretrained_snapshot: Optional[Dict[str, torch.Tensor]] = None) -> None:
        self.model = model
        params, self._param_names = self._collect_trainable_params(model)
        self._param_refs = params

        head_params = []
        backbone_params = []
        for name, p in zip(self._param_names, params):
            if 'classifier' in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.lr,
            })
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.lr * self.head_lr_multiplier,
            })
        if not param_groups:
            param_groups = params

        self.optimizer = torch.optim.SGD(
            param_groups, momentum=0.9, weight_decay=self.weight_decay,
        )

        self.source_weights = {}
        for name, p in zip(self._param_names, params):
            self.source_weights[name] = p.detach().cpu().clone()

        self.teacher_weights = {k: v.clone() for k, v in self.source_weights.items()}

        self.pretrained_weights = pretrained_snapshot

    def adapt_step(self, batch: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
        B = batch.shape[0]
        num_classes = self.model.get_num_classes()
        device = batch.device

        # ---- Save current (student) weights on CPU ----
        saved = {}
        for name, ref in zip(self._param_names, self._param_refs):
            saved[name] = ref.data.cpu().clone()

        # ---- Load teacher weights for pseudo-label computation ----
        # Blend with pretrained weights if ensemble is enabled (de-biases teacher for cross-domain)
        if self.teacher_ensemble_weight > 0 and self.pretrained_weights is not None:
            blended = {}
            for name in self.teacher_weights:
                if name in self.pretrained_weights:
                    blended[name] = (
                        self.teacher_ensemble_weight * self.pretrained_weights[name] +
                        (1 - self.teacher_ensemble_weight) * self.teacher_weights[name]
                    )
                else:
                    blended[name] = self.teacher_weights[name]
            self._load_weights(blended)
        else:
            self._load_weights(self.teacher_weights)

        # ---- Teacher pseudo-labels in eval mode (disable drop_path noise) ----
        was_training = self.model.training
        self.model.eval()

        avg_logits = torch.zeros(B, num_classes, device=device)
        with torch.no_grad():
            for _ in range(self.num_augmentations):
                aug = self._apply_augmentation(batch)
                out = self.model.forward(aug)
                avg_logits += out
        avg_logits /= self.num_augmentations
        avg_probs = F.softmax(avg_logits / self.teacher_temperature, dim=-1)

        if was_training:
            self.model.train()

        # ---- Restore student weights from CPU ----
        for name, ref in zip(self._param_names, self._param_refs):
            ref.data.copy_(saved[name].to(ref.device))
        del saved

        # ---- Student forward ----
        student_logits = self.model.forward(batch)
        student_probs = F.softmax(student_logits, dim=-1)

        teacher_conf, teacher_hard = avg_probs.max(dim=-1)

        # ---- Class-specific confidence thresholds (PLF paper: arxiv 2406.02609) ----
        # Instead of single global threshold, compute per-class thresholds based on
        # per-class confidence statistics. Classes with lower mean confidence get
        # lower thresholds to prevent class collapse.
        if self.use_class_specific_thresholds:
            per_class_thresholds = torch.full((num_classes,), self.confidence_threshold, device=device)
            for c in range(num_classes):
                class_mask = teacher_hard == c
                if class_mask.sum() > 0:
                    class_conf_mean = teacher_conf[class_mask].mean().item()
                    # Dynamic threshold: min(base_threshold, class_mean_conf * 0.8)
                    # This allows underconfident classes (like class-0) to get pseudo-labels
                    per_class_thresholds[c] = min(self.confidence_threshold, class_conf_mean * 0.8)
            # Apply per-class thresholds
            confident_mask = torch.zeros(B, device=device, dtype=torch.bool)
            for c in range(num_classes):
                class_mask = teacher_hard == c
                confident_mask |= (class_mask & (teacher_conf > per_class_thresholds[c]))
        else:
            confident_mask = teacher_conf > self.confidence_threshold

        # ---- Per-class cap: prevent any single class from dominating the KL signal ----
        # This is the primary defence against class-3 prototype hijack.
        # With per_class_cap=4 and B=16: even if all 16 samples are pseudo-labelled class-3,
        # only the 4 most confident are kept, leaving room for class-0 and others.
        if self.per_class_cap > 0 and confident_mask.sum() > 0:
            capped_mask = torch.zeros_like(confident_mask, dtype=torch.bool)
            for c in range(num_classes):
                class_mask = confident_mask & (teacher_hard == c)
                n_c = int(class_mask.sum().item())
                if n_c > self.per_class_cap:
                    # Keep only the top-per_class_cap most confident samples for this class
                    conf_c = teacher_conf * class_mask.float()
                    _, top_idx = conf_c.topk(self.per_class_cap)
                    capped_mask[top_idx] = True
                else:
                    capped_mask |= class_mask
            confident_mask = capped_mask

        # ---- Class-conditional pseudo-label forcing ----
        # When a class has 0 pseudo-labels for N consecutive batches, force-include
        # the most confident sample for that class regardless of threshold.
        # Breaks the positive feedback loop of class starvation.
        self._class_forcing_active = False
        if self.class_forcing_threshold > 0:
            for c in range(num_classes):
                if self._class_starvation_counts.get(c, 0) >= self.class_forcing_threshold:
                    # Find the sample where teacher assigns class c with highest confidence
                    class_c_mask = teacher_hard == c
                    if class_c_mask.sum() > 0:
                        class_c_conf = teacher_conf * class_c_mask.float()
                        best_idx = class_c_conf.argmax()
                        if not confident_mask[best_idx]:
                            confident_mask[best_idx] = True
                            self._class_forcing_active = True
                            logger.debug(f"Forced class-{c} pseudo-label at batch (starvation={self._class_starvation_counts[c]})")

            # Update starvation counts
            for c in range(num_classes):
                n_c = int((teacher_hard[confident_mask] == c).sum().item()) if confident_mask.sum() > 0 else 0
                if n_c == 0:
                    self._class_starvation_counts[c] = self._class_starvation_counts.get(c, 0) + 1
                else:
                    self._class_starvation_counts[c] = 0

        n_confident = int(confident_mask.sum().item())

        # ---- Loss components ----
        loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        diversity_loss = torch.tensor(0.0, device=device)

        # 1) Consistency loss: KL(teacher || student) on confident samples only
        #    With optional label smoothing and class-reliability weighting
        if n_confident > 0:
            target_probs = avg_probs[confident_mask]

            # Label smoothing: reduces the impact of incorrect pseudo-labels.
            # A wrong class-3 pseudo-label at peak=0.95 becomes peak≈0.875 with α=0.1,
            # reducing the gradient magnitude proportionally.
            if self.kl_label_smoothing > 0.0:
                smooth = self.kl_label_smoothing / num_classes
                target_probs = (1.0 - self.kl_label_smoothing) * target_probs + smooth

            if self.class_weights is not None:
                # Weight each sample's KL contribution by the baseline accuracy of its
                # pseudo-label class. Class-3 (23.5% accurate) gets weight 0.235;
                # class-0 (84.9% accurate) gets weight 0.849.
                # This softly down-votes unreliable pseudo-labels without discarding them.
                cw = torch.tensor(self.class_weights, device=device, dtype=torch.float32)
                conf_classes = teacher_hard[confident_mask]
                sample_weights = cw[conf_classes]
                # Normalise so total weight = n_confident (preserves KL magnitude scale)
                sample_weights = sample_weights / (sample_weights.sum() + 1e-8) * n_confident

                kl_per_sample = F.kl_div(
                    F.log_softmax(student_logits[confident_mask], dim=-1),
                    target_probs,
                    reduction="none",
                ).sum(dim=-1)
                kl_loss = (kl_per_sample * sample_weights).mean()
            else:
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits[confident_mask], dim=-1),
                    target_probs,
                    reduction="batchmean",
                )
            loss = loss + kl_loss

        # 2) Entropy minimization on ALL student outputs
        # Weighted low (entropy_weight) to avoid dominating when KL=0
        if self.entropy_weight > 0:
            s_ent_per_sample = -(student_probs * (student_probs.clamp_min(1e-10)).log()).sum(dim=-1)
            entropy_loss = s_ent_per_sample.mean()
            loss = loss + self.entropy_weight * entropy_loss

        # 3) Diversity regularization — always active to prevent mode collapse
        if self.diversity_weight > 0 and B > 1:
            avg_pred = student_probs.mean(dim=0)
            batch_diversity_entropy = -(avg_pred * (avg_pred + 1e-10).log()).sum()
            diversity_loss = batch_diversity_entropy
            loss = loss - self.diversity_weight * batch_diversity_entropy

        # 4) Class Prior Alignment (CPA) from PLF paper (arxiv 2406.02609)
        # Encourages diverse predictions by aligning batch class distribution to uniform prior
        # Prevents class collapse by penalizing concentration of predictions on few classes
        if self.class_prior_alignment_weight > 0 and B > 1:
            batch_class_dist = student_probs.mean(dim=0)  # [num_classes]
            uniform_prior = torch.ones(num_classes, device=device) / num_classes
            # KL divergence from uniform prior (lower when predictions are diverse)
            cpa_loss = F.kl_div(
                (batch_class_dist + 1e-10).log(),
                uniform_prior,
                reduction="batchmean",
            )
            loss = loss + self.class_prior_alignment_weight * cpa_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._param_refs, self.max_grad_norm)
        self.optimizer.step()

        # ---- Teacher EMA (CPU) ----
        self._ema_update()

        # ---- Stochastic restore to source (skip if teacher unconfident) ----
        skip_restore = self.confidence_gated_restore and n_confident == 0
        self._restore_parameters(skip=skip_restore)

        # ---- Metrics ----
        with torch.no_grad():
            max_probs = student_probs.max(dim=-1).values
            mean_max_prob = float(max_probs.mean().item())
            t_ent = -(avg_probs * (avg_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()
            s_ent = -(student_probs * (student_probs.clamp_min(1e-10)).log()).sum(-1).mean().item()

            # Per-class confident sample count (after capping) for diagnostics
            if n_confident > 0:
                conf_hard = teacher_hard[confident_mask]
                per_class_conf = {}
                for c in range(num_classes):
                    per_class_conf[int(c)] = int((conf_hard == c).sum().item())
            else:
                per_class_conf = {c: 0 for c in range(num_classes)}

        return {
            "loss": loss.item(),
            "kl_loss": float(kl_loss.item()),
            "teacher_entropy": t_ent,
            "student_entropy": s_ent,
            "entropy_loss": float(entropy_loss.item()),
            "diversity_loss": float(diversity_loss.item()),
            "restoration_count": self._restoration_count,
            "mean_max_prob": mean_max_prob,
            "num_confident_samples": n_confident,
            "per_class_confident": per_class_conf,
            "teacher_mean_max_prob": float(teacher_conf.mean().item()),
            "class_forcing_active": self._class_forcing_active,
            "starvation_counts": dict(self._class_starvation_counts),
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
            "pretrained_weights": self.pretrained_weights,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.source_weights = state.get("source_weights", {})
        self.teacher_weights = state.get("teacher_weights", {})
        self.pretrained_weights = state.get("pretrained_weights", None)
        if state.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(state["optimizer"])

    def set_teacher_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Overwrite teacher weights (e.g. restore post-source checkpoint)."""
        self.teacher_weights = {k: v.clone() for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_trainable_params(self, model: FoundationModel):
        """Return (list_of_params, list_of_names) for restricted trainable set.

        Adapts: LayerNorm throughout (that are trainable) + last N transformer
        blocks + classifier head.

        Note: blocks 0..freeze_layers-1 have requires_grad=False, so their
        LayerNorm parameters are automatically excluded by the requires_grad check.
        """
        names, params = [], []
        try:
            total_blocks = len(list(model.backbone.blocks))
        except AttributeError:
            total_blocks = 24

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_ln = ('norm' in n or 'ln' in n) and ('weight' in n or 'bias' in n)

            is_late_block = False
            if 'backbone.blocks.' in n:
                try:
                    block_idx = int(n.split('backbone.blocks.')[1].split('.')[0])
                    is_late_block = block_idx >= (total_blocks - self.adapt_last_n_blocks)
                except (IndexError, ValueError):
                    pass

            is_head = 'classifier' in n

            if (self.adapt_layernorm and is_ln) or is_late_block or is_head:
                names.append(n)
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

    def _restore_parameters(self, skip=False):
        if skip:
            self._restoration_count = 0
            return 0
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
        """Apply one of 7 retinal-specific augmentations on GPU.

        Channel permutation (original aug_type=7) removed (Round 3): it destroys
        color-diagnostic features (hemorrhage redness, exudate brightness)
        that are informative for DR grading. All 7 remaining types are
        grade-preserving by domain expertise.
        """
        import torchvision.transforms.functional as TF
        aug_type = torch.randint(0, 7, (1,)).item()   # 7 types, not 8
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
            blur = TF.gaussian_blur(batch, kernel_size=k, sigma=sigma)
            return blur
        elif aug_type == 4:
            brightness = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            return torch.clamp(batch * brightness.view(1, 1, 1, 1), 0, 1)
        elif aug_type == 5:
            contrast = 0.6 + 0.8 * torch.rand(1, device=batch.device)
            mean = batch.mean(dim=[2, 3], keepdim=True)
            return torch.clamp((batch - mean) * contrast.view(1, 1, 1, 1) + mean, 0, 1)
        else:  # aug_type == 6 — gaussian noise
            return torch.clamp(batch + 0.05 * torch.randn_like(batch), 0, 1)
