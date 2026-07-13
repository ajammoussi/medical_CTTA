import torch
import torch.nn.functional as F
from typing import Dict, Optional
from src.shift_detection.base import ShiftDetector


class EMAShiftDetector(ShiftDetector):
    """Shift detection using EMA of prediction entropy."""

    def __init__(self, entropy_threshold: float = 0.3, ema_momentum: float = 0.9):
        self.entropy_threshold = entropy_threshold
        self.ema_momentum = ema_momentum
        self.ema_entropy = 0.0
        self.baseline_entropy = 0.0
        self.is_calibrated = False

    def calibrate(self, entropy_values: list) -> None:
        """Calibrate baseline entropy from source domain."""
        self.baseline_entropy = sum(entropy_values) / len(entropy_values)
        self.is_calibrated = True

    def update(self, batch: torch.Tensor, logits: torch.Tensor,
               embeddings: Optional[torch.Tensor] = None) -> None:
        """Update EMA entropy with new batch."""
        probs = F.softmax(logits, dim=-1)
        entropy = self._compute_entropy(probs)

        # Update EMA
        if self.ema_entropy == 0.0:
            self.ema_entropy = entropy
        else:
            self.ema_entropy = self.ema_momentum * self.ema_entropy + (1 - self.ema_momentum) * entropy

    def detect(self) -> bool:
        """Detect if shift occurred based on entropy increase."""
        if not self.is_calibrated:
            return False
        return self.ema_entropy - self.baseline_entropy > self.entropy_threshold

    def get_signals(self) -> Dict[str, float]:
        """Get current signal values."""
        return {
            "ema_entropy": self.ema_entropy,
            "baseline_entropy": self.baseline_entropy,
            "entropy_diff": self.ema_entropy - self.baseline_entropy
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.ema_entropy = 0.0

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute mean entropy of probability distribution."""
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean().item()