import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from src.shift_detection.base import ShiftDetector


class DistributionDriftDetector(ShiftDetector):
    """Shift detection using prediction distribution drift."""

    def __init__(self, distribution_threshold: float = 0.1, window_size: int = 100):
        self.distribution_threshold = distribution_threshold
        self.window_size = window_size
        self.baseline_distribution: Optional[torch.Tensor] = None
        self.window_predictions: list = []
        self.num_classes: Optional[int] = None

    def calibrate(self, predictions: list, num_classes: int = 5) -> None:
        self.num_classes = num_classes
        if len(predictions) == 0:
            raise ValueError("Cannot calibrate with empty predictions")
        counts = torch.zeros(num_classes)
        for pred in predictions:
            counts[pred] += 1
        self.baseline_distribution = counts / counts.sum()

    def update(self, batch: torch.Tensor, logits: torch.Tensor,
               embeddings: Optional[torch.Tensor] = None) -> None:
        preds = torch.argmax(logits, dim=-1)
        self.window_predictions.extend(preds.cpu().numpy().tolist())
        if len(self.window_predictions) > self.window_size:
            self.window_predictions = self.window_predictions[-self.window_size:]

    def detect(self) -> bool:
        if self.baseline_distribution is None or len(self.window_predictions) == 0:
            return False
        counts = torch.zeros(self.num_classes)
        for pred in self.window_predictions:
            counts[pred] += 1
        current_distribution = counts / counts.sum()
        jsd = self._jensen_shannon_divergence(current_distribution, self.baseline_distribution)
        return jsd > self.distribution_threshold

    def get_signals(self) -> Dict[str, float]:
        if self.baseline_distribution is None or len(self.window_predictions) == 0:
            return {"jsd": 0.0, "current_distribution": None}
        counts = torch.zeros(self.num_classes)
        for pred in self.window_predictions:
            counts[pred] += 1
        current_distribution = counts / counts.sum()
        jsd = self._jensen_shannon_divergence(current_distribution, self.baseline_distribution)
        return {
            "jsd": jsd,
            "current_distribution": current_distribution.tolist(),
            "baseline_distribution": self.baseline_distribution.tolist(),
            "window_size": len(self.window_predictions)
        }

    def reset(self) -> None:
        self.window_predictions.clear()

    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute Jensen-Shannon divergence with correct KL direction."""
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        kl_pm = F.kl_div(m.log(), p, reduction='sum')
        kl_qm = F.kl_div(m.log(), q, reduction='sum')
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.item()
