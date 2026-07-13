import torch
from typing import Dict, List
from src.shift_detection.base import ShiftDetector
from src.shift_detection.signal_ema_entropy import EMAShiftDetector
from src.shift_detection.signal_embedding_drift import EmbeddingDriftDetector
from src.shift_detection.signal_distribution import DistributionDriftDetector


class CombinedShiftDetector(ShiftDetector):
    """Combined shift detection using multiple signals."""

    def __init__(self, entropy_threshold: float = 0.3, drift_threshold: float = 0.05,
                 distribution_threshold: float = 0.05, ema_momentum: float = 0.9,
                 window_size: int = 50):
        self.entropy_detector = EMAShiftDetector(entropy_threshold, ema_momentum)
        self.drift_detector = EmbeddingDriftDetector(drift_threshold, window_size)
        self.distribution_detector = DistributionDriftDetector(distribution_threshold, window_size)
        self.signals: List[str] = []

    def calibrate(self, entropy_values: list, embeddings: list, predictions: list, num_classes: int = 5) -> None:
        """Calibrate all detectors from source domain."""
        self.entropy_detector.calibrate(entropy_values)
        self.drift_detector.calibrate(embeddings)
        self.distribution_detector.calibrate(predictions, num_classes)

    def update(self, batch: torch.Tensor, logits: torch.Tensor, embeddings: torch.Tensor = None) -> None:
        """Update all detectors with new batch."""
        self.entropy_detector.update(batch, logits)
        if embeddings is not None:
            self.drift_detector.update(batch, logits, embeddings=embeddings)
        self.distribution_detector.update(batch, logits)

    def detect(self) -> bool:
        """Combined trigger logic: (signal1 > θ₁ AND signal2 > θ₂) OR signal3 > θ₃"""
        signal1 = self.entropy_detector.detect()
        signal2 = self.drift_detector.detect()
        signal3 = self.distribution_detector.detect()

        self.signals = []
        if signal1:
            self.signals.append("entropy")
        if signal2:
            self.signals.append("embedding_drift")
        if signal3:
            self.signals.append("distribution_drift")

        return signal1 or signal2 or signal3

    def get_signals(self) -> Dict[str, float]:
        """Get all signal values."""
        signals = {}
        signals.update(self.entropy_detector.get_signals())
        signals.update(self.drift_detector.get_signals())
        signals.update(self.distribution_detector.get_signals())
        signals["triggered_signals"] = self.signals
        return signals

    def reset(self) -> None:
        """Reset all detectors."""
        self.entropy_detector.reset()
        self.drift_detector.reset()
        self.distribution_detector.reset()
        self.signals.clear()