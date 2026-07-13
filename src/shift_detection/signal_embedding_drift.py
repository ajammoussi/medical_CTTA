import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
from src.shift_detection.base import ShiftDetector


class EmbeddingDriftDetector(ShiftDetector):
    """Shift detection using embedding drift."""

    def __init__(self, drift_threshold: float = 0.1, window_size: int = 100):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.running_mean: Optional[torch.Tensor] = None
        self.window_embeddings: list = []
        self.baseline_mean: Optional[torch.Tensor] = None

    def calibrate(self, embeddings: Union[list, torch.Tensor]) -> None:
        """Calibrate baseline embedding mean from source domain."""
        if isinstance(embeddings, list):
            if len(embeddings) == 0:
                raise ValueError("Cannot calibrate with empty embeddings")
            stacked = torch.cat(embeddings, dim=0)
        else:
            stacked = embeddings
        self.baseline_mean = stacked.mean(dim=0).detach()
        self.running_mean = self.baseline_mean.clone()

    def update(self, batch: torch.Tensor, logits: torch.Tensor,
               embeddings: Optional[torch.Tensor] = None) -> None:
        if embeddings is None:
            return
        embedding = embeddings.mean(dim=0)
        if self.running_mean is not None:
            self.running_mean = self.running_mean.to(embedding.device)
        self.window_embeddings.append(embedding.detach().cpu())
        if len(self.window_embeddings) > self.window_size:
            self.window_embeddings.pop(0)
        if self.running_mean is None:
            self.running_mean = embedding.detach()
        else:
            self.running_mean = 0.99 * self.running_mean + 0.01 * embedding.detach()

    def detect(self) -> bool:
        if self.baseline_mean is None or self.running_mean is None:
            return False
        device = self.running_mean.device
        baseline = self.baseline_mean.to(device)
        cosine_sim = F.cosine_similarity(
            self.running_mean.unsqueeze(0),
            baseline.unsqueeze(0)
        ).item()
        drift = 1.0 - cosine_sim
        return drift > self.drift_threshold

    def get_signals(self) -> Dict[str, float]:
        if self.baseline_mean is None or self.running_mean is None:
            return {"drift": 0.0, "cosine_sim": 1.0}
        device = self.running_mean.device
        baseline = self.baseline_mean.to(device)
        cosine_sim = F.cosine_similarity(
            self.running_mean.unsqueeze(0),
            baseline.unsqueeze(0)
        ).item()
        return {
            "drift": 1.0 - cosine_sim,
            "cosine_sim": cosine_sim,
            "window_size": len(self.window_embeddings)
        }

    def reset(self) -> None:
        self.running_mean = None
        self.window_embeddings.clear()
