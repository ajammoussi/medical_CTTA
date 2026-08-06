from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetAdapter(Dataset, ABC):
    """Abstract base class for dataset adapters."""

    def __init__(self, data_dir: str, image_size: int = 224, train: bool = True,
                 normalize_mean=None, normalize_std=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.train = train
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.transform = self._get_transforms()

    def _get_transforms(self) -> transforms.Compose:
        """Get data transforms based on train/test mode."""
        if self.train:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get item at index."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length."""
        pass

    @abstractmethod
    def get_class_distribution(self) -> dict:
        """Get class distribution."""
        pass