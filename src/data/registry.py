from typing import Dict, Type
from src.data.base import DatasetAdapter


class DatasetRegistry:
    """Registry for dataset adapters."""

    _registry: Dict[str, Type[DatasetAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a dataset class."""
        def decorator(dataset_class: Type[DatasetAdapter]):
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DatasetAdapter]:
        """Get dataset class by name."""
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_datasets(cls) -> list:
        """List all registered datasets."""
        return list(cls._registry.keys())


def register_dataset(name: str):
    """Decorator to register a dataset class."""
    return DatasetRegistry.register(name)