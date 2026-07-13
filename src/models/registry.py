from typing import Dict, Type
from src.models.base import FoundationModel


class ModelRegistry:
    """Registry for foundation models."""

    _registry: Dict[str, Type[FoundationModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a model class."""
        def decorator(model_class: Type[FoundationModel]):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[FoundationModel]:
        """Get model class by name."""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._registry.keys())


def register_model(name: str):
    """Decorator to register a model class."""
    return ModelRegistry.register(name)