from typing import Dict, Type
from src.adapters.base import CTTAAdapter


class AdapterRegistry:
    """Registry for CTTA adapters."""

    _registry: Dict[str, Type[CTTAAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Register an adapter class."""
        def decorator(adapter_class: Type[CTTAAdapter]):
            cls._registry[name] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[CTTAAdapter]:
        """Get adapter class by name."""
        if name not in cls._registry:
            raise ValueError(f"Adapter '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_adapters(cls) -> list:
        """List all registered adapters."""
        return list(cls._registry.keys())


def register_adapter(name: str):
    """Decorator to register an adapter class."""
    return AdapterRegistry.register(name)