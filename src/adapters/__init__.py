from src.adapters.registry import AdapterRegistry, register_adapter
from src.adapters.base import CTTAAdapter
# Import implementations to trigger registration
from src.adapters import cotta
from src.adapters import palm