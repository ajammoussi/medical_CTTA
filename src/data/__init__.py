from src.data.registry import DatasetRegistry, register_dataset
from src.data.base import DatasetAdapter
from src.data.download import DatasetDownloader
# Import implementations to trigger registration
from src.data import idrid, aptos2019