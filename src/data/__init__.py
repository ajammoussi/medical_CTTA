from src.data.registry import DatasetRegistry, register_dataset
from src.data.base import DatasetAdapter
from src.data.download import DatasetDownloader
from src.data.fullstream import (
    all_stream_provenance,
    build_adaptation_stream,
    build_raw_split,
    build_raw_stream,
    split_row_counts,
    stream_provenance,
)

# Import implementations to trigger registration
from src.data import idrid, aptos2019, messidor2