import pytest
from src.data.download import DatasetDownloader


def test_downloader_initialization():
    downloader = DatasetDownloader()
    assert downloader.data_dir.exists()


def test_downloader_custom_dir(tmp_path):
    downloader = DatasetDownloader(data_dir=str(tmp_path / "custom"))
    assert downloader.data_dir.exists()


def test_verify_datasets():
    downloader = DatasetDownloader()
    results = downloader.verify_datasets()
    assert "idrid" in results
    assert "aptos2019" in results
    assert all("exists" in v for v in results.values())