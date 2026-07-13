import pytest
from src.data.registry import DatasetRegistry
from src.data.base import DatasetAdapter


def test_dataset_registry():
    assert 'idrid' in DatasetRegistry.list_datasets()
    assert 'aptos2019' in DatasetRegistry.list_datasets()


def test_get_dataset_class():
    idrid_class = DatasetRegistry.get('idrid')
    assert issubclass(idrid_class, DatasetAdapter)
    aptos_class = DatasetRegistry.get('aptos2019')
    assert issubclass(aptos_class, DatasetAdapter)


def test_invalid_dataset():
    with pytest.raises(ValueError):
        DatasetRegistry.get('invalid_dataset')