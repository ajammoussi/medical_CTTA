"""Deterministic full-stream datasets and split provenance.

Used by Component C1 (``src/agents/characterize.py``) and the agentic runners
(``stream_split: "full"``). Everything here builds streams with
``augment=False`` so that:

- the C1 descriptor profile and the online adaptation stream see *identical,
  deterministic* data (no random flip / color-jitter that would otherwise make
  the two populations differ), and
- ``classes_seen`` can be computed over a labelled subset that is exactly a
  subset of the stream, never a different distribution.

``stream_provenance`` is the *measure, don't assume* step: it reports the real
per-split row counts, label sets and any placeholder-label warning (e.g. an
unlabeled competition test split whose labels are all 0) so the APTOS stream
composition decision is made from data, not from a guess.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from torch.utils.data import ConcatDataset, Dataset

from src.data.registry import DatasetRegistry

logger = logging.getLogger(__name__)

ID = "id"
LABEL = "label"


def _make_split(
    name: str,
    data_dir: str,
    image_size: int,
    train: bool,
    normalize_mean: List[float],
    normalize_std: List[float],
    augment: bool,
) -> Dataset:
    ds_class = DatasetRegistry.get(name)
    return ds_class(
        data_dir=data_dir,
        image_size=image_size,
        train=train,
        normalize_mean=list(normalize_mean),
        normalize_std=list(normalize_std),
        augment=augment,
    )


def build_adaptation_stream(
    dataset_config,
    normalize_mean: List[float],
    normalize_std: List[float],
    augment: bool = False,
) -> Dataset:
    """ConcatDataset(train, test) normalized like the legacy eval/adapt path.

    Evaluation always stays on the test split (``_evaluate_model``), so the
    stream is exactly the deployment population the model will adapt on.
    """
    train_ds = _make_split(
        dataset_config.name, dataset_config.data_dir, dataset_config.image_size,
        train=True, normalize_mean=normalize_mean, normalize_std=normalize_std,
        augment=augment,
    )
    test_ds = _make_split(
        dataset_config.name, dataset_config.data_dir, dataset_config.image_size,
        train=False, normalize_mean=normalize_mean, normalize_std=normalize_std,
        augment=augment,
    )
    return ConcatDataset([train_ds, test_ds])


def build_raw_stream(dataset_config) -> Dataset:
    """ConcatDataset(train, test) of raw ``[0, 1]`` resized images (C1 pass).

    Uses identity normalization (mean 0 / std 1) and ``augment=False``, so the
    loaders' ``ToTensor`` transform yields exactly the unnormalized pixels the
    characterizer expects.
    """
    return build_adaptation_stream(
        dataset_config,
        normalize_mean=[0.0, 0.0, 0.0],
        normalize_std=[1.0, 1.0, 1.0],
        augment=False,
    )


def build_raw_split(dataset_config, train: bool) -> Dataset:
    """One raw ``[0, 1]`` split (train or test) of a dataset — used for the
    strictly-labeled subset that drives ``classes_seen`` (C1)."""
    return _make_split(
        dataset_config.name, dataset_config.data_dir, dataset_config.image_size,
        train=train, normalize_mean=[0.0, 0.0, 0.0], normalize_std=[1.0, 1.0, 1.0],
        augment=False,
    )


def split_row_counts(name: str, data_dir: str, image_size: int = 224) -> Dict[str, Dict[str, Any]]:
    """Per-split CSV row counts, unique-label sets and label histograms.

    Deterministic for every loader (Messidor-2 uses ``random_state=42``).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for train in (True, False):
        split_name = "train" if train else "test"
        ds = _make_split(
            name, data_dir, image_size, train=train,
            normalize_mean=[0.0, 0.0, 0.0], normalize_std=[1.0, 1.0, 1.0],
            augment=False,
        )
        labels = ds.df[LABEL].astype(int)
        counts = labels.value_counts().sort_index()
        out[split_name] = {
            "n_rows": int(len(labels)),
            "n_unique_labels": int(labels.nunique()),
            "label_counts": {int(k): int(v) for k, v in counts.items()},
        }
    return out


def stream_provenance(
    name: str, data_dir: str, image_size: int = 224,
) -> Dict[str, Any]:
    """Provenance report used to decide stream composition from data.

    Warns when one split has a single unique label while another has several —
    the classic signature of an unlabeled competition test split carrying
    placeholder ``0`` labels.
    """
    splits = split_row_counts(name, data_dir, image_size)
    n_train = splits["train"]["n_rows"]
    n_test = splits["test"]["n_rows"]
    uniq = {k: splits[k]["n_unique_labels"] for k in splits}
    suspicious = (
        n_train > 0 and n_test > 0 and uniq["test"] == 1 and uniq["train"] > 1
    )
    report = {
        "name": name,
        "data_dir": str(data_dir),
        "n_train": n_train,
        "n_test": n_test,
        "n_stream": n_train + n_test,
        "unique_labels_per_split": uniq,
        "splits": splits,
        "suspicious_placeholder_test": bool(suspicious),
        "notes": (
            "test split has a single unique label while train has several: "
            "treat test labels as placeholders for classes_seen"
            if suspicious else
            "labels appear consistent across splits"
        ),
    }
    return report


def all_stream_provenance(dataset_configs) -> Dict[str, Any]:
    """Run ``stream_provenance`` over several ``DatasetPath`` configs."""
    return {
        cfg.name: stream_provenance(cfg.name, cfg.data_dir, cfg.image_size)
        for cfg in dataset_configs
    }
