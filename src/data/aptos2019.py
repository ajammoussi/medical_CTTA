"""APTOS 2019 dataset loader for diabetic retinopathy grading.

Dataset: https://www.kaggle.com/datasets/mariaherrerot/aptos2019
- ~5,590 images (3,662 train + 1,928 test)
- Labels: id_code, diagnosis (grade 0-4)
- Extremely heterogeneous: multiple sites, cameras, quality
- ~73% grade 0 (severe class imbalance)
- Ideal target domain for CTTA (domain shift from IDRiD)

Kaggle structure:
    APTOS-2019 dataset/
    ├── train_1.csv
    ├── test.csv
    ├── valid.csv
    ├── train_images/
    │   └── train_images/
    │       ├── 000c1434d8d7.png
    │       └── ...
    ├── test_images/
    │   └── test_images/
    └── val_images/
        └── val_images/
"""

import os
import pandas as pd
from PIL import Image
from typing import Tuple, Dict, Optional, List
import torch
from src.data.base import DatasetAdapter
from src.data.registry import register_dataset


@register_dataset("aptos2019")
class APTOS2019Dataset(DatasetAdapter):
    """APTOS 2019 dataset adapter for diabetic retinopathy grading.

    Uses `train` flag to select the correct CSV and image directory:
    - train=True  → train_1.csv + train_images/
    - train=False → test.csv    + test_images/
    """

    TRAIN_CSV_NAMES = ["train_1.csv", "train.csv", "training.csv", "aptos2019.csv"]
    TEST_CSV_NAMES = ["test.csv", "testing.csv"]
    VAL_CSV_NAMES = ["valid.csv", "val.csv", "validation.csv"]

    TRAIN_IMAGE_DIRS = ["train_images", "train", "images"]
    TEST_IMAGE_DIRS = ["test_images", "test", "images"]
    VAL_IMAGE_DIRS = ["val_images", "val", "validation"]

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        train: bool = True,
        train_ratio: float = 0.8,
    ):
        super().__init__(data_dir, image_size, train)
        self.data_dir = data_dir
        self.image_size = image_size
        self.train = train
        self.transform = self._get_transforms()

        csv_path = self._find_csv()
        if csv_path is None:
            raise FileNotFoundError(
                f"No {'training' if train else 'testing'} CSV found in {data_dir}"
            )
        self.df = self._load_csv(csv_path)

        self.image_dir = self._find_image_dir()
        if self.image_dir is None:
            raise FileNotFoundError(
                f"No {'training' if train else 'testing'} image directory found in {data_dir}"
            )

    def _find_csv(self) -> Optional[str]:
        csv_names = self.TRAIN_CSV_NAMES if self.train else self.TEST_CSV_NAMES

        # Pass 1: known names in data_dir
        for csv_name in csv_names:
            csv_path = os.path.join(self.data_dir, csv_name)
            if os.path.exists(csv_path):
                return csv_path

        # Pass 2: known names in subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for csv_name in csv_names:
                if csv_name in files:
                    return os.path.join(root, csv_name)

        # Pass 3: any CSV with id+label columns
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.csv'):
                    csv_path = os.path.join(root, f)
                    if self._is_classification_csv(csv_path):
                        return csv_path

        # Pass 4: any CSV
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.csv'):
                    return os.path.join(root, f)
        return None

    def _is_classification_csv(self, csv_path: str) -> bool:
        classification_cols = {"id", "id_code", "image_id", "filename"}
        label_cols = {"diagnosis", "grade", "retinopathy", "label"}
        try:
            cols = set(pd.read_csv(csv_path, nrows=0).columns.str.strip().str.lower())
            return bool((cols & classification_cols) and (cols & label_cols))
        except Exception:
            return False

    def _find_image_dir(self) -> Optional[str]:
        img_dirs = self.TRAIN_IMAGE_DIRS if self.train else self.TEST_IMAGE_DIRS

        # Pass 1: known names in data_dir
        for img_dir in img_dirs:
            dir_path = os.path.join(self.data_dir, img_dir)
            if os.path.isdir(dir_path):
                # Check for nested directory (e.g. train_images/train_images/)
                nested = os.path.join(dir_path, img_dir)
                if os.path.isdir(nested):
                    if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(nested)):
                        return nested
                if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(dir_path)):
                    return dir_path

        # Pass 2: known names in subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for img_dir in img_dirs:
                dir_path = os.path.join(root, img_dir)
                if os.path.isdir(dir_path):
                    nested = os.path.join(dir_path, img_dir)
                    if os.path.isdir(nested):
                        if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(nested)):
                            return nested
                    if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(dir_path)):
                        return dir_path

        # Pass 3: any directory with images
        for root, dirs, files in os.walk(self.data_dir):
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                return root
        return None

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        col_mapping = {
            "id_code": "id",
            "id": "id",
            "image_id": "id",
            "filename": "id",
            "diagnosis": "label",
            "grade": "label",
            "retinopathy": "label",
            "label": "label",
        }
        df = df.rename(columns={
            c: col_mapping[c] for c in df.columns if c in col_mapping
        })
        if "id" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"CSV must have 'id' and 'label' columns. "
                f"Found: {list(df.columns)}"
            )
        return df

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image_id = str(row["id"])
        label = int(row["label"])
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f"Image not found: {image_id} in {self.image_dir}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.df)

    def get_class_distribution(self) -> Dict[int, int]:
        return self.df["label"].value_counts().sort_index().to_dict()
