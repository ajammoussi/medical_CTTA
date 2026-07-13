"""IDRiD dataset loader for diabetic retinopathy grading.

Dataset: https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset
- 516 images total (413 train / 103 test)
- Labels: ID, Diabetic Retinopathy (grade 0-4)
- Single camera (Kowa VX-10a), single clinic
- Ideal source domain for CTTA

Kaggle structure (B. Disease Grading):
    B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/    (IDRiD_001.jpg ... IDRiD_413.jpg)
    │   └── b. Testing Set/     (IDRiD_001.jpg ... IDRiD_103.jpg)
    └── 2. Groundtruth/
        ├── a. IDRiD_Disease_Grading_Training.csv
        └── b. IDRiD_Disease_Grading_Testing.csv
"""

import os
import pandas as pd
from PIL import Image
from typing import Tuple, Dict, Optional, List
import torch
from src.data.base import DatasetAdapter
from src.data.registry import register_dataset


@register_dataset("idrid")
class IDRiDDataset(DatasetAdapter):
    """IDRiD dataset adapter for diabetic retinopathy grading.

    Uses `train` flag to select the correct CSV and image directory:
    - train=True  → training CSV + a. Training Set/
    - train=False → testing CSV  + b. Testing Set/
    """

    # Known CSV names (order matters for preference)
    TRAIN_CSV_NAMES = [
        "a. IDRiD_Disease Grading_Training Labels.csv",
        "train.csv", "train_labels.csv", "training.csv",
    ]
    TEST_CSV_NAMES = [
        "b. IDRiD_Disease Grading_Testing Labels.csv",
        "test.csv", "testing.csv",
    ]
    # Fallback: any CSV with these column patterns
    CLASSIFICATION_COLS = {"id", "image_id", "image name", "filename",
                           "file", "image no"}
    LABEL_COLS = {"diabetic retinopathy", "diagnosis", "grade",
                  "retinopathy", "retinopathy grade", "label"}

    # Known image directory names (order matters for preference)
    TRAIN_IMAGE_DIRS = [
        "a. Training Set", "training", "train", "train_images",
    ]
    TEST_IMAGE_DIRS = [
        "b. Testing Set", "testing", "test", "test_images",
    ]

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
        # Pass 1: known names in data_dir
        csv_names = self.TRAIN_CSV_NAMES if self.train else self.TEST_CSV_NAMES
        for csv_name in csv_names:
            csv_path = os.path.join(self.data_dir, csv_name)
            if os.path.exists(csv_path):
                return csv_path

        # Pass 2: known names anywhere under data_dir
        for root, dirs, files in os.walk(self.data_dir):
            for csv_name in csv_names:
                if csv_name in files:
                    return os.path.join(root, csv_name)

        # Pass 3: find ALL CSVs, pick the one matching train/test
        all_csvs = self._find_all_csvs()
        train_keywords = {"training", "train"}
        test_keywords = {"testing", "test"}
        keywords = train_keywords if self.train else test_keywords

        for csv_path in all_csvs:
            lower = csv_path.lower()
            if any(kw in lower for kw in keywords):
                if self._is_classification_csv(csv_path):
                    return csv_path

        # Pass 4: any classification CSV
        for csv_path in all_csvs:
            if self._is_classification_csv(csv_path):
                return csv_path

        # Pass 5: any CSV
        if all_csvs:
            return all_csvs[0]
        return None

    def _find_all_csvs(self) -> List[str]:
        csvs = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith('.csv'):
                    csvs.append(os.path.join(root, f))
        return csvs

    def _is_classification_csv(self, csv_path: str) -> bool:
        try:
            cols = set(pd.read_csv(csv_path, nrows=0).columns.str.strip().str.lower())
            return bool((cols & self.CLASSIFICATION_COLS) and (cols & self.LABEL_COLS))
        except Exception:
            return False

    def _find_image_dir(self) -> Optional[str]:
        # Pass 1: known names in data_dir
        img_dirs = self.TRAIN_IMAGE_DIRS if self.train else self.TEST_IMAGE_DIRS
        for img_dir in img_dirs:
            dir_path = os.path.join(self.data_dir, img_dir)
            if os.path.isdir(dir_path):
                if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(dir_path)):
                    return dir_path

        # Pass 2: known names under Disease Grading
        for root, dirs, files in os.walk(self.data_dir):
            if 'disease grading' not in root.lower():
                continue
            for img_dir in img_dirs:
                dir_path = os.path.join(root, img_dir)
                if os.path.isdir(dir_path):
                    if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(dir_path)):
                        return dir_path

        # Pass 3: any directory with images under Disease Grading
        for root, dirs, files in os.walk(self.data_dir):
            if 'disease grading' not in root.lower():
                continue
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                return root

        # Pass 4: any directory with images (last resort)
        for root, dirs, files in os.walk(self.data_dir):
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                return root

        return None

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        col_mapping = {
            "id": "id",
            "image_id": "id",
            "image name": "id",
            "image no": "id",
            "filename": "id",
            "file": "id",
            "diabetic retinopathy": "label",
            "diagnosis": "label",
            "grade": "label",
            "retinopathy": "label",
            "retinopathy grade": "label",
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
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
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
