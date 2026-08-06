"""Messidor-2 dataset loader for diabetic retinopathy grading.

Dataset: https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess
- 1,748 images (874 examinations, 2 images per eye)
- Labels: id_code, diagnosis (grade 0-4), adjudicated_dme, adjudicated_gradable
- Topcon TRC NW6 camera (45° FOV), non-mydriatic, French population
- ~97% gradable (4 images with adjudicated_gradable=0 lack DR/DME labels)

Kaggle structure (messidor2preprocess):
    messidor-2/
    ├── messidor-2/
    │   └── preprocess/
    │       ├── 20051020_61757_0100_PP.png
    │       ├── 20051020_61804_0100_PP.png
    │       └── ...
    └── messidor_data.csv

CSV columns:
    id_code:          image filename (e.g. "20051020_61757_0100_PP")
    diagnosis:        DR grade (0=None, 1=Mild, 2=Moderate, 3=Severe, 4=PDR)
    adjudicated_dme:  0=No DME, 1=Referable DME
    adjudicated_gradable: 0=Ungradable, 1=Gradable

Note: No predefined train/test split. The loader creates one via train_ratio.
"""

import os
import pandas as pd
from PIL import Image
from typing import Tuple, Dict, Optional, List
import torch
from src.data.base import DatasetAdapter
from src.data.registry import register_dataset


@register_dataset("messidor2")
class Messidor2Dataset(DatasetAdapter):
    """Messidor-2 dataset adapter for diabetic retinopathy grading.

    Unlike IDRiD/APTOS, Messidor-2 has no predefined train/test split.
    The loader creates one using ``train_ratio`` (default 0.8).
    Ungradable images (adjudicated_gradable=0) are excluded.
    """

    # Known CSV names (order matters for preference)
    CSV_NAMES = [
        "messidor_data.csv",
        "messidor2.csv",
        "messidor.csv",
        "train.csv",
        "test.csv",
    ]

    # Fallback: any CSV with these column patterns
    CLASSIFICATION_COLS = {"id", "id_code", "image_id", "filename", "file"}
    LABEL_COLS = {"diagnosis", "grade", "retinopathy", "label"}

    # Known image directory names (order matters for preference)
    IMAGE_DIRS = [
        "preprocess",
        "images",
        "messidor-2",
        "messidor2",
    ]

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        train: bool = True,
        train_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(data_dir, image_size, train, **kwargs)
        self.data_dir = os.path.realpath(data_dir)
        self.image_size = image_size
        self.train = train
        self.train_ratio = train_ratio
        self.transform = self._get_transforms()

        csv_path = self._find_csv()
        if csv_path is None:
            raise FileNotFoundError(
                f"No Messidor-2 CSV found in {self.data_dir}"
            )
        self.df = self._load_csv(csv_path)

        self.image_dir = self._find_image_dir()
        if self.image_dir is None:
            raise FileNotFoundError(
                f"No image directory found in {self.data_dir}"
            )
        self.image_dir = os.path.realpath(self.image_dir)

    def _find_csv(self) -> Optional[str]:
        # Pass 1: known names in data_dir
        for csv_name in self.CSV_NAMES:
            csv_path = os.path.join(self.data_dir, csv_name)
            if os.path.exists(csv_path):
                return csv_path

        # Pass 2: known names anywhere under data_dir
        for root, dirs, files in os.walk(self.data_dir):
            for csv_name in self.CSV_NAMES:
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
        try:
            cols = set(pd.read_csv(csv_path, nrows=0).columns.str.strip().str.lower())
            return bool((cols & self.CLASSIFICATION_COLS) and (cols & self.LABEL_COLS))
        except Exception:
            return False

    _IMAGE_EXTS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP')

    def _find_image_dir(self) -> Optional[str]:
        # Pass 1: known names in data_dir
        for img_dir in self.IMAGE_DIRS:
            dir_path = os.path.join(self.data_dir, img_dir)
            if os.path.isdir(dir_path):
                if any(f.endswith(self._IMAGE_EXTS) for f in os.listdir(dir_path)):
                    return dir_path

        # Pass 2: known names in subdirectories (handles nested messidor-2/messidor-2/)
        for root, dirs, files in os.walk(self.data_dir):
            for img_dir in self.IMAGE_DIRS:
                dir_path = os.path.join(root, img_dir)
                if os.path.isdir(dir_path):
                    if any(f.endswith(self._IMAGE_EXTS) for f in os.listdir(dir_path)):
                        return dir_path

        # Pass 3: any directory with images
        for root, dirs, files in os.walk(self.data_dir):
            if any(f.endswith(self._IMAGE_EXTS) for f in files):
                return root

        return None

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Map Messidor-2 column names to standard id/label
        col_mapping = {
            "id_code": "id",
            "id": "id",
            "image_id": "id",
            "filename": "id",
            "file": "id",
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

        # Strip file extensions and whitespace from id column (case-insensitive)
        df["id"] = df["id"].astype(str).str.strip().str.replace(
            r"(?i)\.(png|jpg|jpeg|bmp)$", "", regex=True
        )

        # Filter out ungradable images if gradable column exists
        if "adjudicated_gradable" in df.columns:
            before = len(df)
            df = df[df["adjudicated_gradable"] == 1].reset_index(drop=True)
            filtered = before - len(df)
            if filtered > 0:
                import logging
                logging.getLogger(__name__).info(
                    "Filtered %d ungradable images from Messidor-2", filtered
                )

        # Create train/test split (Messidor-2 has no predefined split)
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(
            df, train_size=self.train_ratio, random_state=42,
            stratify=df["label"] if df["label"].nunique() > 1 else None,
        )
        self.df_full = df  # keep full df for class distribution
        self.df = df_train if self.train else df_test

        return self.df

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image_id = str(row["id"])
        label = int(row["label"])
        image_path = None
        for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".BMP"]:
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
        return self.df_full["label"].value_counts().sort_index().to_dict()
