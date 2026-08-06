"""Dataset download utilities using kagglehub (official Kaggle library).

Usage:
  from data.download import DatasetDownloader
  dl = DatasetDownloader(data_dir="./data")
  dl.download_idrid()
  dl.download_aptos2019()

No kaggle.json needed. For public datasets kagglehub works out of the box.
If a dataset requires accepting terms, set KAGGLE_API_TOKEN as an environment
variable (from kaggle.com/settings/api).
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _download_with_kagglehub(dataset_slug: str, dest_path: Path, force: bool = False) -> Path:
    import kagglehub

    dest_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s via kagglehub ...", dataset_slug)
    cache_path = Path(kagglehub.dataset_download(dataset_slug, force_download=force))

    for item in cache_path.iterdir():
        dest_item = dest_path / item.name
        if item.is_dir():
            if dest_item.exists():
                shutil.rmtree(dest_item)
            shutil.copytree(item, dest_item)
        else:
            shutil.copy2(item, dest_item)

    logger.info("%s -> %s", dataset_slug, dest_path)
    return dest_path


class DatasetDownloader:
    """Download datasets from Kaggle using kagglehub."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(
        self,
        dataset_slug: str,
        dest_name: str,
        force: bool = False,
    ) -> Path:
        dest_path = self.data_dir / dest_name
        if dest_path.exists() and not force:
            logger.info("Dataset already exists at %s, skipping", dest_path)
            return dest_path
        return _download_with_kagglehub(dataset_slug, dest_path, force=force)

    def download_idrid(self, force: bool = False) -> Path:
        return self.download_kaggle_dataset(
            dataset_slug="aaryapatel98/indian-diabetic-retinopathy-image-dataset",
            dest_name="IDRiD",
            force=force,
        )

    def download_aptos2019(self, force: bool = False) -> Path:
        return self.download_kaggle_dataset(
            dataset_slug="mariaherrerot/aptos2019",
            dest_name="APTOS2019",
            force=force,
        )

    def download_messidor2(self, force: bool = False) -> Path:
        dest_path = self.data_dir / "MESSIDOR2"
        if dest_path.exists() and not force:
            logger.info("Dataset already exists at %s, skipping", dest_path)
            return dest_path

        # Try kagglehub download first
        try:
            return self.download_kaggle_dataset(
                dataset_slug="mariaherrerot/messidor2preprocess",
                dest_name="MESSIDOR2",
                force=force,
            )
        except Exception as e:
            logger.warning("kagglehub download failed: %s", e)

        # Fallback: check /kaggle/input/ for manually attached dataset
        kaggle_input_candidates = [
            "/kaggle/input/messidor2preprocess",
            "/kaggle/input/messidor-2",
            "/kaggle/input/messidor2",
        ]
        for candidate in kaggle_input_candidates:
            if os.path.isdir(candidate):
                logger.info("Found Messidor-2 at %s, copying to %s", candidate, dest_path)
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(candidate, dest_path, dirs_exist_ok=True)
                return dest_path

        raise FileNotFoundError(
            f"Could not download or find Messidor-2 dataset. "
            f"Tried kagglehub and {kaggle_input_candidates}"
        )

    def download_all(self, force: bool = False) -> dict:
        paths = {}
        paths["idrid"] = self.download_idrid(force=force)
        paths["aptos2019"] = self.download_aptos2019(force=force)
        paths["messidor2"] = self.download_messidor2(force=force)
        return paths

    def verify_datasets(self) -> dict:
        results = {}
        idrid_path = self.data_dir / "IDRiD"
        results["idrid"] = {
            "exists": idrid_path.exists(),
            "path": str(idrid_path),
            "has_images": any(idrid_path.rglob("*.jpg")) if idrid_path.exists() else False,
            "has_csv": any(idrid_path.rglob("*.csv")) if idrid_path.exists() else False,
        }
        aptos_path = self.data_dir / "APTOS2019"
        results["aptos2019"] = {
            "exists": aptos_path.exists(),
            "path": str(aptos_path),
            "has_images": any(aptos_path.rglob("*.png")) if aptos_path.exists() else False,
            "has_csv": any(aptos_path.rglob("*.csv")) if aptos_path.exists() else False,
        }
        messidor_path = self.data_dir / "MESSIDOR2"
        results["messidor2"] = {
            "exists": messidor_path.exists(),
            "path": str(messidor_path),
            "has_images": any(messidor_path.rglob("*.[jJ][pP][gG]")) or any(messidor_path.rglob("*.[pP][nN][gG]")) if messidor_path.exists() else False,
            "has_csv": any(messidor_path.rglob("*.csv")) if messidor_path.exists() else False,
        }
        return results
