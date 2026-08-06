"""VisionFM weight download utility.

Downloads VFM_Fundus_weights.pth from Google Drive.
Weights are cached locally to avoid re-downloading between runs.
"""

import os
import logging

logger = logging.getLogger(__name__)

VFM_FUNDUS_GDRIVE_ID = "13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4"
VFM_FUNDUS_URL = f"https://drive.google.com/uc?id={VFM_FUNDUS_GDRIVE_ID}"
VFM_FUNDUS_FILENAME = "VFM_Fundus_weights.pth"


def download_visionfm_weights(cache_dir: str = "./cache/visionfm") -> str:
    """Download VisionFM Fundus pretrained weights from Google Drive.

    Checks if weights are already cached. If not, downloads them.
    Requires ``gdown`` package (pip install gdown).

    Args:
        cache_dir: Directory to cache downloaded weights.

    Returns:
        Absolute path to the downloaded .pth file.

    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, VFM_FUNDUS_FILENAME)

    if os.path.exists(target_path):
        logger.info("VisionFM Fundus weights found at %s (cached)", target_path)
        return os.path.abspath(target_path)

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download VisionFM weights. "
            "Install it with: pip install gdown"
        )

    logger.info("Downloading VisionFM Fundus weights from Google Drive...")
    try:
        gdown.download(VFM_FUNDUS_URL, target_path, quiet=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download VisionFM weights from Google Drive: {e}. "
            f"Try downloading manually from: https://drive.google.com/file/d/{VFM_FUNDUS_GDRIVE_ID}/view"
        )

    if not os.path.exists(target_path):
        raise RuntimeError(f"Download completed but file not found at {target_path}")

    logger.info("VisionFM Fundus weights downloaded to %s", target_path)
    return os.path.abspath(target_path)
