import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = None, level: int = logging.INFO) -> None:
    """Setup logging with file + stdout. Adds handlers without removing
    existing file handlers, so run.log + per-domain adapt logs all capture
    the per-batch adapt lines."""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Single stdout handler (dedupe duplicates)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        target = str(Path(log_file).resolve())
        if not any(isinstance(h, logging.FileHandler)
                   and str(Path(h.baseFilename).resolve()) == target
                   for h in root.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)