"""Environment bootstrap for medical_CTTA notebooks.

Detects whether we're running on Kaggle or locally,
and performs the right setup for each.

Usage in any notebook (first cell):
    from src.env import init
    init()
"""

import sys
import os
import subprocess
import glob


def _is_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def _find_project_root_from_cwd() -> str:
    """Walk up from CWD to find setup.py."""
    d = os.getcwd()
    while True:
        if os.path.exists(os.path.join(d, "setup.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return ""


def _ensure_kaggle_pth():
    """Install a .pth file so Python finds src/ before any notebook cell runs.

    On Kaggle, CWD is /kaggle/working but the project is in a subdirectory.
    The .pth file adds the project root to sys.path at interpreter startup,
    solving the chicken-and-egg problem of 'from src.env import init'.
    """
    project_root = "/kaggle/working/medical_CTTA"
    if not os.path.isdir(project_root):
        return

    try:
        import site
        site_dir = site.getsitepackages()[0]
    except Exception:
        return

    pth_path = os.path.join(site_dir, "medical_CTTA.pth")
    if os.path.exists(pth_path):
        return

    try:
        with open(pth_path, "w") as f:
            f.write(project_root + "\n")
    except Exception:
        pass


def _ensure_kaggle_data_symlink(project_root: str):
    """On Kaggle, symlink project data/ to /kaggle/working/data/ so datasets
    survive code overwrites (user copies local project with empty data/)."""
    persistent_data = "/kaggle/working/data"
    project_data = os.path.join(project_root, "data")

    os.makedirs(persistent_data, exist_ok=True)

    if os.path.islink(project_data):
        # Already a symlink — verify target
        if os.readlink(project_data) == persistent_data:
            return
        os.remove(project_data)
    elif os.path.isdir(project_data):
        # Real directory — check if it has content
        if os.listdir(project_data):
            # Has content (datasets downloaded here) — move to persistent location
            import shutil
            for item in os.listdir(project_data):
                src = os.path.join(project_data, item)
                dst = os.path.join(persistent_data, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(project_data, ignore_errors=True)
        else:
            # Empty dir — safe to replace with symlink
            os.rmdir(project_data)
    elif os.path.exists(project_data):
        os.remove(project_data)

    # Create symlink: data/ -> /kaggle/working/data/
    os.symlink(persistent_data, project_data)
    print(f"[env] Dataset symlink: {project_data} -> {persistent_data}")


def _init_kaggle():
    # Ensure the .pth file exists for future kernel restarts
    _ensure_kaggle_pth()

    # Find project root from CWD (handles any nesting depth)
    project_root = _find_project_root_from_cwd()

    if project_root:
        os.chdir(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    else:
        # Fallback: try known location
        known = "/kaggle/working/medical_CTTA"
        if os.path.isdir(known):
            os.chdir(known)
            if known not in sys.path:
                sys.path.insert(0, known)

    # Persist datasets outside project dir to survive code overwrites
    project_root = project_root or "/kaggle/working/medical_CTTA"
    _ensure_kaggle_data_symlink(project_root)


def _init_local():
    """Local dev — package should already be pip install -e ."""
    pass


def init():
    """Detect environment and perform the right setup.

    After calling this, you can do:
        from src.config import load_config
        config = load_config("configs/default.yaml")
    """
    if _is_kaggle():
        _init_kaggle()
    else:
        _init_local()
