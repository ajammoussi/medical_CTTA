#!/usr/bin/env python3
"""
Kaggle code sync utility.

Bundles the local medical_CTTA project as a Kaggle Dataset version,
so notebooks can import code from /kaggle/input/ without internet.

Usage:
    python scripts/kaggle_sync.py          # Bundle code into Kaggle Dataset
    python scripts/kaggle_sync.py --push   # Bundle + upload to Kaggle
    python scripts/kaggle_sync.py --help   # Show instructions
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SLUG = "<username>/medical-ctta-code"


def check_kaggle_cli():
    """Check if the Kaggle CLI is installed."""
    try:
        result = subprocess.run(["kaggle", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def bundle_code():
    """Bundle the project source into a temporary directory for dataset creation."""
    tmp = tempfile.mkdtemp(prefix="kaggle_sync_")
    dst = os.path.join(tmp, "medical_ctta_code")

    # Copy only what's needed: src/, configs/, setup.py, requirements.txt
    os.makedirs(dst, exist_ok=True)

    for item in ["src", "configs", "setup.py", "requirements.txt"]:
        src = os.path.join(PROJECT_ROOT, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst_item,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif os.path.isfile(src):
            shutil.copy2(src, dst_item)

    # Create dataset-metadata.json
    metadata = {
        "title": "medical_ctta_code",
        "id": DEFAULT_SLUG,
        "licenses": [{"name": "MIT"}]
    }
    import json
    with open(os.path.join(tmp, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return tmp, dst


def print_instructions():
    print("=" * 60)
    print("  Kaggle Code Sync Instructions")
    print("=" * 60)
    print()
    print("  This script bundles your local code into a Kaggle Dataset")
    print("  that notebooks can import from /kaggle/input/.")
    print()
    print("  1. Install Kaggle CLI:")
    print("     pip install kaggle")
    print()
    print("  2. Place your API token:")
    print("     ~/.kaggle/kaggle.json  (from kaggle.com/settings -> API)")
    print()
    print("  3. Run the sync:")
    print("     python scripts/kaggle_sync.py --push")
    print()
    print("  4. In your Kaggle notebook, add this dataset:")
    print("     Settings -> Add Data -> search 'medical-ctta-code'")
    print()
    print("  5. Import in notebook:")
    print("     import sys")
    print('     sys.path.insert(0, "/kaggle/input/medical-ctta-code/medical_ctta_code")')
    print('     sys.path.insert(0, "/kaggle/input/medical-ctta-code/medical_ctta_code/src")')
    print()


def main():
    parser = argparse.ArgumentParser(description="Sync code to Kaggle Dataset")
    parser.add_argument("--push", action="store_true",
                        help="Bundle + upload to Kaggle")
    parser.add_argument("--bundle-only", action="store_true",
                        help="Only bundle the code (no upload)")
    args = parser.parse_args()

    if not (args.push or args.bundle_only):
        print_instructions()
        return

    tmp_dir, bundle_path = bundle_code()
    print(f"Code bundled at: {bundle_path}")

    if args.bundle_only:
        print(f"\nTo create dataset manually, run from {tmp}:")
        print(f"  kaggle datasets create -p {tmp}")
        return

    if not check_kaggle_cli():
        print("ERROR: Kaggle CLI not found. Install: pip install kaggle")
        print(f"Bundle saved at {bundle_path} for manual upload.")
        sys.exit(1)

    print("Uploading to Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", tmp],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode == 0:
        print("Dataset created/updated!")
        print(f"Slug: {DEFAULT_SLUG}")
    else:
        # Might already exist - try version
        print(f"Create failed (may already exist). Trying version update...")
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", tmp, "-m", "code update"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("Dataset version updated!")
        else:
            print(f"Failed: {result.stderr}")
            print(f"Bundle saved at {bundle_path} -- upload manually.")

    # cleanup
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()