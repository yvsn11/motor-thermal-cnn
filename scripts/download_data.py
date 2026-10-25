"""Download the Kaggle "Electric Motor Temperature" dataset into data/.

Usage: python scripts/download_data.py [--dest DIR] [--link]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATASET = "wkirgsn/electric-motor-temperature"
FILENAME = "measures_v2.csv"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data"

REMINDER = """\
Reminder
  - The dataset licence has not been verified by this project; check the terms on
    https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
  - Do not redistribute the raw data (data/ is gitignored).
  - Cite: Kirchgassner, Wallscheid, Bocker, IEEE Trans. Power Electronics 36(7), 2021,
    doi 10.1109/tpel.2020.3045596"""


class DownloadError(RuntimeError):
    """Raised with a user-readable message when the download cannot be done."""


def fetch(dataset: str = DATASET) -> Path:
    """Download the dataset with kagglehub and return the local directory."""
    try:
        import kagglehub
    except ImportError as exc:
        raise DownloadError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc
    try:
        return Path(kagglehub.dataset_download(dataset))
    except Exception as exc:  # kagglehub raises several error types
        raise DownloadError(
            f"Could not download '{dataset}': {exc}\n"
            "Check your Kaggle credentials (~/.kaggle/kaggle.json, or the KAGGLE_USERNAME "
            "and KAGGLE_KEY environment variables) and your network access."
        ) from exc


def install_file(src_dir: Path, dest_dir: Path, link: bool = False) -> Path:
    """Copy (or symlink) measures_v2.csv from src_dir into dest_dir."""
    matches = sorted(src_dir.rglob(FILENAME))
    if not matches:
        raise DownloadError(f"{FILENAME} not found in the downloaded dataset at {src_dir}")
    src = matches[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / FILENAME
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if link:
        dest.symlink_to(src)
    else:
        shutil.copy2(src, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="target directory")
    parser.add_argument("--link", action="store_true", help="symlink instead of copying")
    args = parser.parse_args(argv)
    try:
        src_dir = fetch()
        dest = install_file(src_dir, args.dest, link=args.link)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset ready: {dest}")
    print(REMINDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
