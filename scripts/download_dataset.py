"""Endoscapes2023 download helper.

The dataset is not redistributed with this repository. `ensure_dataset` checks
whether it is already present and, if not, fetches it from the official CAMMA
release, unpacks it and deletes the archive.
"""
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
REQUIRED_SPLITS = ("train", "val", "test")


def _is_present(dataset_dir: Path) -> bool:
    """The dataset is usable once every split directory holds frames."""
    return all((dataset_dir / split).is_dir() and any((dataset_dir / split).glob("*.jpg"))
               for split in REQUIRED_SPLITS)


def _format_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}TB"


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    """Stream `url` to `destination`, reporting progress on one line.

    `timeout` applies to each socket operation, not to the transfer as a whole,
    so a slow-but-alive download is never cut short.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(destination, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {_format_size(downloaded)} / {_format_size(total)}"
                          f"  ({downloaded / total:.1%})", end="", flush=True)
                else:
                    print(f"\r  {_format_size(downloaded)}", end="", flush=True)
    print()


def ensure_dataset(dataset_dir, url: str = DATASET_URL) -> Path:
    """Return `dataset_dir`, downloading and unpacking the dataset if missing.

    The archive unpacks to an `endoscapes/` directory, so it is extracted into
    the parent of `dataset_dir` (`./dataset` for the default `./dataset/endoscapes`).
    """
    dataset_dir = Path(dataset_dir)

    if _is_present(dataset_dir):
        print(f"Dataset found at {dataset_dir}")
        return dataset_dir

    extract_root = dataset_dir.parent
    extract_root.mkdir(parents=True, exist_ok=True)
    archive_path = extract_root / "endoscapes.zip"

    print(f"Dataset not found at {dataset_dir} — downloading from {url}")
    print("This is a ~5.9GB download and only happens once.")
    try:
        _download(url, archive_path)

        print(f"Extracting to {extract_root} ...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
    except BaseException:
        # Never leave a half-written archive behind — it would look like a
        # valid cache on the next run.
        archive_path.unlink(missing_ok=True)
        raise

    archive_path.unlink(missing_ok=True)
    print("Removed the archive.")

    if not _is_present(dataset_dir):
        raise RuntimeError(
            f"Dataset was downloaded and extracted into {extract_root}, but "
            f"{dataset_dir} does not contain the expected "
            f"{'/'.join(REQUIRED_SPLITS)} splits. Check DATASET_DIR in your config."
        )

    print(f"Dataset ready at {dataset_dir} ({_format_size(_directory_size(dataset_dir))})")
    return dataset_dir


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the Endoscapes2023 dataset.")
    parser.add_argument("--dataset_dir", default="./dataset/endoscapes",
                        help="Where the dataset should live (default: ./dataset/endoscapes)")
    args = parser.parse_args()
    ensure_dataset(args.dataset_dir)
