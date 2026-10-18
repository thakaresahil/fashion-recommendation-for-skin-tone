"""Download product images for the catalog into ``data/images/{id}.jpg``.

The images come from the "Fashion Product Images (Small)" dataset. They are not committed
to the repository (about 44k files), so fetch them with one of these sources:

    python -m scripts.fetch_images                          # Hugging Face mirror, no account needed
    python -m scripts.fetch_images --source kaggle          # Kaggle (needs ~/.kaggle/kaggle.json)
    python -m scripts.fetch_images --source local --path ~/Downloads/archive.zip
    python -m scripts.fetch_images --sample 1500            # small, colour-balanced demo subset

Only images of products in the catalog are kept. Install the extras first:
``pip install -r requirements-data.txt``.
"""

import argparse
import io
import logging
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.catalog import Catalog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_images")

HF_REPO = "ashraq/fashion-product-images-small"
KAGGLE_DATASET = "paramaggarwal/fashion-product-images-small"
JPEG_MAGIC = b"\xff\xd8"


def sample_ids(df: pd.DataFrame, n: int, min_per_group: int = 3) -> set[int]:
    """A deterministic subset with every (gender, colour) pair represented."""
    groups = df.groupby(["gender", "baseColour"], sort=True)
    picked = []
    for _, group in groups:
        k = min(len(group), max(min_per_group, round(n * len(group) / len(df))))
        picked.append(group.sample(n=k, random_state=0))
    return set(pd.concat(picked)["id"].astype(int))


def from_huggingface(wanted: set[int]) -> Iterator[tuple[int, bytes]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revision = None
    files = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".parquet")]
    if not files:  # datasets stored in another format are auto-converted to parquet on this branch
        revision = "refs/convert/parquet"
        files = [
            f for f in api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision) if f.endswith(".parquet")
        ]
    if not files:
        raise RuntimeError(f"No parquet files found in the {HF_REPO} dataset")

    for name in sorted(files):
        path = hf_hub_download(HF_REPO, name, repo_type="dataset", revision=revision)
        yield from read_parquet(pq.ParquetFile(path), wanted)


def read_parquet(parquet, wanted: set[int]) -> Iterator[tuple[int, bytes]]:
    for batch in parquet.iter_batches(columns=["id", "image"], batch_size=512):
        for pid, image in zip(batch.column("id").to_pylist(), batch.column("image").to_pylist(), strict=True):
            if pid is None or int(pid) not in wanted or image is None:
                continue
            # The Hugging Face Image feature stores {"bytes": ..., "path": ...}.
            data = image.get("bytes") if isinstance(image, dict) else image
            if data:
                yield int(pid), data


def from_local(path: Path, wanted: set[int]) -> Iterator[tuple[int, bytes]]:
    """A Kaggle download: an extracted folder or the zip archive, with images/{id}.jpg inside."""
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                pid = _image_id(info.filename)
                if pid in wanted:
                    yield pid, archive.read(info)
        return
    for file in path.rglob("*.jpg"):
        pid = _image_id(file.name)
        if pid in wanted:
            yield pid, file.read_bytes()


def from_kaggle(wanted: set[int]) -> Iterator[tuple[int, bytes]]:
    import kagglehub

    yield from from_local(Path(kagglehub.dataset_download(KAGGLE_DATASET)), wanted)


def _image_id(filename: str) -> int | None:
    stem, _, ext = Path(filename).name.rpartition(".")
    return int(stem) if ext.lower() == "jpg" and stem.isdigit() else None


def to_jpeg(data: bytes) -> bytes:
    if data[:2] == JPEG_MAGIC:
        return data
    from PIL import Image

    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue()


def fetch(source: Iterator[tuple[int, bytes]], out: Path, wanted: set[int]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for pid, data in source:
        target = out / f"{pid}.jpg"
        if not target.exists():
            target.write_bytes(to_jpeg(data))
        written += 1
        wanted.discard(pid)
        if written % 2000 == 0:
            log.info("%d images saved", written)
    return written


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["huggingface", "kaggle", "local"], default="huggingface")
    parser.add_argument("--path", type=Path, help="Folder or zip for --source local")
    parser.add_argument("--sample", type=int, help="Keep only about N images, balanced across gender and colour")
    parser.add_argument("--out", type=Path, default=settings.images_dir)
    args = parser.parse_args()
    if args.source == "local" and args.path is None:
        parser.error("--source local needs --path")

    catalog = Catalog.from_csv(settings.data_dir / name for name in settings.catalog_files)
    wanted = sample_ids(catalog.df, args.sample) if args.sample else set(catalog.df["id"].astype(int))
    total = len(wanted)
    already = {pid for pid in wanted if (args.out / f"{pid}.jpg").exists()}
    wanted -= already
    log.info("%d images wanted, %d already present, fetching %d from %s", total, len(already), len(wanted), args.source)
    if not wanted:
        return

    if args.source == "huggingface":
        source = from_huggingface(wanted.copy())
    elif args.source == "kaggle":
        source = from_kaggle(wanted.copy())
    else:
        source = from_local(args.path.expanduser(), wanted.copy())

    try:
        written = fetch(source, args.out, wanted)
    except Exception as exc:  # network, auth or archive problems: explain instead of a traceback
        hint = "Download it from Kaggle in a browser and use --source local --path <zip>."
        sys.exit(f"Image download from {args.source} failed: {exc}\n{hint}")
    log.info("Saved %d images to %s; %d products have no image in the source", written, args.out, len(wanted))


if __name__ == "__main__":
    main()
