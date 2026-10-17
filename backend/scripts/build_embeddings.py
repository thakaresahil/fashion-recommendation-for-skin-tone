"""Build the dense CLIP index used for hybrid search.

Encodes product images (from the Kaggle "Fashion Product Images (Small)" dataset, placed in
``data/images``) with CLIP. Products without an image fall back to a CLIP encoding of
their metadata sentence, so the index always covers the full catalog. CLIP maps text and
images into the same space, so both kinds of vector can be compared with a text query.

    pip install -r requirements-ml.txt
    python -m scripts.build_embeddings            # images where available, text otherwise
    python -m scripts.build_embeddings --source text

Writes ``embeddings.npy`` (float16, L2-normalised), ``ids.npy`` and ``source.txt`` to
``data/embeddings``; the API picks them up on its next start.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.catalog import Catalog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_embeddings")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "image", "text"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Only encode the first N products (for a quick try)")
    parser.add_argument("--out", type=Path, default=settings.embeddings_dir)
    args = parser.parse_args()

    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    catalog = Catalog.from_csv(settings.data_dir / name for name in settings.catalog_files)
    df = catalog.df if args.limit is None else catalog.df.head(args.limit)
    model = SentenceTransformer(settings.clip_model)

    def image_path(pid: int) -> Path:
        return settings.images_dir / f"{pid}.jpg"

    use_images = args.source != "text"
    with_image = df["id"].map(lambda pid: use_images and image_path(pid).exists()).to_numpy()
    if args.source == "image" and not with_image.any():
        sys.exit(f"No images found in {settings.images_dir}")
    log.info("Encoding %d products: %d images, %d text", len(df), with_image.sum(), (~with_image).sum())

    vectors = np.zeros((len(df), model.get_sentence_embedding_dimension()), dtype=np.float32)
    image_rows = np.flatnonzero(with_image)
    for start in tqdm(range(0, len(image_rows), args.batch_size), desc="images"):
        rows = image_rows[start : start + args.batch_size]
        images = [Image.open(image_path(df["id"].iat[r])).convert("RGB") for r in rows]
        vectors[rows] = model.encode(images, normalize_embeddings=True, batch_size=args.batch_size)

    text_rows = np.flatnonzero(~with_image)
    if text_rows.size:
        texts = df["text"].iloc[text_rows].tolist()
        vectors[text_rows] = model.encode(
            texts, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=True
        )

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", vectors.astype(np.float16))
    np.save(args.out / "ids.npy", df["id"].to_numpy())
    source = "image" if with_image.all() else "text" if not with_image.any() else "image+text"
    (args.out / "source.txt").write_text(source)
    log.info("Wrote %s embeddings (%s) to %s", vectors.shape, source, args.out)


if __name__ == "__main__":
    main()
