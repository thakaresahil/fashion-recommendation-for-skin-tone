"""Builds the long-lived objects (catalog, indexes, models) once at startup."""

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.core.catalog import Catalog
from app.core.palettes import PaletteBook
from app.core.recommender import Recommender
from app.core.search import BM25Index, HybridSearch, build_dense

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    catalog: Catalog
    palettes: PaletteBook
    search: HybridSearch
    recommender: Recommender
    image_ids: set[int]

    def image_url(self, product_id: int) -> str | None:
        return f"/api/v1/products/{product_id}/image" if product_id in self.image_ids else None


def build_services(settings: Settings) -> Services:
    started = time.perf_counter()
    catalog = Catalog.from_csv(settings.data_dir / name for name in settings.catalog_files)
    palettes = PaletteBook.load(settings.palettes_file)

    dense = None
    if settings.enable_dense:
        dense = build_dense(settings.embeddings_dir, catalog.df["id"].to_numpy(), settings.clip_model)
    search = HybridSearch(BM25Index(catalog.search_texts), dense, settings.dense_weight)

    image_ids: set[int] = set()
    if settings.images_dir.is_dir():
        image_ids = {int(p.stem) for p in settings.images_dir.glob("*.jpg") if p.stem.isdigit()}

    logger.info(
        "Loaded %d products, %d images, search mode %s in %.1fs",
        len(catalog),
        len(image_ids),
        search.mode,
        time.perf_counter() - started,
    )
    return Services(settings, catalog, palettes, search, Recommender(catalog, search), image_ids)
