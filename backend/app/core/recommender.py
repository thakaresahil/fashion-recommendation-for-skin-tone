"""Turns a skin tone plus shopper preferences into colour-grouped product recommendations.

The notebooks ran a fuzzy semantic query ("<colour> clothes for men") per palette colour,
which could return off-colour products. Here each palette colour becomes a hard filter on
the catalog's ``baseColour`` (expanded to its colour family), so every recommendation is
guaranteed to be in a flattering colour. Within that set, products are ranked by hybrid
search when the shopper types a query, or sampled with a seed so "shuffle" gives variety.
Sampling prefers products that have a photo, so a partial image download still looks complete.
"""

import zlib
from dataclasses import dataclass, replace

import numpy as np

from app.core import colors
from app.core.catalog import Catalog, Filters
from app.core.palettes import Tone
from app.core.search import HybridSearch


@dataclass
class ColorGroup:
    color: str
    rows: list[int]
    total: int


class Recommender:
    def __init__(self, catalog: Catalog, search: HybridSearch, has_image: np.ndarray | None = None):
        self.catalog = catalog
        self.search = search
        self.has_image = np.zeros(len(catalog), dtype=bool) if has_image is None else has_image

    def recommend(
        self,
        tone: Tone,
        filters: Filters,
        query: str | None = None,
        per_color: int = 8,
        seed: int = 0,
    ) -> list[ColorGroup]:
        groups = []
        for color in tone.colors:
            mask = self.catalog.mask(replace(filters, colors=tuple(colors.family(color))))
            total = int(mask.sum())
            rows: list[int] = []
            if query and query.strip():
                hits, _ = self.search.search(query, mask=mask, limit=per_color)
                rows = [h.row for h in hits]
            if not rows and total:
                rows = self._sample(np.flatnonzero(mask), per_color, [seed, zlib.crc32(color.encode())])
            groups.append(ColorGroup(color=color, rows=rows, total=total))
        return groups

    def _sample(self, candidates: np.ndarray, k: int, seed: list[int]) -> list[int]:
        rng = np.random.default_rng(seed)
        picked: list[int] = []
        for pool in (candidates[self.has_image[candidates]], candidates[~self.has_image[candidates]]):
            take = min(k - len(picked), len(pool))
            if take > 0:
                picked += rng.choice(pool, size=take, replace=False).tolist()
        return picked
