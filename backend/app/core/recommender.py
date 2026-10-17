"""Turns a skin tone plus shopper preferences into colour-grouped product recommendations.

The notebooks ran a fuzzy semantic query ("<colour> clothes for men") per palette colour,
which could return off-colour products. Here each palette colour becomes a hard filter on
the catalog's ``baseColour`` (expanded to its colour family), so every recommendation is
guaranteed to be in a flattering colour. Within that set, products are ranked by hybrid
search when the shopper types a query, or sampled with a seed so "shuffle" gives variety.
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
    def __init__(self, catalog: Catalog, search: HybridSearch):
        self.catalog = catalog
        self.search = search

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
                candidates = np.flatnonzero(mask)
                rng = np.random.default_rng([seed, zlib.crc32(color.encode())])
                rows = rng.choice(candidates, size=min(per_color, total), replace=False).tolist()
            groups.append(ColorGroup(color=color, rows=rows, total=total))
        return groups
