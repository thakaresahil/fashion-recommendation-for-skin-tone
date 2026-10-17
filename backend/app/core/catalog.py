"""Product catalog: loading, cleaning and boolean filtering of the product metadata."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "year",
    "usage",
    "productDisplayName",
]

FILTER_COLUMNS = ["gender", "masterCategory", "subCategory", "baseColour", "usage"]

# Rows in these categories are not clothing and would only add noise to recommendations.
EXCLUDED_MASTER_CATEGORIES = {"Free Items", "Personal Care"}

# Products for adults of a gender also include unisex items.
GENDER_EXPANSION = {
    "Men": ["Men", "Unisex"],
    "Women": ["Women", "Unisex"],
    "Boys": ["Boys"],
    "Girls": ["Girls"],
    "Unisex": ["Unisex"],
}


@dataclass(frozen=True)
class Filters:
    gender: str | None = None
    colors: tuple[str, ...] = ()
    master_category: str | None = None
    sub_categories: tuple[str, ...] = ()
    usage: str | None = None


class Catalog:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._row_by_id = {int(pid): i for i, pid in enumerate(self.df["id"])}
        # Lower-cased column arrays make filtering a handful of vectorised comparisons.
        self._cols = {c: self.df[c].str.lower().to_numpy(dtype=object) for c in FILTER_COLUMNS}

    @classmethod
    def from_csv(cls, paths: Iterable[Path]) -> "Catalog":
        frames = [pd.read_csv(p, on_bad_lines="skip", usecols=lambda c: c in COLUMNS) for p in paths]
        return cls(clean(pd.concat(frames, ignore_index=True)))

    def __len__(self) -> int:
        return len(self.df)

    @property
    def texts(self) -> list[str]:
        return self.df["text"].tolist()

    @property
    def search_texts(self) -> list[str]:
        # Field boosting for BM25: colour and article type are what shoppers search by most,
        # so they are repeated to outweigh rarer but less decisive words such as the occasion.
        df = self.df
        return (df["text"] + " " + df["baseColour"] + " " + df["articleType"]).tolist()

    def row_of(self, product_id: int) -> int | None:
        return self._row_by_id.get(product_id)

    def record(self, row: int) -> dict:
        return self.df.iloc[row].to_dict()

    def mask(self, f: Filters) -> np.ndarray:
        m = np.ones(len(self.df), dtype=bool)
        if f.gender:
            genders = [g.lower() for g in GENDER_EXPANSION.get(f.gender.title(), [f.gender])]
            m &= np.isin(self._cols["gender"], genders)
        if f.colors:
            m &= np.isin(self._cols["baseColour"], [c.lower() for c in f.colors])
        if f.master_category:
            m &= self._cols["masterCategory"] == f.master_category.lower()
        if f.sub_categories:
            m &= np.isin(self._cols["subCategory"], [s.lower() for s in f.sub_categories])
        if f.usage:
            m &= self._cols["usage"] == f.usage.lower()
        return m

    def facets(self) -> dict[str, list[dict]]:
        def counts(col: str) -> list[dict]:
            vc = self.df[col].value_counts()
            return [{"value": k, "count": int(v)} for k, v in vc.items()]

        return {c: counts(c) for c in FILTER_COLUMNS}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["id", "baseColour", "productDisplayName"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id")

    text_cols = [c for c in COLUMNS if c not in ("id", "year")]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[(df["baseColour"] != "") & (df["baseColour"] != "NA")]
    df = df[~df["masterCategory"].isin(EXCLUDED_MASTER_CATEGORIES)]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # Same "metadata as a sentence" representation the research notebooks indexed.
    df["text"] = df[text_cols].agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True)
    return df.reset_index(drop=True)
