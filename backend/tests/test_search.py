import numpy as np

from app.core import colors
from app.core.catalog import Filters
from app.core.search import BM25Index, DenseIndex, HybridSearch, tokenize

TEXTS = [
    "Men Apparel Topwear Shirts Navy Blue Formal Peter England Navy Formal Shirt",
    "Men Apparel Topwear Tshirts Red Casual Nike Red T-shirt",
    "Women Apparel Dress Dresses Red Party Red Party Dress",
    "Men Footwear Shoes Sports Shoes Black Sports Running Shoes",
]


class FakeEncoder:
    """Maps a query to the embedding of the document it names, for deterministic tests."""

    def __init__(self, vectors: np.ndarray, lookup: dict[str, int]):
        self.vectors, self.lookup = vectors, lookup

    def encode(self, texts):
        return np.stack([self.vectors[self.lookup[t]] for t in texts])


def test_tokenize_normalises_plurals_and_tshirts():
    assert tokenize("Red T-shirts and Shoes") == ["red", "tshirt", "and", "shoe"]
    assert tokenize("Dress Dresses Watches Jeans") == ["dress", "dress", "watch", "jean"]


def test_bm25_ranks_most_specific_document_first():
    index = BM25Index(TEXTS)
    scores = index.scores("red dress")
    assert int(np.argmax(scores)) == 2
    assert scores[3] == 0
    assert not index.scores("zzz unknown").any()


def test_search_respects_mask_and_pagination():
    search = HybridSearch(BM25Index(TEXTS))
    mask = np.array([True, True, False, True])
    hits, total = search.search("red", mask=mask)
    assert [h.row for h in hits] == [1]
    assert total == 1
    hits, _ = search.search("men", limit=1, offset=1)
    assert len(hits) == 1


def test_dense_retriever_contributes_to_fusion():
    vectors = np.eye(4, dtype=np.float32)
    dense = DenseIndex(vectors, np.ones(4, dtype=bool), FakeEncoder(vectors, {"sneakers": 3}), source="test")
    search = HybridSearch(BM25Index(TEXTS), dense, dense_weight=0.8)
    hits, _ = search.search("sneakers")
    # "sneakers" has no BM25 match; the dense retriever alone finds the shoes.
    assert hits[0].row == 3
    assert hits[0].sparse_rank is None and hits[0].dense_rank == 1
    assert search.mode.startswith("hybrid")


def test_similar_excludes_the_item_itself():
    search = HybridSearch(BM25Index(TEXTS))
    hits = search.similar(1, TEXTS[1], np.ones(4, dtype=bool))
    assert 1 not in [h.row for h in hits]
    assert hits[0].row == 2  # the other red item


def test_catalog_filters(services):
    catalog = services.catalog
    m = catalog.mask(Filters(gender="Women", colors=("Red",), master_category="Apparel"))
    sub = catalog.df[m]
    assert len(sub) > 0
    assert set(sub["gender"]) <= {"Women", "Unisex"}
    assert set(sub["baseColour"]) == {"Red"}
    assert set(sub["masterCategory"]) == {"Apparel"}


def test_colour_extraction():
    assert colors.extract("Navy Blue formal shirt") == ["Navy Blue"]
    assert sorted(colors.extract("red and white sneakers")) == ["Red", "White"]
    assert colors.extract("skin care multi pack") == []
    assert colors.extract("bluetooth speaker") == []


def test_catalog_cleaning(services):
    df = services.catalog.df
    assert df["id"].is_unique
    assert "NA" not in set(df["baseColour"])
    assert not df["masterCategory"].isin(["Free Items", "Personal Care"]).any()


def test_dense_index_loads_and_aligns_to_catalog_rows(tmp_path):
    np.save(tmp_path / "embeddings.npy", np.array([[0, 2], [3, 0]], dtype=np.float16))
    np.save(tmp_path / "ids.npy", np.array([30, 10]))
    (tmp_path / "source.txt").write_text("image")
    dense = DenseIndex.load(tmp_path, np.array([10, 20, 30]), FakeEncoder(np.eye(2), {}))
    assert dense.present.tolist() == [True, False, True]
    np.testing.assert_allclose(dense.vectors, [[1, 0], [0, 0], [0, 1]])
    assert dense.source == "image"
