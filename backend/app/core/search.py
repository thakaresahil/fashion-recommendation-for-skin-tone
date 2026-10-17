"""Hybrid product retrieval: sparse BM25 + dense CLIP embeddings, merged with weighted RRF.

The research notebooks used LangChain's ``EnsembleRetriever`` over a BM25 retriever and a
Qdrant collection. This module implements the same idea without the framework overhead:

* **Sparse** – BM25 over each product's metadata sentence, pre-computed as a SciPy sparse
  matrix so a query is a single column-sum.
* **Dense** – CLIP embeddings (of product images, or of the metadata text when images are
  not available) built offline by ``scripts/build_embeddings.py``. CLIP shares one space for
  text and images, so a text query can be compared with image vectors directly. At ~22k
  items an exact matrix product is faster than an ANN index and needs no extra service.
* **Fusion** – weighted Reciprocal Rank Fusion, which is robust to the two retrievers
  producing scores on different scales.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 200

_TOKEN = re.compile(r"[a-z0-9]+")


def _singular(token: str) -> str:
    # A light plural stemmer is enough for catalog vocabulary: shirts, dresses, watches.
    if len(token) <= 3 or not token.endswith("s") or token.endswith("ss"):
        return token
    if token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    return token[:-1]


def tokenize(text: str) -> list[str]:
    return [_singular(t) for t in _TOKEN.findall(text.lower().replace("t-shirt", "tshirt"))]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Rewrite each non-zero tf entry in place into its BM25 term weight.
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm[rows])
        self.weights = tf.tocsc()
        self.vocab = self.vectorizer.vocabulary_

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class ClipTextEncoder:
    """Lazily loads a sentence-transformers CLIP model (optional ``requirements-ml.txt``)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class DenseIndex:
    """L2-normalised embeddings aligned to catalog rows; rows without a vector are masked out."""

    def __init__(self, vectors: np.ndarray, present: np.ndarray, encoder: TextEncoder, source: str):
        self.vectors = vectors
        self.present = present
        self.encoder = encoder
        self.source = source

    @classmethod
    def load(cls, directory: Path, product_ids: np.ndarray, encoder: TextEncoder) -> "DenseIndex":
        emb = np.load(directory / "embeddings.npy").astype(np.float32)
        ids = np.load(directory / "ids.npy")
        source = (directory / "source.txt").read_text().strip() if (directory / "source.txt").exists() else "unknown"

        position = {int(pid): i for i, pid in enumerate(ids)}
        vectors = np.zeros((len(product_ids), emb.shape[1]), dtype=np.float32)
        present = np.zeros(len(product_ids), dtype=bool)
        for row, pid in enumerate(product_ids):
            i = position.get(int(pid))
            if i is not None:
                vectors[row] = emb[i]
                present[row] = True
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return cls(vectors, present, encoder, source)

    def scores_for_query(self, query: str) -> np.ndarray:
        return self.vectors @ self.encoder.encode([query])[0]

    def scores_for_row(self, row: int) -> np.ndarray:
        return self.vectors @ self.vectors[row]


@dataclass
class Hit:
    row: int
    score: float
    sparse_rank: int | None
    dense_rank: int | None


def _top_ranks(scores: np.ndarray, mask: np.ndarray, n: int) -> dict[int, int]:
    """Rank (1-based) of the top ``n`` masked rows with a positive score."""
    candidates = np.flatnonzero(mask & (scores > 0))
    if candidates.size > n:
        candidates = candidates[np.argpartition(-scores[candidates], n)[:n]]
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
    return {int(r): i + 1 for i, r in enumerate(ordered)}


class HybridSearch:
    def __init__(self, bm25: BM25Index, dense: DenseIndex | None = None, dense_weight: float = 0.6):
        self.bm25 = bm25
        self.dense = dense
        self.dense_weight = dense_weight

    @property
    def mode(self) -> str:
        return f"hybrid (bm25 + clip-{self.dense.source})" if self.dense else "bm25"

    def search(
        self,
        query: str,
        mask: np.ndarray | None = None,
        limit: int = 20,
        offset: int = 0,
        dense_weight: float | None = None,
    ) -> tuple[list[Hit], int]:
        """Return a page of hits and the total number of fused candidates."""
        n = self.bm25.weights.shape[0]
        mask = np.ones(n, dtype=bool) if mask is None else mask
        sparse_scores = self.bm25.scores(query)
        dense_scores = self.dense.scores_for_query(query) if self.dense else None
        return self._fuse(sparse_scores, dense_scores, mask, limit, offset, dense_weight)

    def similar(self, row: int, text: str, mask: np.ndarray, limit: int = 12) -> list[Hit]:
        mask = mask.copy()
        mask[row] = False
        sparse_scores = self.bm25.scores(text)
        dense_scores = None
        if self.dense is not None and self.dense.present[row]:
            dense_scores = self.dense.scores_for_row(row)
        hits, _ = self._fuse(sparse_scores, dense_scores, mask, limit, 0, None)
        return hits

    def _fuse(
        self,
        sparse_scores: np.ndarray,
        dense_scores: np.ndarray | None,
        mask: np.ndarray,
        limit: int,
        offset: int,
        dense_weight: float | None,
    ) -> tuple[list[Hit], int]:
        w_dense = self.dense_weight if dense_weight is None else dense_weight
        sparse_ranks = _top_ranks(sparse_scores, mask, CANDIDATES_PER_RETRIEVER)
        dense_ranks: dict[int, int] = {}
        if dense_scores is not None and w_dense > 0:
            dense_mask = mask & self.dense.present
            # Cosine similarity can be <= 0; shift so every present candidate is rankable.
            dense_ranks = _top_ranks(dense_scores + 2.0, dense_mask, CANDIDATES_PER_RETRIEVER)
        else:
            w_dense = 0.0

        fused: dict[int, float] = {}
        for r, rank in sparse_ranks.items():
            fused[r] = fused.get(r, 0.0) + (1 - w_dense) / (RRF_K + rank)
        for r, rank in dense_ranks.items():
            fused[r] = fused.get(r, 0.0) + w_dense / (RRF_K + rank)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        page = ordered[offset : offset + limit]
        hits = [Hit(r, s, sparse_ranks.get(r), dense_ranks.get(r)) for r, s in page]
        return hits, len(ordered)


def build_dense(directory: Path, product_ids: np.ndarray, model_name: str) -> DenseIndex | None:
    """Load the dense index if embeddings were built and the ML extras are installed."""
    if not (directory / "embeddings.npy").exists():
        logger.info("No embeddings in %s; running BM25-only search", directory)
        return None
    try:
        encoder = ClipTextEncoder(model_name)
    except Exception as exc:  # ImportError, or a model download failure
        logger.warning("Dense search disabled, could not load CLIP model: %s", exc)
        return None
    return DenseIndex.load(directory, product_ids, encoder)
