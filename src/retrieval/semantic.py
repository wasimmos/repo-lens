"""Semantic search engine using Latent Semantic Indexing (LSI) with NumPy.

Projects codebase passages into dense latent semantic vectors via TF-IDF + SVD,
allowing semantic similarity queries without external heavy vector databases.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.retrieval.search import document_text, tokenize


class SemanticSearchEngine:
    """Dense semantic retrieval engine for codebase passages."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        k_dimensions: int = 64,
    ):
        self.records = records
        self.num_docs = len(records)
        self.k_dimensions = min(k_dimensions, max(self.num_docs, 1))

        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.U_k: np.ndarray = np.array([])
        self.S_k: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])

        if self.num_docs > 0:
            self._build_index()

    def _build_index(self) -> None:
        """Tokenize documents, compute TF-IDF, and compute dense SVD embeddings."""
        doc_terms_list = [
            tokenize(document_text(rec), code_aware=True)
            for rec in self.records
        ]

        # Build vocabulary
        df_counter = Counter()
        for terms in doc_terms_list:
            df_counter.update(set(terms))

        # Filter out rare terms (df == 0) and build vocabulary map
        sorted_terms = [t for t, df in df_counter.items() if df >= 1]
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}
        vocab_size = len(self.vocab)

        if vocab_size == 0:
            return

        # Compute IDF
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = df_counter[term]
            self.idf[idx] = math.log((self.num_docs + 1) / (df + 1)) + 1.0

        # Build Term-Document TF-IDF matrix (shape: vocab_size x num_docs)
        matrix = np.zeros((vocab_size, self.num_docs), dtype=np.float32)

        for doc_idx, terms in enumerate(doc_terms_list):
            if not terms:
                continue
            counts = Counter(terms)
            total = len(terms)
            for term, count in counts.items():
                if term in self.vocab:
                    term_idx = self.vocab[term]
                    tf = count / total
                    matrix[term_idx, doc_idx] = tf * self.idf[term_idx]

        # Truncated SVD via NumPy
        k = min(self.k_dimensions, vocab_size, self.num_docs)
        if k < 1:
            k = 1

        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            self.U_k = U[:, :k]
            self.S_k = S[:k]
            self.doc_vectors = Vt[:k, :].T  # shape: (num_docs, k)

            # L2-normalize document vectors for fast cosine similarity dot products
            norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.doc_vectors = self.doc_vectors / norms
        except np.linalg.LinAlgError:
            # Fallback if SVD fails to converge
            self.doc_vectors = np.zeros((self.num_docs, k), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.05,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Query semantic index using cosine similarity in latent space."""
        if self.doc_vectors.size == 0 or not self.vocab:
            return []

        query_terms = tokenize(query, code_aware=True)
        if not query_terms:
            return []

        # Vectorize query
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(query_terms)
        total = len(query_terms)

        for term, count in counts.items():
            if term in self.vocab:
                t_idx = self.vocab[term]
                tf = count / total
                q_vec[t_idx] = tf * self.idf[t_idx]

        if np.all(q_vec == 0):
            return []

        # Project query into latent semantic space
        # q_proj = q^T * U_k * diag(1/S_k)
        denom = self.S_k.copy()
        denom[denom == 0] = 1.0
        q_proj = (q_vec @ self.U_k) / denom

        q_norm = np.linalg.norm(q_proj)
        if q_norm == 0:
            return []
        q_proj /= q_norm

        # Cosine similarity
        similarities = self.doc_vectors @ q_proj

        top_indices = np.argsort(-similarities)[:top_k]
        results: list[tuple[float, dict[str, Any]]] = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                results.append((score, self.records[idx]))

        return results

    def save_cache(self, file_path: Path) -> None:
        """Cache pre-computed vector matrices to disk."""
        np.savez_compressed(
            file_path,
            doc_vectors=self.doc_vectors,
            U_k=self.U_k,
            S_k=self.S_k,
            idf=self.idf,
            vocab_keys=list(self.vocab.keys()),
            vocab_vals=list(self.vocab.values()),
        )

    @classmethod
    def load_cache(
        cls, file_path: Path, records: list[dict[str, Any]]
    ) -> Optional[SemanticSearchEngine]:
        """Load pre-computed vector matrices from disk."""
        if not file_path.exists():
            return None
        try:
            data = np.load(file_path)
            engine = cls.__new__(cls)
            engine.records = records
            engine.num_docs = len(records)
            engine.doc_vectors = data["doc_vectors"]
            engine.U_k = data["U_k"]
            engine.S_k = data["S_k"]
            engine.idf = data["idf"]
            keys = data["vocab_keys"].tolist()
            vals = data["vocab_vals"].tolist()
            engine.vocab = dict(zip(keys, vals))
            engine.k_dimensions = engine.doc_vectors.shape[1] if engine.doc_vectors.ndim > 1 else 1
            return engine
        except Exception:
            return None
