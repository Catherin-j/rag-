"""Lightweight fixed-size embeddings compatible with Qdrant."""

import re

import numpy as np

from config import EMBEDDING_DIM


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric/hyphen tokens."""
    return re.findall(r"[a-z0-9-]+", text.lower())


def embed(text: str) -> np.ndarray:
    """Generate a fixed-size normalized vector using token hashing."""
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=float)

    vector = np.zeros(EMBEDDING_DIM, dtype=float)
    for token in tokens:
        idx = hash(token) % EMBEDDING_DIM
        vector[idx] += 1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector

    return vector / norm