"""Hybrid search combining vector similarity and BM25 keyword search."""

from rank_bm25 import BM25Okapi

from config import HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT
from rag.embeddings import embed
from rag.vector_db import search
from rag.query_parser import extract_brand


bm25 = None
documents = []
dataset_global = None


def build_bm25(dataset):
    """Build BM25 index from dataset.
    
    Args:
        dataset: List of documents with text field
    """
    global bm25, documents, dataset_global

    dataset_global = dataset
    documents = [d["text"] for d in dataset]
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)


def hybrid_search(query):
    """Perform hybrid search combining vector and BM25 results.
    
    Args:
        query: User query string
    
    Returns:
        List of ranked results (payload dictionaries)
    """
    brand = extract_brand(query, dataset_global)
    vector = embed(query)
    vector_results = search(vector, brand)

    if bm25 is None:
        raise RuntimeError("BM25 index is not initialized. Call build_bm25(dataset) during startup.")

    if not vector_results:
        return []

    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_score_by_text = {
        text: float(bm25_scores[idx])
        for idx, text in enumerate(documents)
    }

    results = []

    for r in vector_results:
        payload_text = r.payload.get("text", "")
        bm = bm25_score_by_text.get(payload_text, 0.0)
        # Use configurable weights instead of hardcoded values
        score = HYBRID_VECTOR_WEIGHT * r.score + HYBRID_BM25_WEIGHT * bm
        results.append((score, r.payload))

    results.sort(key=lambda item: item[0], reverse=True)
    return [r[1] for r in results]