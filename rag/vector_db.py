"""Persistent vector database operations backed by local Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from config import EMBEDDING_DIM, QDRANT_COLLECTION_NAME, QDRANT_STORAGE_PATH, SEARCH_LIMIT
from rag.embeddings import embed


client = QdrantClient(path=QDRANT_STORAGE_PATH)


def close_db() -> None:
    """Close Qdrant client cleanly."""
    try:
        client.close()
    except Exception:
        # Ignore shutdown-time cleanup errors.
        pass


def init_db() -> None:
    """Create collection if missing and keep data persistent on disk."""
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if QDRANT_COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def reset_collection() -> None:
    """Recreate collection for full reindex from latest dataset file."""
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def insert_dataset(dataset: list[dict]) -> None:
    """Insert full dataset into Qdrant collection."""
    points: list[PointStruct] = []
    for idx, item in enumerate(dataset):
        vector = embed(item["text"]).tolist()
        points.append(PointStruct(id=idx, vector=vector, payload=item))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


def search(vector, brand: str | None = None, limit: int | None = None):
    """Search Qdrant collection by vector similarity with optional brand filter."""
    query_filter = None
    if brand:
        query_filter = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=brand.lower()))]
        )

    result = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=vector.tolist(),
        query_filter=query_filter,
        limit=limit or SEARCH_LIMIT,
    )
    return result.points