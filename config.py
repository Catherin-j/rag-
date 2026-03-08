"""
Configuration settings for BatteryBrain RAG system.
All hardcoded values should be defined here or loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# API Configuration
# =============================================================================
API_TITLE = os.getenv("API_TITLE", "BatteryBrain RAG API")
API_DESCRIPTION = os.getenv(
    "API_DESCRIPTION",
    "AI-powered battery knowledge assistant - answers ONLY battery-related queries"
)
API_VERSION = os.getenv("API_VERSION", "1.0.0")


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_PATH = os.getenv("DATASET_PATH", "data/battery_dataset.txt")


# =============================================================================
# Retrieval Configuration
# =============================================================================
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "qdrant_storage")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "battery_knowledge")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


# =============================================================================
# LLM Configuration
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))
LLM_TOP_RESULTS = int(os.getenv("LLM_TOP_RESULTS", "3"))  # Number of search results to use as context


# =============================================================================
# Hybrid Search Configuration
# =============================================================================
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))


# =============================================================================
# Validation Settings
# =============================================================================
MIN_QUERY_LENGTH = int(os.getenv("MIN_QUERY_LENGTH", "3"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
