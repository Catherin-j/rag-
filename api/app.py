from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    DATASET_PATH,
    MIN_QUERY_LENGTH,
    MAX_QUERY_LENGTH,
    LLM_TOP_RESULTS
)
from rag.dataset_loader import load_dataset
from rag.vector_db import close_db, init_db, insert_dataset, reset_collection
from rag.hybrid_search import build_bm25, hybrid_search
from rag.llm import generate_answer
from rag.query_parser import is_battery_related, extract_brands_from_dataset

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

dataset = None
available_brands = []


def _build_indexes() -> None:
    """Load dataset and rebuild all indexes."""
    global dataset, available_brands

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} battery documents from {DATASET_PATH}")

    if not dataset:
        raise RuntimeError("Dataset is empty or invalid. Please check DATASET_PATH and file format.")

    available_brands = extract_brands_from_dataset(dataset)
    print(f"Detected brands: {', '.join(available_brands)}")

    reset_collection()
    insert_dataset(dataset)
    build_bm25(dataset)


@app.on_event("startup")
def startup():
    """Initialize database, load dataset, and build search indices on startup."""
    
    print("Starting BatteryBrain RAG API...")
    init_db()
    print("Initialized vector collection")
    _build_indexes()
    print("Built vector and BM25 indexes")
    print("BatteryBrain is ready")


@app.on_event("shutdown")
def shutdown() -> None:
    """Close vector DB resources cleanly."""
    close_db()


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "BatteryBrain RAG API is running",
        "status": "healthy",
        "documents_loaded": len(dataset) if dataset else 0,
        "available_brands": available_brands,
        "scope": "Battery technology queries only"
    }


class QueryRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What battery chemistry does Lenovo use?"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources_found: int
    context_snippets: list[str]


class ReloadResponse(BaseModel):
    status: str
    documents_loaded: int
    available_brands: list[str]


@app.post("/battery-query", response_model=QueryResponse)
def battery_query(request: QueryRequest):
    """Answer battery-related questions using RAG. Rejects non-battery queries."""
    
    question = request.question.strip()
    
    # Validate input length
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(question) < MIN_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too short. Minimum {MIN_QUERY_LENGTH} characters required."
        )
    
    if len(question) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
        )
    
    # STRICT GUARDRAIL: Check if query is battery-related (pass dataset for brand detection)
    if not is_battery_related(question, dataset):
        brand_examples = ", ".join(available_brands[:4]) if available_brands else "battery-related topics"
        raise HTTPException(
            status_code=400,
            detail=f"This system only answers battery-related questions. Please ask about battery chemistry, specifications, voltage, capacity, or brands like {brand_examples}."
        )
    
    # Perform hybrid search
    docs = hybrid_search(question)
    
    if not docs:
        return QueryResponse(
            answer="I couldn't find relevant information in my battery knowledge base. Try rephrasing your question or ask about specific battery brands or chemistry types.",
            question=question,
            sources_found=0,
            context_snippets=[]
        )
    
    # Extract top results for context (use config value)
    top_docs = docs[:LLM_TOP_RESULTS]
    context = "\n\n".join([d["text"] for d in top_docs])
    
    # Generate answer with LLM
    answer = generate_answer(question, context)
    
    return QueryResponse(
        answer=answer,
        question=question,
        sources_found=len(docs),
        context_snippets=[d["text"] for d in top_docs]
    )


@app.post("/admin/reload-index", response_model=ReloadResponse)
def reload_index():
    """Reload dataset file and rebuild indexes for newly added records."""
    _build_indexes()
    return ReloadResponse(
        status="reloaded",
        documents_loaded=len(dataset),
        available_brands=available_brands,
    )