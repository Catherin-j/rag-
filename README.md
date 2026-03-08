# BatteryBrain RAG

Battery-only RAG API optimized for low-resource deployment (Render/Railway free tier).

## What This Version Optimizes

- No heavy embedding model downloads.
- Persistent local Qdrant vector database.
- Lightweight fixed-size hashing embeddings + BM25 hybrid retrieval.
- Strict battery-domain guardrails based on your dataset.
- LLM-based answer generation grounded in retrieved dataset context.

## Project Structure

```text
batterybrain-rag/
|-- api/
|   `-- app.py
|-- rag/
|   |-- dataset_loader.py
|   |-- embeddings.py
|   |-- vector_db.py
|   |-- hybrid_search.py
|   |-- query_parser.py
|   `-- llm.py
|-- data/
|   `-- battery_dataset.txt
|-- config.py
|-- requirements.txt
`-- README.md
```

## Dataset Format

Each line in `data/battery_dataset.txt` must be:

```text
Brand - Chemistry
```

Example:

```text
Lenovo - LCO
LG - LCO
Lapgrade - NMC 523
Trilot - NMC 622
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Environment

Create `.env`:

```env
GROQ_API_KEY=your_key_here
```

`GROQ_API_KEY` is required for answer generation.

## Run

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## API

- `GET /` health and metadata
- `POST /battery-query`

Request body:

```json
{
  "question": "What battery chemistry does Lenovo use?"
}
```

## Deploy Notes (Render/Railway)

Start command:

```bash
uvicorn api.app:app --host 0.0.0.0 --port $PORT
```

Free-tier tips:

- Keep dataset small-to-medium for memory safety.
- Avoid very large `SEARCH_LIMIT`.
- Use Groq key only if needed; fallback mode removes outbound LLM dependency.

uvicorn api.app:app --reload