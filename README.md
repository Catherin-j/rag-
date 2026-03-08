

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