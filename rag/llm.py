"""LLM integration for generating answers using Groq API."""

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS


client = None


def _get_client() -> Groq:
    """Create Groq client lazily so non-LLM modules can import safely."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file before calling the LLM.")
    return Groq(api_key=GROQ_API_KEY)


def generate_answer(query, context):
    """Generate answer using Groq LLM with strict battery-only guardrails.
    
    Args:
        query: User question
        context: Retrieved context from knowledge base
    Returns:
        Generated answer string
    """
    global client

    if client is None:
        client = _get_client()
    
    if not context or context.strip() == "":
        return "I couldn't find relevant information in the battery knowledge base."
    
    prompt = f"""
You are a battery technology specialist assistant. Your ONLY purpose is to answer questions about battery chemistry, specifications, and technology.

IMPORTANT RULES:
1. Answer ONLY using the context provided below
2. If the context doesn't contain the answer, say: "I don't have that information in my battery knowledge base."
3. Give a slightly detailed answer (3-6 sentences)
4. Include chemistry meaning/full form and key properties when present in context
5. NEVER answer questions unrelated to batteries
6. NEVER make up information not in the context

Context from battery knowledge base:
{context}

Question:
{query}

Answer (battery-related only):
"""

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )

    return response.choices[0].message.content
