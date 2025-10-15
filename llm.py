import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = "You are a helpful assistant. Using the provided document excerpts, answer the user's question concisely and cite which documents you used."

def synthesize_answer(query: str, contexts: list, max_tokens: int = 512) -> str:
    """
    Build a prompt with top-k contexts and call the LLM to synthesize an answer.
    """
    # Build context block
    context_text = "\n\n---\n\n".join([f"Document excerpt {i+1}:\n{c}" for i, c in enumerate(contexts)])
    prompt = f"""
You are given several document excerpts and a user question. Using only the supplied excerpts, answer the question as succinctly as possible. If the answer cannot be derived from the excerpts, say you don't know and list which excerpts you used.

EXCERPTS:
{context_text}

QUESTION:
{query}

Answer concisely and include a short "SOURCES" line listing excerpt numbers used.
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
