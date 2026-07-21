"""
Shared config + retrieval helpers for the NSE RAG agent eval harness.

Mirrors the retrieve()/format_context() logic in app.py and the notebook,
so the harness queries Chroma exactly the same way the live agent does --
we're comparing generation models, not accidentally comparing retrieval
setups too.
"""
from pathlib import Path

import chromadb
import ollama

# ---------------------------------------------------------------------
# Keep these in sync with app.py / the notebook
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "nse_filings_news"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 6

SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's
question about NSE-listed stocks using ONLY the provided context (filings and
news excerpts). If the context doesn't contain enough information, say so
clearly rather than guessing. Always mention which ticker(s) and dates your
answer draws from. Do not give investment advice or price predictions --
stick to summarizing and explaining what the context says."""

# ---------------------------------------------------------------------
# The 3 models under comparison. Pull each first:
#   ollama pull llama3.1:8b
#   ollama pull llama3.2:3b
#   ollama pull phi3
MODELS = ["llama3.1:8b", "llama3.2:3b", "phi3"]

# Judge model used for LLM-as-judge scoring (faithfulness / correctness /
# instruction adherence). Deliberately NOT one of the 3 models being
# compared, so it isn't grading its own homework. Pull it separately:
#   ollama pull qwen2.5:7b
JUDGE_MODEL = "qwen2.5:7b"


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return get_client().get_collection(COLLECTION_NAME)


def retrieve(collection, query: str, ticker_filter: str | None = None, k: int = TOP_K):
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    where = {"ticker": ticker_filter} if ticker_filter else None
    return collection.query(query_embeddings=[query_embedding], n_results=k, where=where)


def format_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(f"[{meta['ticker']} | {meta['kind']} | {meta['date']}] {meta['title']}\n{doc}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"
