"""
Gradio UI for the NSE RAG agent.

Run this from the same project folder as your notebook (needs access to
chroma_db/ built by section 4). Launch with:

    python app.py

Then open the local URL it prints (usually http://127.0.0.1:7860).
"""
from pathlib import Path

import chromadb
import gradio as gr
import ollama

# ---------------------------------------------------------------------
# Same config as the notebook -- keep these in sync if you change them there
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "nse_filings_news"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
TOP_K = 6

SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's
question about NSE-listed stocks using ONLY the provided context (filings and
news excerpts). If the context doesn't contain enough information, say so
clearly rather than guessing. Always mention which ticker(s) and dates your
answer draws from. Do not give investment advice or price predictions --
stick to summarizing and explaining what the context says."""

# ---------------------------------------------------------------------
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION_NAME)

# Pull the list of tickers actually indexed, for the filter dropdown
_all_metas = collection.get(include=["metadatas"])["metadatas"]
AVAILABLE_TICKERS = sorted({m["ticker"] for m in _all_metas}) if _all_metas else []


def retrieve(query: str, ticker_filter: str | None, k: int = TOP_K):
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    where = {"ticker": ticker_filter} if ticker_filter and ticker_filter != "All tickers" else None
    return collection.query(query_embeddings=[query_embedding], n_results=k, where=where)


def format_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(f"[{meta['ticker']} | {meta['kind']} | {meta['date']}] {meta['title']}\n{doc}")
    return "\n\n---\n\n".join(blocks)


def format_sources_markdown(results) -> str:
    """Human-readable source list shown in the sidebar under the chat."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "_No sources retrieved for this query._"

    lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        snippet = doc[:220].strip() + ("..." if len(doc) > 220 else "")
        link = f"[source]({meta['source_url']})" if meta.get("source_url") else ""
        lines.append(
            f"**{i}. {meta['ticker']} — {meta['kind']} — {meta['date']}**  \n"
            f"{meta['title']}  \n"
            f"> {snippet}  \n"
            f"{link}"
        )
    return "\n\n".join(lines)


def respond(message: str, history: list, ticker_filter: str):
    results = retrieve(message, ticker_filter)

    if not results["documents"][0]:
        yield "No relevant filings or news found in the index for that query.", "_No sources retrieved._"
        return

    context = format_context(results)
    user_prompt = f"Context:\n{context}\n\nQuestion: {message}"
    sources_md = format_sources_markdown(results)

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    partial = ""
    for chunk in stream:
        partial += chunk["message"]["content"]
        yield partial, sources_md


# ---------------------------------------------------------------------
with gr.Blocks(title="NSE RAG Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📈 NSE RAG Agent\n"
        "Ask questions about NSE filings and recent news. Answers are grounded "
        "**only** in what's been indexed locally -- no price predictions, no "
        "investment advice, everything runs on your machine via Ollama."
    )

    with gr.Row():
        with gr.Column(scale=2):
            ticker_dropdown = gr.Dropdown(
                choices=["All tickers"] + AVAILABLE_TICKERS,
                value="All tickers",
                label="Filter to a ticker",
            )
            chatbot = gr.Chatbot(height=480, label="Chat")
            msg = gr.Textbox(
                placeholder="e.g. What's driving RELIANCE stock this week?",
                label="Your question",
            )
            clear = gr.Button("Clear chat")

        with gr.Column(scale=1):
            gr.Markdown("### Sources used")
            sources_box = gr.Markdown("_Ask a question to see retrieved sources here._")

    def user_submit(message, history, ticker_filter):
        history = history + [{"role": "user", "content": message}]
        return "", history

    def bot_respond(history, ticker_filter):
        raw_content = history[-1]["content"]
        if isinstance(raw_content, list):
            user_message = "".join(
                part.get("text", "") for part in raw_content if isinstance(part, dict)
            )
        else:
            user_message = raw_content
        history = history + [{"role": "assistant", "content": ""}]
        for partial, sources_md in respond(user_message, history[:-1], ticker_filter):
            history[-1]["content"] = partial
            yield history, sources_md

    msg.submit(
        user_submit, [msg, chatbot, ticker_dropdown], [msg, chatbot], queue=False
    ).then(
        bot_respond, [chatbot, ticker_dropdown], [chatbot, sources_box]
    )

    clear.click(lambda: ([], "_Ask a question to see retrieved sources here._"), None, [chatbot, sources_box])

if __name__ == "__main__":
    print(f"Indexed tickers found: {AVAILABLE_TICKERS}")
    demo.launch()
