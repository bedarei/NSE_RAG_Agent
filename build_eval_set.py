"""
Generates a starter eval_set.json from whatever's actually indexed in
chroma_db -- real tickers, real dates, real filing/news types -- instead
of hand-typing questions blind.

This is a STARTER set, not a finished one. Run it, then open eval_set.json
and:
  1. Skim each question -- delete/edit ones that don't make sense for your data
  2. For as many rows as you have patience for, fill in "reference_answer"
     by reading the retrieved_context yourself. Rows with a reference_answer
     get scored on correctness; rows without only get faithfulness/relevancy/
     adherence scores. Aim for at least 10-15 filled in.
  3. Keep the "no_context" category questions as-is -- they test whether a
     model correctly says "not enough info" instead of hallucinating, which
     only works if the ticker genuinely isn't indexed.

Usage:
    python build_eval_set.py
"""
import json
import random

from rag_utils import get_collection, retrieve, format_context, TOP_K

OUTPUT_PATH = "eval_set.json"
N_TICKER_SPECIFIC = 10
N_CROSS_TICKER = 8
N_NO_CONTEXT = 4

QUESTION_TEMPLATES_TICKER = [
    "What's the latest news or filing update for {ticker}?",
    "Have there been any recent board announcements for {ticker}?",
    "What's driving {ticker} stock recently?",
    "Summarize the most recent filing from {ticker}.",
    "Has {ticker} announced any dividends recently?",
]

QUESTION_TEMPLATES_CROSS = [
    "Which companies have had board meetings scheduled recently?",
    "What recent news is there across the indexed stocks?",
    "Are there any companies with upcoming AGMs?",
    "Which stocks have recent price-moving news?",
]

FAKE_TICKERS_FOR_NO_CONTEXT = ["ZZTOPCORP", "NOTAREALTICKER", "FAKEINDLTD", "GHOSTSTOCKS"]


def main():
    collection = get_collection()
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    if not all_metas:
        raise SystemExit("chroma_db collection is empty -- run the ingestion notebook cells first.")

    tickers = sorted({m["ticker"] for m in all_metas})
    print(f"Found {len(tickers)} indexed tickers: {tickers}")

    eval_rows = []
    row_id = 0

    # 1. Ticker-specific questions, sampled across real tickers
    sample_tickers = random.sample(tickers, k=min(N_TICKER_SPECIFIC, len(tickers)))
    templates_cycle = QUESTION_TEMPLATES_TICKER * (len(sample_tickers) // len(QUESTION_TEMPLATES_TICKER) + 1)
    for ticker, template in zip(sample_tickers, templates_cycle):
        question = template.format(ticker=ticker)
        results = retrieve(collection, question, ticker_filter=ticker, k=TOP_K)
        if not results["documents"][0]:
            continue
        eval_rows.append({
            "id": f"q{row_id:03d}",
            "category": "ticker_specific",
            "ticker_filter": ticker,
            "question": question,
            "retrieved_context": format_context(results),
            "reference_answer": None,
        })
        row_id += 1

    # 2. Cross-ticker questions, no filter
    for template in QUESTION_TEMPLATES_CROSS[:N_CROSS_TICKER]:
        results = retrieve(collection, template, ticker_filter=None, k=TOP_K)
        if not results["documents"][0]:
            continue
        eval_rows.append({
            "id": f"q{row_id:03d}",
            "category": "cross_ticker",
            "ticker_filter": None,
            "question": template,
            "retrieved_context": format_context(results),
            "reference_answer": None,
        })
        row_id += 1

    # 3. Deliberately unanswerable questions (ticker not indexed) --
    #    tests whether the model admits "not enough info" instead of hallucinating
    for fake_ticker in FAKE_TICKERS_FOR_NO_CONTEXT[:N_NO_CONTEXT]:
        question = f"What's the latest news for {fake_ticker}?"
        eval_rows.append({
            "id": f"q{row_id:03d}",
            "category": "no_context",
            "ticker_filter": fake_ticker,
            "question": question,
            "retrieved_context": "",  # deliberately empty -- expect "no info found"
            "reference_answer": "Should state that no relevant information was found.",
        })
        row_id += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)

    n_filled = sum(1 for r in eval_rows if r["reference_answer"])
    print(f"Wrote {len(eval_rows)} questions to {OUTPUT_PATH}")
    print(f"({n_filled} have a reference_answer already; go fill in more for ticker_specific/cross_ticker rows)")


if __name__ == "__main__":
    main()
