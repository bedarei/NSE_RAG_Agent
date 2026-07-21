"""
Runs every question in eval_set.json through all 3 models (rag_utils.MODELS),
logging the raw answer plus timing/resource stats for each.

Ollama's chat response already includes token counts and durations, so we
get latency + tokens/sec for free without needing to hand-roll timers.
Resource footprint (RAM the model is holding) comes from `ollama ps`,
sampled right after each call.

Usage:
    python run_harness.py

Output: results_raw.json -- one row per (model, question) pair.
Then run score_harness.py to turn this into scored metrics.
"""
import json
import subprocess
import time

import ollama

from rag_utils import get_collection, retrieve, format_context, build_user_prompt, SYSTEM_PROMPT, MODELS, TOP_K

EVAL_SET_PATH = "eval_set.json"
OUTPUT_PATH = "results_raw.json"


def get_loaded_model_size_mb(model_name: str) -> float | None:
    """Parses `ollama ps` to find how much memory the currently loaded model
    is using. Returns None if the model isn't listed (e.g. it already
    unloaded, or the size column format changed on your Ollama version --
    if this returns None a lot, just check `ollama ps` output manually and
    adjust the parsing below)."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    for line in out.stdout.splitlines()[1:]:  # skip header row
        parts = line.split()
        if not parts:
            continue
        if parts[0] == model_name or parts[0].startswith(model_name.split(":")[0]):
            for token in parts:
                value_str = token[:-2]
                try:
                    if token.upper().endswith("GB") and value_str:
                        return float(value_str) * 1024
                    if token.upper().endswith("MB") and value_str:
                        return float(value_str)
                except ValueError:
                    continue  # not actually a size value (e.g. "100%GPU") -- skip it
    return None


def run_one(model: str, question_row: dict, collection) -> dict:
    question = question_row["question"]
    ticker_filter = question_row["ticker_filter"]

    # Use the pre-computed retrieval from build_eval_set.py for no_context rows
    # (empty on purpose); re-retrieve live for the rest so it reflects whatever
    # your chroma_db looks like right now.
    if question_row["category"] == "no_context":
        context = question_row["retrieved_context"]
    else:
        results = retrieve(collection, question, ticker_filter=ticker_filter, k=TOP_K)
        context = format_context(results) if results["documents"][0] else ""

    user_prompt = build_user_prompt(question, context)

    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    wall_clock_seconds = time.perf_counter() - start

    mem_mb = get_loaded_model_size_mb(model)

    eval_count = response.get("eval_count")  # tokens generated
    eval_duration_ns = response.get("eval_duration")  # ns spent generating
    tokens_per_sec = (
        eval_count / (eval_duration_ns / 1e9) if eval_count and eval_duration_ns else None
    )

    return {
        "question_id": question_row["id"],
        "category": question_row["category"],
        "model": model,
        "question": question,
        "context_used": context,
        "reference_answer": question_row.get("reference_answer"),
        "model_answer": response["message"]["content"],
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "total_duration_seconds": round(response.get("total_duration", 0) / 1e9, 3),
        "tokens_generated": eval_count,
        "tokens_per_second": round(tokens_per_sec, 2) if tokens_per_sec else None,
        "loaded_model_mem_mb": mem_mb,
    }


def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    collection = get_collection()
    results = []

    for model in MODELS:
        print(f"\n=== Running {model} over {len(eval_set)} questions ===")
        for i, row in enumerate(eval_set, start=1):
            print(f"  [{i}/{len(eval_set)}] {row['id']}: {row['question'][:60]}...")
            try:
                result = run_one(model, row, collection)
                results.append(result)
            except Exception as e:
                print(f"    !! failed: {e}")
                results.append({
                    "question_id": row["id"],
                    "category": row["category"],
                    "model": model,
                    "question": row["question"],
                    "error": str(e),
                })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()