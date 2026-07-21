"""
Scores results_raw.json using an LLM judge (JUDGE_MODEL in rag_utils.py --
deliberately not one of the 3 compared models) on 3 rubric dimensions:

  - faithfulness:          1-5, is the answer grounded in context_used?
  - correctness:           1-5, only scored when reference_answer exists
  - instruction_adherence: 1-5, did it avoid investment advice/price predictions
                            and cite ticker/date per the system prompt?

Latency and resource-footprint metrics don't need a judge -- they're already
numeric in results_raw.json from run_harness.py.

Usage:
    python score_harness.py

Output:
    results_scored.csv   -- one row per (model, question) with judge scores
    summary_by_model.csv -- averaged metrics per model, ready to drop into
                             your report table
"""
import json
import re

import ollama
import pandas as pd

from rag_utils import JUDGE_MODEL

RAW_RESULTS_PATH = "results_raw.json"
SCORED_CSV_PATH = "results_scored.csv"
SUMMARY_CSV_PATH = "summary_by_model.csv"

JUDGE_PROMPT_TEMPLATE = """You are grading one response from a financial RAG assistant. \
Score strictly and briefly. Respond with ONLY a JSON object, no other text.

QUESTION:
{question}

CONTEXT PROVIDED TO THE MODEL:
{context}

REFERENCE ANSWER:
{reference_answer}

MODEL'S ANSWER TO GRADE:
{model_answer}

Score each on a 1-5 integer scale (5 = best):
- faithfulness: Is every claim in the model's answer actually supported by the context?
  Score 1 if it invents facts not in the context, 5 if fully grounded.
- correctness: Does the answer match the reference answer's substance? A REFERENCE ANSWER
  IS PROVIDED ABOVE -- it is not empty -- so you MUST give an integer 1-5 here, never null.
  Score 1 if the answer contradicts or completely misses the reference, 3 if it's partial or
  vaguer than the reference, 5 if it covers the same substance. Do not penalize the model for
  giving a shorter or more cautious answer than the reference -- score whether what it DID say
  is consistent with the reference, not whether it said everything the reference said.
- instruction_adherence: Does it avoid investment advice/price predictions and mention
  which ticker(s)/date(s) it drew from, as instructed? 1 if it gives advice/predictions
  or omits sourcing, 5 if it fully follows the instruction.

Respond with exactly this JSON shape (correctness must be an integer, not null, since a
reference answer was provided above):
{{"faithfulness": <1-5>, "correctness": <1-5>, "instruction_adherence": <1-5>, "notes": "<one short sentence>"}}
"""

RETRY_NUDGE = """Your previous response gave "correctness": null, but a reference answer WAS \
provided. Re-score correctness as an integer from 1 (contradicts/misses the reference) to \
5 (matches its substance) -- do not use null. Respond with ONLY the same JSON object, \
with correctness now filled in as an integer."""


def judge_one(row: dict) -> dict:
    if "error" in row:
        return {"faithfulness": None, "correctness": None, "instruction_adherence": None, "notes": "run error, not scored"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=row["question"],
        context=row["context_used"] or "(no context retrieved)",
        reference_answer=row.get("reference_answer") or "(none provided)",
        model_answer=row["model_answer"],
    )
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(model=JUDGE_MODEL, messages=messages)
    parsed = _parse_judge_json(response["message"]["content"], row["question_id"])

    has_reference = bool(row.get("reference_answer"))
    if has_reference and parsed.get("correctness") is None:
        # Judge dodged with null despite a reference being provided -- nudge it once
        messages.append({"role": "assistant", "content": response["message"]["content"]})
        messages.append({"role": "user", "content": RETRY_NUDGE})
        retry_response = ollama.chat(model=JUDGE_MODEL, messages=messages)
        retry_parsed = _parse_judge_json(retry_response["message"]["content"], row["question_id"] + " (retry)")
        if retry_parsed.get("correctness") is not None:
            parsed["correctness"] = retry_parsed["correctness"]

    return parsed


def _parse_judge_json(raw: str, question_id: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"    !! could not parse judge output for {question_id}: {raw[:200]}")
        return {"faithfulness": None, "correctness": None, "instruction_adherence": None, "notes": "unparseable judge output"}


def main():
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        raw_results = json.load(f)

    scored_rows = []
    for i, row in enumerate(raw_results, start=1):
        print(f"[{i}/{len(raw_results)}] scoring {row['question_id']} / {row['model']}")
        scores = judge_one(row)
        scored_rows.append({**row, **scores})

    df = pd.DataFrame(scored_rows)
    df.to_csv(SCORED_CSV_PATH, index=False)
    print(f"\nWrote per-question scores to {SCORED_CSV_PATH}")

    summary = df.groupby("model").agg(
        n_questions=("question_id", "count"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_correctness=("correctness", "mean"),
        avg_instruction_adherence=("instruction_adherence", "mean"),
        avg_wall_clock_seconds=("wall_clock_seconds", "mean"),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_loaded_model_mem_mb=("loaded_model_mem_mb", "mean"),
    ).round(2)

    summary.to_csv(SUMMARY_CSV_PATH)
    print(f"Wrote per-model summary to {SUMMARY_CSV_PATH}\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()