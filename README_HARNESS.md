# Model comparison harness — NSE RAG agent

Compares Llama 3.1 8B vs Llama 3.2 3B vs Phi-3 as the generation model,
on faithfulness, correctness, instruction adherence, latency, and
resource footprint.

## Setup

1. Drop these files into the same folder as your `chroma_db/` (i.e. next
   to `app.py`).
2. Pull the 3 models being compared, plus a separate judge model
   (deliberately not one of the 3, so it isn't grading itself):
   ```
   ollama pull llama3.1:8b
   ollama pull llama3.2:3b
   ollama pull phi3
   ollama pull qwen2.5:7b
   ```
3. `pip install pandas` (everything else is already in your requirements.txt)

## Run order

```
python build_eval_set.py    # samples your real chroma_db -> eval_set.json
```
**Stop here and open `eval_set.json`.** Skim the auto-generated questions,
delete/edit any that don't make sense for your data, and fill in
`reference_answer` for as many `ticker_specific`/`cross_ticker` rows as you
have patience for (10-15 is plenty). Rows without a reference answer still
get scored on faithfulness/adherence, just not correctness.

```
python run_harness.py       # eval_set.json -> results_raw.json
python score_harness.py     # results_raw.json -> results_scored.csv + summary_by_model.csv
```

`summary_by_model.csv` is the table for your report. `results_scored.csv`
has the per-question detail if you want to pull specific examples for the
tradeoffs write-up (e.g. "Phi-3 hallucinated on q004 but was 3x faster").

## Notes / things worth knowing before you run it

- `run_harness.py` re-runs retrieval live per question (except the
  deliberately-unanswerable `no_context` rows) so it reflects your current
  index, not a stale snapshot from when you built the eval set.
- Resource footprint comes from parsing `ollama ps` right after each call.
  If `loaded_model_mem_mb` comes back empty a lot, run `ollama ps` by hand
  while a model is loaded and check the column format matches what
  `get_loaded_model_size_mb()` expects — Ollama has changed this output
  format between versions before.
- The judge is still an LLM, not ground truth — spot-check a handful of its
  scores against your own read of the transcripts before trusting the
  aggregate numbers in the report.
