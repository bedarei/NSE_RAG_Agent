# NSE RAG Agent

An on-prem retrieval-augmented generation (RAG) system that answers questions about NSE-listed companies using locally indexed filings and news. Runs entirely through Ollama, with no data leaving the machine.

## Overview

The agent indexes company filings and news articles into a Chroma vector store, retrieves relevant context for a user's question, and generates a grounded answer using a local LLM. It's built around four NSE-listed tickers (RELIANCE, INFY, TCS, HDFCBANK) and exposes a Gradio chat interface for querying.

**Stack:** Ollama (Llama 3.1/3.2, embeddings via nomic-embed-text), ChromaDB, Gradio, Python

## Project structure
- nse_rag_agent.ipynb          (data ingestion + indexing into Chroma)  
- app.py                       (Gradio UI for querying the agent)  
- requirements.txt  
- NSE_RAG_Model_Comparison_Report.pdf    (write-up comparing 3 LLMs for this task)  
- harness/                     (evaluation harness used for the model comparison)  
- README.md                 (setup + run order for the harness)  
- rag_utils.py              (shared retrieval/config, mirrors app.py)  
- build_eval_set.py         (generates eval questions from the real index)  
- run_harness.py            (runs eval set through each model)  
- score_harness.py          (LLM-judge scoring (faithfulness, correctness, instruction adherence)  
- eval_set.json  
- results_raw.json  
- results_scored.csv  
- summary_by_model.csv  

## Setup

1. Install [Ollama](https://ollama.com) and pull the models used by the agent:  
- ollama pull llama3.1:8b  
- ollama pull nomic-embed-text  
2. Install Python dependencies:  
pip install -r requirements.txt
3. Run `nse_rag_agent.ipynb` to ingest filings/news and build the local Chroma index (`chroma_db/`, not included in this repo since it's regenerated locally).
4. Launch the UI:  
python app.py

## Model comparison

As an extension of the project, three candidate generation models (Llama 3.1 8B, Llama 3.2 3B, and Phi-3) were evaluated head to head on faithfulness, correctness, instruction adherence, latency, and throughput. The full methodology, results, and findings are in [`NSE_RAG_Model_Comparison_Report.pdf`](./NSE_RAG_Model_Comparison_Report.pdf); the evaluation harness itself is in [`harness/`](./harness), which is reusable for evaluating any future model swapped into this agent.

Headline finding: the largest model tested was, counterintuitively, the worst at admitting when it had no relevant information, tending to fabricate plausible-sounding details on unanswerable questions rather than saying so.
