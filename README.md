# Query_Search — Retrieval-Augmented Generation (RAG) 

A simple local knowledge-base RAG project that:

- ingests documents of type(PDF / text),
- creates embeddings and stores them in a FAISS vector store,
- retrieves top-k relevant chunks for a user query,
- synthesizes an answer with an LLM (OpenAI-compatible client), and
- evaluates the LLM output with lightweight metrics.

Contents
--------

- `app.py` — FastAPI server exposing `/ingest` and `/query` endpoints.
- `frontend_streamlit.py` — Streamlit UI for ingesting files and querying the knowledge base.
- `ingestion.py` — helpers to extract text from files and chunk text into passages.
- `vectorstore.py` — FAISS-based vector store with simple metadata support.
- `llm.py` — thin wrapper to call the configured LLM for synthesizing answers.
- `evaluation.py` — lightweight evaluation utilities for LLM outputs (retrieval accuracy, sources cited, length/conciseness, prompt following).
- `requirements.txt` — Python dependencies.

Quickstart
----------

1. Create a Python virtual environment and activate it:

	```powershell
	python -m venv venv
	.\venv\Scripts\activate
	```

2. Install dependencies:

	```powershell
	pip install -r requirements.txt
	```

3. Create a `.env` file with your OpenAI API key and optionally override defaults:

	```text
	OPENAI_API_KEY=sk-...
	LLM_MODEL=gpt-4
	EMBEDDING_MODEL=text-embedding-3-large
	PERSIST_DIR=./persist
	API_URL=http://localhost:8000
	```

	Important: do NOT commit `.env` to your repository. `.gitignore` already includes `.env`.

4. Run the API server:

	```powershell
	uvicorn app:app --reload
	```

5. Run the Streamlit frontend in another terminal:

	```powershell
	streamlit run frontend_streamlit.py
	```

Usage
-----

- Ingest a file via Streamlit or POST `/ingest` to upload PDFs or text files. The server will extract text, chunk it, create embeddings, and persist them.
- Query the knowledge base via Streamlit or POST `/query` with JSON `{ "query": "...", "top_k": 4 }`. The response includes the synthesized answer, retrieved chunk metadata, and a small `evaluation` object with metrics.

Evaluation metrics
------------------

The project computes a few focused evaluation metrics for each synthesized answer:

- `retrieval_accuracy`: fraction of retrieved contexts that the answer appears to use (Jaccard token overlap with a small threshold).
- `sources_cited`: whether the LLM included a `SOURCES:` line as requested by the prompt.
- `answer_length`: number of characters in the generated answer.
- `concise`: boolean heuristic for short answers (<150 tokens).
- `follows_prompt`: whether the answer followed prompt guidance (non-empty + cited sources).

These metrics are intentionally simple and meant to provide a lightweight signal. For production, consider semantic similarity checks, human evaluation, or automated fact-checking against ground truth.

## Demo of  API server

[![Watch the demo](https://github.com/ambatiredd/query_search/blob/main/video/fast_api.gif)](https://github.com/ambatiredd/query_search/blob/main/video/fast_api.mp4)


## Demo of Streamlit frontend
[![Watch the demo](https://github.com/ambatiredd/query_search/blob/main/video/frontend.gif)](https://github.com/ambatiredd/query_search/blob/main/video/frontend.mp4)


Notes and next steps
--------------------

- Add unit tests for ingestion and vector store operations.
- Improve evaluation with semantic overlap metrics (embedding-based similarity) or automatic fact-checking.
- Add CI to run linting and tests before allowing merges.

Contact
-------

pulagampremanayana2005@gmail.com

Prema Nayana Reddy P
