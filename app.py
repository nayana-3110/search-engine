import os
import io
import json
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ingestion import extract_text_from_file, chunk_text
from vectorstore import VectorStore
from llm import synthesize_answer
from dotenv import load_dotenv
from evaluation import evaluate_llm_output

load_dotenv()

PERSIST_DIR = os.environ.get("PERSIST_DIR", "./persist")
os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI(title="Knowledge-base RAG API")
vs = VectorStore(persist_dir=PERSIST_DIR)  # loads existing or new

class QueryRequest(BaseModel):
    query: str
    top_k: int = int(os.environ.get("TOP_K", 4))

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), source_name: str = Form(None)):
    """
    Upload a file (PDF or text) to ingest. Returns ingestion metadata.
    """
    content = await file.read()
    filename = file.filename or f"upload-{uuid.uuid4().hex}"
    text = extract_text_from_file(content, filename)
    # chunk text
    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())
    # add to vector store
    ids = vs.add_documents(chunks, metadata={"source": filename, "doc_id": doc_id, "source_name": source_name})
    return JSONResponse({"status": "ok", "doc_id": doc_id, "num_chunks": len(chunks), "ids": ids})

@app.post("/query")
def query(q: QueryRequest):
    """
    Query the knowledge base. Returns synthesized answer and sources.
    """
    # retrieve top-k chunks
    hits = vs.similarity_search(q.query, top_k=q.top_k)
    retrieved_texts = [h["text"] for h in hits]
    sources = [h["metadata"] for h in hits]
    # synthesize
    answer = synthesize_answer(query=q.query, contexts=retrieved_texts)
    # evaluate
    evaluation = evaluate_llm_output(answer, q.query, retrieved_texts)
    return {"answer": answer, "sources": sources, "evaluation": evaluation}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)), reload=True)
