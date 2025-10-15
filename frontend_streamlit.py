import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Knowledge-base RAG — Query Interface")

with st.expander("Ingest a file (PDF / text)"):
    uploaded = st.file_uploader("Upload file", type=["pdf", "txt"])
    source_name = st.text_input("Source name (optional)")
    if st.button("Ingest") and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.read())}
        data = {"source_name": source_name}
        res = requests.post(f"{API_URL}/ingest", files=files, data=data)
        st.write(res.json())

st.header("Query the KB")
q = st.text_input("Question")
top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=4)
if st.button("Ask") and q:
    res = requests.post(f"{API_URL}/query", json={"query": q, "top_k": int(top_k)})
    if res.ok:
        r = res.json()
        st.subheader("Answer")
        st.write(r["answer"])
        st.subheader("Sources (retrieved chunks metadata)")
        st.write(r["sources"])
        if "evaluation" in r:
            st.subheader("Evaluation Metrics")
            for k, v in r["evaluation"].items():
                st.write(f"{k}: {v}")
    else:
        st.error("Error: " + res.text)
