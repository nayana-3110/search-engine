import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./persist")
EMBED_DIM = None

client = OpenAI(api_key=OPENAI_API_KEY)

class VectorStore:
    """
    Simple file-backed FAISS + metadata store.
    """
    def __init__(self, persist_dir: str = "./persist"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "meta.pkl")
        self.texts_path = os.path.join(self.persist_dir, "texts.pkl")
        self._load()

    def _load(self):
        self.id_to_meta = {}
        self.id_to_text = {}
        self.next_id = 0
        self.index = None
        # load texts and meta if exist
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.id_to_meta = pickle.load(f)
        if os.path.exists(self.texts_path):
            with open(self.texts_path, "rb") as f:
                self.id_to_text = pickle.load(f)
        # load faiss
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # infer embed dim from index
            global EMBED_DIM
            EMBED_DIM = self.index.d
            self.next_id = len(self.id_to_text)
        else:
            self.index = None

    def _save(self):
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.id_to_meta, f)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.id_to_text, f)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def _ensure_index(self, dim: int):
        global EMBED_DIM
        if self.index is None:
            EMBED_DIM = dim
            self.index = faiss.IndexFlatL2(dim)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # call OpenAI embeddings API in a single batch
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [item.embedding for item in resp.data]
        return embeddings

    def add_documents(self, chunks: List[str], metadata: Dict = None) -> List[int]:
        """
        Add chunks (list of strings) to vector store. Returns internal ids.
        """
        if metadata is None:
            metadata = {}
        embeddings = self._embed(chunks)
        dim = len(embeddings[0])
        self._ensure_index(dim)
        # prepare
        arr = np.array(embeddings, dtype="float32")
        self.index.add(arr)
        ids = []
        for i, chunk in enumerate(chunks):
            idx = self.next_id
            self.id_to_text[idx] = chunk
            self.id_to_meta[idx] = {"index": idx, **(metadata or {})}
            ids.append(idx)
            self.next_id += 1
        # persist
        self._save()
        return ids

    def similarity_search(self, query: str, top_k: int = 4):
        """
        Returns list of hits with text and metadata sorted by distance ascending.
        """
        q_emb = self._embed([query])[0]
        xq = np.array([q_emb], dtype="float32")
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(xq, top_k)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            hits.append({
                "id": int(idx),
                "score": float(dist),
                "text": self.id_to_text[int(idx)],
                "metadata": self.id_to_meta.get(int(idx), {})
            })
        return hits
