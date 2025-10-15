import io
import os
import pdfplumber
from typing import List
import re
from dotenv import load_dotenv

load_dotenv()
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE_CHARS", 2000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_CHARS", 200))

def extract_text_from_file(content: bytes, filename: str) -> str:
    """
    Support PDF and plain text. Returns concatenated extracted text.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        # use pdfplumber for better extraction
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    else:
        # treat as text
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")

def _clean_text(s: str) -> str:
    # basic cleanup
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str) -> List[str]:
    """
    Chunk text into overlapping windows based on chars (simple).
    """
    text = _clean_text(text)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + CHUNK_SIZE, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - CHUNK_OVERLAP
    return chunks
