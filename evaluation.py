"""
Evaluation utilities for LLM output.
"""
from typing import List, Dict, Optional
import re

def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def evaluate_llm_output(
    answer: str,
    query: str,
    contexts: List[str],
    expected_keywords: Optional[List[str]] = None,
    ground_truth: Optional[str] = None
) -> Dict:
    """
    Evaluate the LLM's answer for main metrics: retrieval accuracy, sources cited, answer length, conciseness, and prompt following.
    Uses Jaccard similarity for retrieval accuracy.
    Returns a dict with evaluation metrics.
    """
    result = {}
    # --- Retrieval Accuracy (Jaccard similarity) ---
    threshold = 0.2  # consider context used if Jaccard similarity > 0.2
    overlap_count = sum(1 for c in contexts if jaccard_similarity(c, answer) > threshold)
    result["retrieval_accuracy"] = overlap_count / max(len(contexts), 1)
    # --- Sources Cited ---
    result["sources_cited"] = bool(re.search(r"SOURCES\s*[:：]", answer, re.I))
    # --- Answer Length ---
    result["answer_length"] = len(answer)
    # --- Conciseness ---
    result["concise"] = len(answer.split()) < 150  # arbitrary threshold
    # --- Follows Prompt ---
    result["follows_prompt"] = result["sources_cited"] and bool(answer.strip())
    return result
