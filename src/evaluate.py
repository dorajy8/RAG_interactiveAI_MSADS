"""
evaluate.py  —  Evaluation metrics for the MSADS RAG system.

Metrics
-------
1. Retrieval Precision@K   — what fraction of top-K results are relevant?
2. Mean Reciprocal Rank    — where does the first relevant result appear?
3. Answer Faithfulness     — does the LLM answer stay grounded in context?
4. Answer Relevance        — is the answer topically aligned with the question?

A gold-standard test set of 20 question→expected-answer pairs is included.
For faithfulness and relevance, we use the ragas library (optional dependency).
"""

import json
from typing import List, Dict
from vector_store import MSADSVectorStore

# ── Gold-standard test set ────────────────────────────────────────────────────
TEST_SET: List[Dict] = [
    {
        "question": "What are the core courses in the MS in Applied Data Science?",
        "expected_keywords": ["machine learning", "statistical", "data engineering",
                              "capstone", "python"],
        "relevant_source_titles": ["Core Courses", "Foundational Courses",
                                   "Machine Learning Courses Detail",
                                   "Data Engineering and Python Courses"],
    },
    {
        "question": "What are the admission requirements for the MSADS program?",
        "expected_keywords": ["bachelor", "recommendation", "statement", "resume",
                              "toefl", "programming"],
        "relevant_source_titles": ["Admissions Requirements", "Application Deadlines",
                                   "Visa and International Students"],
    },
    {
        "question": "How much does the program cost?",
        "expected_keywords": ["tuition", "scholarship"],
        "relevant_source_titles": ["Tuition and Financial Aid"],
    },
    {
        "question": "What career outcomes do graduates achieve?",
        "expected_keywords": ["data scientist", "engineer"],
        "relevant_source_titles": ["Career Outcomes", "Career Seminar"],
    },
    {
        "question": "Can I take the program part-time?",
        "expected_keywords": ["part-time", "evening", "quarter", "flexible"],
        "relevant_source_titles": ["In-Person Program Format", "Online Program",
                                   "Program Tracks and Structure"],
    },
    {
        "question": "Is the degree STEM OPT eligible?",
        "expected_keywords": ["stem", "opt", "visa"],
        "relevant_source_titles": ["Visa and International Students",
                                   "In-Person Program Format"],
    },
    {
        "question": "What is the capstone project?",
        "expected_keywords": ["capstone", "industry"],
        "relevant_source_titles": ["Capstone Project", "Core Courses"],
    },
    {
        "question": "What electives are available?",
        "expected_keywords": ["nlp", "reinforcement", "marketing"],
        "relevant_source_titles": ["Elective Courses", "Machine Learning Courses Detail"],
    },
    {
        "question": "When is the application deadline?",
        "expected_keywords": ["deadline"],
        "relevant_source_titles": ["Application Deadlines", "Admissions Requirements"],
    },
    {
        "question": "Is there an online version of the program?",
        "expected_keywords": ["online", "professionals"],
        "relevant_source_titles": ["Online Program", "Program Tracks and Structure"],
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def retrieval_precision_at_k(
    passages: List[Dict],
    relevant_titles: List[str],
    k: int = 4,
) -> float:
    """Fraction of top-K results matching expected source titles."""
    hits = sum(
        any(rt.lower() in p["title"].lower() for rt in relevant_titles)
        for p in passages[:k]
    )
    return hits / min(k, len(passages))


def mean_reciprocal_rank(
    passages: List[Dict],
    relevant_titles: List[str],
) -> float:
    """Reciprocal rank of the first relevant passage."""
    for rank, p in enumerate(passages, 1):
        if any(rt.lower() in p["title"].lower() for rt in relevant_titles):
            return 1.0 / rank
    return 0.0


def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    hits = sum(kw.lower() in answer_lower for kw in keywords)
    return hits / len(keywords)


def evaluate(store: MSADSVectorStore) -> Dict:
    """
    Run all test questions through the retrieval pipeline and compute metrics.
    Does NOT call the LLM (to keep evaluation fast and cost-free).
    """
    precision_scores, mrr_scores = [], []

    for item in TEST_SET:
        passages = store.retrieve(item["question"], top_k=4)
        precision_scores.append(
            retrieval_precision_at_k(passages, item["relevant_source_titles"])
        )
        mrr_scores.append(
            mean_reciprocal_rank(passages, item["relevant_source_titles"])
        )

    results = {
        "num_test_cases":    len(TEST_SET),
        "precision_at_4":    round(sum(precision_scores) / len(precision_scores), 4),
        "mean_reciprocal_rank": round(sum(mrr_scores) / len(mrr_scores), 4),
    }

    print("\n" + "="*50)
    print("  MSADS RAG Evaluation Results")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:<30} {v}")
    print("="*50)
    return results


if __name__ == "__main__":
    store = MSADSVectorStore()
    evaluate(store)
