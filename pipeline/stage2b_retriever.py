"""
Stage 2b — Hybrid RAG Retrieval

Changes from old version:
- Query embedding now uses all-MiniLM-L6-v2 (not the DistilBERT classifier encoder)
  Reason: compliance docs are written in policy language, not ticket language.
  all-MiniLM-L6-v2 is domain-agnostic and trained for semantic similarity.
- CrossEncoder swapped to BAAI/bge-reranker-base (better document reranking)
- CE score gap returned so router can detect when RAG is also uncertain
- encoder/tokenizer/device params removed from retrieval functions (no longer needed)
"""

import re, pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from pipeline.logger import get_logger

log = get_logger("stage2b.retriever")


def tokenize_for_bm25(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def load_rag_artifacts(hf_repo_id, hf_token):
    log.info(f"Loading RAG artifacts from {hf_repo_id}")

    faiss_index = faiss.read_index(
        hf_hub_download(hf_repo_id, "rag_compliance_index.faiss",
                        token=hf_token, repo_type="model")
    )
    with open(hf_hub_download(hf_repo_id, "rag_bm25_index.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        bm25_data = pickle.load(f)
    with open(hf_hub_download(hf_repo_id, "rag_compliance_metadata.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        all_chunks = pickle.load(f)

    priority_index = faiss.read_index(
        hf_hub_download(hf_repo_id, "rag_priority_index.faiss",
                        token=hf_token, repo_type="model")
    )
    with open(hf_hub_download(hf_repo_id, "rag_priority_metadata.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        priority_chunks = pickle.load(f)

    # Sentence embedder for query → FAISS search
    # Uses all-MiniLM-L6-v2: domain-agnostic, trained for semantic similarity
    # NOT the DistilBERT classifier — that encoder learned ticket patterns,
    # not policy-doc patterns, causing domain mismatch in retrieval.
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # BGE reranker: trained for document reranking (not web search like ms-marco)
    # Gives more meaningful CE scores for policy-language compliance docs
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    log.info(f"Dept index: {faiss_index.ntotal} vectors | "
             f"Priority index: {priority_index.ntotal} vectors")
    log.info(f"Dept chunks: {len(all_chunks)} | Priority chunks: {len(priority_chunks)}")
    return (faiss_index, bm25_data["bm25"], all_chunks,
            embedder, cross_encoder, priority_index, priority_chunks)


def hybrid_retrieve(query, embedder, faiss_index, bm25, all_chunks, cross_encoder,
                    top_k_dense=10, top_k_bm25=10, top_n_final=4):
    """
    Retrieve top dept-definition chunks for the query.

    Returns list of dicts:
        {"chunk": {...}, "ce_score": float}
    Plus a "rag_gap" float = CE score difference between rank-1 and rank-2.
    Large gap = RAG is confident. Small gap = RAG is also uncertain → tell LLM.
    """
    log.debug(f"Retrieving for: {query[:100]!r}")

    # Embed query with all-MiniLM-L6-v2
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    _, dense_ids = faiss_index.search(q_emb, top_k_dense)
    dense_ids = dense_ids[0].tolist()

    bm25_scores = bm25.get_scores(tokenize_for_bm25(query))
    bm25_ids = np.argsort(bm25_scores)[::-1][:top_k_bm25].tolist()

    # RRF fusion
    rrf = {}
    for rank, idx in enumerate(dense_ids):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_ids):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)

    candidate_ids = [
        i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:20]
    ]

    raw_scores = cross_encoder.predict(
        [[query, all_chunks[i]["text"]] for i in candidate_ids]
    )
    # BGE reranker returns raw logits — apply sigmoid to get 0-1 scores
    import math
    ce_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
    ranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: x[1], reverse=True)

    results = [
        {"chunk": all_chunks[idx], "ce_score": float(score)}
        for idx, score in ranked[:top_n_final]
    ]

    # CE gap: how much more confident is rank-1 vs rank-2?
    # Used by router to detect when RAG is itself uncertain
    rag_gap = (
        float(ranked[0][1] - ranked[1][1])
        if len(ranked) >= 2 else 1.0
    )

    log.info(
        f"Stage 2b — top: {results[0]['chunk']['dept']} (CE={results[0]['ce_score']:.3f})"
        + (f" | 2nd: {results[1]['chunk']['dept']} (CE={results[1]['ce_score']:.3f})"
           if len(results) > 1 else "")
        + f" | gap={rag_gap:.3f}"
    )
    return results, rag_gap


def retrieve_priority_chunk(query, embedder, priority_index, priority_chunks, cross_encoder):
    """Retrieve best matching priority criteria section (HIGH / MEDIUM / LOW)."""
    if not priority_index or not priority_chunks:
        return None

    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    k = min(len(priority_chunks), 3)
    _, ids = priority_index.search(q_emb, k)
    cand_ids = ids[0].tolist()

    raw_scores = cross_encoder.predict(
        [[query, priority_chunks[i]["text"]] for i in cand_ids]
    )
    import math
    ce_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
    best = int(np.argmax(ce_scores))

    chunk = priority_chunks[cand_ids[best]]
    score = float(ce_scores[best])
    log.info(f"Stage 2b priority — section: {chunk['section']} (CE={score:.3f})")
    return {"chunk": chunk, "ce_score": score}