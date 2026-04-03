"""
Stage 2b — Hybrid RAG Retrieval
"""

import re, pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from pipeline.logger import get_logger

log = get_logger("stage2b.retriever")

# Dedicated retrieval embedding model: trained for semantic similarity,
# not repurposed from the classification backbone.
RETRIEVAL_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize_for_bm25(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def load_rag_artifacts(hf_repo_id, hf_token):
    log.info(f"Loading RAG artifacts from {hf_repo_id}")

    # Dedicated retrieval embedder — loaded here alongside the indexes it was used to build
    embedder = SentenceTransformer(RETRIEVAL_MODEL_ID)
    log.info(f"Retrieval embedder ready — dim: {embedder.get_sentence_embedding_dimension()}")

    # Dept definition index
    faiss_index = faiss.read_index(
        hf_hub_download(hf_repo_id, "rag_dept_index.faiss",
                        token=hf_token, repo_type="model")
    )
    with open(hf_hub_download(hf_repo_id, "rag_bm25_index.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        bm25_data = pickle.load(f)
    with open(hf_hub_download(hf_repo_id, "rag_dept_metadata.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        all_chunks = pickle.load(f)

    # Priority escalation index
    priority_index = faiss.read_index(
        hf_hub_download(hf_repo_id, "rag_priority_index.faiss",
                        token=hf_token, repo_type="model")
    )
    with open(hf_hub_download(hf_repo_id, "rag_priority_metadata.pkl",
                              token=hf_token, repo_type="model"), "rb") as f:
        priority_chunks = pickle.load(f)

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    log.info(f"Dept index: {faiss_index.ntotal} vectors | Priority index: {priority_index.ntotal} vectors")
    log.info(f"Dept chunks: {len(all_chunks)} | Priority chunks: {len(priority_chunks)}")
    return embedder, faiss_index, bm25_data["bm25"], all_chunks, cross_encoder, priority_index, priority_chunks


def _embed_query(query: str, embedder: SentenceTransformer) -> np.ndarray:
    """Encode a query string into a normalised float32 vector."""
    emb = embedder.encode(query, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype("float32").reshape(1, -1)


def hybrid_retrieve(query, embedder,
                    faiss_index, bm25, all_chunks, cross_encoder,
                    top_k_dense=10, top_k_bm25=10, top_n_final=4):
    log.debug(f"Retrieving for: {query[:100]!r}")

    # Sentence embedder -> query vector (domain-matched to how docs were indexed)
    q_emb = _embed_query(query, embedder)

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

    candidate_ids = [i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:20]]
    ce_scores = cross_encoder.predict([[query, all_chunks[i]["text"]] for i in candidate_ids])
    ranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: x[1], reverse=True)

    results = [{"chunk": all_chunks[idx], "ce_score": float(score)} for idx, score in ranked[:top_n_final]]

    log.info(f"Stage 2b — top: {results[0]['chunk']['dept']} (CE={results[0]['ce_score']:.3f})"
             + (f" | 2nd: {results[1]['chunk']['dept']} (CE={results[1]['ce_score']:.3f})" if len(results) > 1 else ""))
    return results


def retrieve_priority_chunk(query, embedder,
                            priority_index, priority_chunks, cross_encoder):
    if not priority_index or not priority_chunks:
        return None

    q_emb = _embed_query(query, embedder)

    k = min(len(priority_chunks), 3)
    _, ids = priority_index.search(q_emb, k)
    cand_ids = ids[0].tolist()

    ce_scores = cross_encoder.predict([[query, priority_chunks[i]["text"]] for i in cand_ids])
    best = int(np.argmax(ce_scores))

    chunk = priority_chunks[cand_ids[best]]
    score = float(ce_scores[best])
    log.info(f"Stage 2b priority — section: {chunk['section']} (CE={score:.3f})")
    return {"chunk": chunk, "ce_score": score}