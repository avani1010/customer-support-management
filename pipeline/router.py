"""
router.py — Orchestrates all 3 stages.
"""

from dataclasses import dataclass, field
from pipeline.logger import get_logger
from pipeline.stage1_rewriter import rewrite_query
from pipeline.stage2a_transformer import transformer_predict
from pipeline.stage2b_retriever import hybrid_retrieve, retrieve_priority_chunk
from pipeline.stage3_generator import generate_routing

log = get_logger("router")


@dataclass
class RoutingResult:
    raw_text: str
    cleaned_text: str
    department: str
    priority: str
    confidence: str
    reasoning: str
    transformer_dept: str
    transformer_conf: float
    transformer_top3: list
    priority_probs: dict
    rag_chunks: list
    priority_chunk: dict | None = field(default=None)


def route_ticket(raw_text, groq_client,
                 model, tokenizer, queue_encoder, priority_encoder, device,
                 faiss_index, bm25, all_chunks, cross_encoder,
                 priority_index, priority_chunks):

    log.info("=" * 60)
    log.info(f"NEW TICKET ({len(raw_text)} chars): {raw_text[:80]!r}")
    log.info("=" * 60)

    # Stage 1 : clean and restructure raw ticket
    cleaned_text = rewrite_query(raw_text, groq_client)

    # Stage 2a : transformer classifies dept + priority
    transformer_result = transformer_predict(
        cleaned_text, model, tokenizer, queue_encoder, priority_encoder, device
    )

    # Stage 2b : retrieve dept chunks (transformer encoder → HNSW + BM25 + RRF + CrossEncoder)
    retrieved_chunks = hybrid_retrieve(
        cleaned_text, model.encoder, tokenizer, device,
        faiss_index, bm25, all_chunks, cross_encoder,
        top_n_final=4
    )

    # Stage 2b : retrieve priority chunk separately (never competes with dept slots)
    priority_chunk = retrieve_priority_chunk(
        cleaned_text, model.encoder, tokenizer, device,
        priority_index, priority_chunks, cross_encoder
    )

    # Stage 3 : LLM reads ticket + transformer signals + dept chunks + priority chunk
    generation = generate_routing(
        cleaned_text, transformer_result, retrieved_chunks, groq_client,
        priority_chunk=priority_chunk
    )

    agree = generation["department"] == transformer_result["dept"]
    log.info("─" * 60)
    log.info(f"FINAL → dept      : {generation['department']}")
    log.info(f"FINAL → priority  : {generation['priority']}")
    log.info(f"FINAL → confidence: {generation['confidence']}")
    log.info(f"FINAL → agreement : {'✓ transformer agrees' if agree else '✗ Groq overrides transformer'}")
    log.info("─" * 60)

    return RoutingResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        department=generation["department"],
        priority=generation["priority"],
        confidence=generation["confidence"],
        reasoning=generation["reasoning"],
        transformer_dept=transformer_result["dept"],
        transformer_conf=transformer_result["dept_conf"],
        transformer_top3=transformer_result["top3_dept"],
        priority_probs=transformer_result["priority_probs"],
        rag_chunks=retrieved_chunks,
        priority_chunk=priority_chunk,
    )
