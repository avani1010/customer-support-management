from dataclasses import dataclass, field
from pipeline.logger import get_logger
from pipeline.stage1_rewriter import rewrite_query
from pipeline.stage2a_transformer import transformer_predict
from pipeline.stage2b_retriever import hybrid_retrieve, retrieve_priority_chunk
from pipeline.stage3_generator import generate_routing

log = get_logger("router")

# ── Dept confidence threshold ─────────────────────────────────────────────────
# Single threshold for all classes — calibrated against real-world eval.
# Per-class thresholds were derived from synthetic test set and were too
# permissive (0.60-0.72), causing fast path to fire on wrong predictions.
# At 0.90+ the transformer is almost certainly correct on unambiguous tickets.
# When dept confidence is below this, full hybrid RAG retrieval runs.
DEPT_CONFIDENT_THRESHOLD = 0.90

# RAG gap below this → both transformer AND retrieval are uncertain on dept
# → LLM is told explicitly so it can set confidence to low
RAG_GAP_UNCERTAIN = 0.15


@dataclass
class RoutingResult:
    raw_text:         str
    cleaned_text:     str
    department:       str
    priority:         str
    confidence:       str
    reasoning:        str
    transformer_dept: str
    transformer_conf: float
    transformer_top3: list
    priority_probs:   dict
    rag_chunks:       list
    rag_gap:          float = 0.0
    priority_chunk:   dict | None = field(default=None)
    # True = transformer was confident on dept, dept RAG skipped
    # Stage 3 still always runs for priority
    dept_rag_skipped: bool = False


def route_ticket(raw_text, groq_client,
                 model, tokenizer, queue_encoder, priority_encoder, device,
                 embedder,
                 faiss_index, bm25, all_chunks, cross_encoder,
                 priority_index, priority_chunks):

    log.info("=" * 60)
    log.info(f"NEW TICKET ({len(raw_text)} chars): {raw_text[:80]!r}")
    log.info("=" * 60)

    # ── Stage 1: rewrite raw ticket ───────────────────────────────────────
    cleaned_text = rewrite_query(raw_text)

    # ── Stage 2a: transformer classifies dept + priority ──────────────────
    transformer_result = transformer_predict(
        cleaned_text, model, tokenizer, queue_encoder, priority_encoder, device
    )

    dept_conf      = transformer_result["dept_conf"]
    predicted_dept = transformer_result["dept"]
    dept_confident = dept_conf >= DEPT_CONFIDENT_THRESHOLD

    log.info(
        f"Dept confidence: {dept_conf:.3f} vs {DEPT_CONFIDENT_THRESHOLD} → "
        f"{'dept RAG SKIPPED' if dept_confident else 'dept RAG RUNNING'}"
    )
    log.info(
        f"Priority: transformer says {transformer_result['priority']} "
        f"({transformer_result['priority_conf']*100:.1f}%) — "
        f"IGNORED, Stage 3 will derive from criteria doc"
    )

    # ── Stage 2b: retrieval ───────────────────────────────────────────────
    retrieved_chunks = []
    rag_gap = 0.0

    if not dept_confident:
        # Dept uncertain — run full hybrid retrieval
        retrieved_chunks, rag_gap = hybrid_retrieve(
            cleaned_text, embedder,
            faiss_index, bm25, all_chunks, cross_encoder,
            top_n_final=4
        )
        if rag_gap < RAG_GAP_UNCERTAIN:
            log.warning(
                f"RAG gap={rag_gap:.3f} < {RAG_GAP_UNCERTAIN} — "
                f"both transformer and RAG uncertain on dept"
            )
    else:
        log.info(
            f"Dept RAG skipped — transformer very confident: "
            f"{predicted_dept} ({dept_conf*100:.1f}%)"
        )

    # Priority chunk — always retrieved, never gated
    priority_chunk = retrieve_priority_chunk(
        cleaned_text, embedder,
        priority_index, priority_chunks, cross_encoder
    )

    # ── Stage 3: LLM decides dept + priority ──────────────────────────────
    # priority_confident always False — Stage 3 always derives priority itself
    generation = generate_routing(
        cleaned_text, transformer_result, retrieved_chunks, groq_client,
        priority_chunk=priority_chunk,
        rag_gap=rag_gap,
        dept_confident=dept_confident,
        priority_confident=False,
    )

    agree = generation["department"] == predicted_dept
    log.info("─" * 60)
    log.info(f"FINAL → dept      : {generation['department']}")
    log.info(f"FINAL → priority  : {generation['priority']}")
    log.info(f"FINAL → confidence: {generation['confidence']}")
    log.info(
        f"FINAL → dept agreement: "
        f"{'transformer agrees' if agree else 'LLM overrides transformer'}"
    )
    log.info("─" * 60)

    return RoutingResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        department=generation["department"],
        priority=generation["priority"],
        confidence=generation["confidence"],
        reasoning=generation["reasoning"],
        transformer_dept=predicted_dept,
        transformer_conf=dept_conf,
        transformer_top3=transformer_result["top3_dept"],
        priority_probs=transformer_result["priority_probs"],
        rag_chunks=retrieved_chunks,
        rag_gap=rag_gap,
        priority_chunk=priority_chunk,
        dept_rag_skipped=dept_confident,
    )