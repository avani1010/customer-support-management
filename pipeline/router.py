"""
router.py — Orchestrates all stages with confidence-gated RAG.

Flow:
  Stage 1  : LLM rewrites raw ticket into clean structured text.
  Stage 2a : Fine-tuned DistilBERT predicts dept + priority with confidence scores.
  Gate     : If transformer is confident (dept_conf >= DEPT_CONF_THRESHOLD and
             priority_conf >= PRIORITY_CONF_THRESHOLD), trust it directly — skip
             RAG and the second LLM call entirely.
  Stage 2b : Only reached when transformer is uncertain. Hybrid retrieval
             (FAISS + BM25 + RRF + CrossEncoder) fetches compliance chunks.
  Stage 3  : LLM arbitrates between transformer signals and RAG evidence.
"""

from dataclasses import dataclass, field
from pipeline.logger import get_logger
from pipeline.stage1_rewriter import rewrite_query
from pipeline.stage2a_transformer import transformer_predict
from pipeline.stage2b_retriever import hybrid_retrieve, retrieve_priority_chunk
from pipeline.stage3_generator import generate_routing

log = get_logger("router")

DEPT_CONF_THRESHOLD     = 0.85
PRIORITY_CONF_THRESHOLD = 0.60


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
    rag_used: bool                        # whether RAG was triggered
    rag_reason: str                       # human-readable explanation of why RAG was/wasn't used
    rag_chunks: list
    priority_chunk: dict | None = field(default=None)


def route_ticket(raw_text, groq_client,
                 model, tokenizer, queue_encoder, priority_encoder, device,
                 retrieval_embedder,
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

    dept_conf     = transformer_result["dept_conf"]
    priority_conf = transformer_result["priority_conf"]
    dept_ok       = dept_conf >= DEPT_CONF_THRESHOLD
    priority_ok   = priority_conf >= PRIORITY_CONF_THRESHOLD
    high_conf     = dept_ok and priority_ok

    # Build a precise human-readable reason for the gate decision
    if high_conf:
        rag_reason = (
            f"Dept {dept_conf*100:.1f}% ≥ {DEPT_CONF_THRESHOLD*100:.0f}% threshold  ·  "
            f"Priority {priority_conf*100:.1f}% ≥ {PRIORITY_CONF_THRESHOLD*100:.0f}% threshold  →  RAG skipped"
        )
    elif not dept_ok and not priority_ok:
        rag_reason = (
            f"Dept {dept_conf*100:.1f}% < {DEPT_CONF_THRESHOLD*100:.0f}% threshold  ·  "
            f"Priority {priority_conf*100:.1f}% < {PRIORITY_CONF_THRESHOLD*100:.0f}% threshold  →  RAG invoked"
        )
    elif not dept_ok:
        rag_reason = (
            f"Dept {dept_conf*100:.1f}% < {DEPT_CONF_THRESHOLD*100:.0f}% threshold  ·  "
            f"Priority {priority_conf*100:.1f}% ✓  →  RAG invoked"
        )
    else:
        rag_reason = (
            f"Dept {dept_conf*100:.1f}% ✓  ·  "
            f"Priority {priority_conf*100:.1f}% < {PRIORITY_CONF_THRESHOLD*100:.0f}% threshold  →  RAG invoked"
        )

    log.info(f"Confidence gate — {rag_reason}")

    if high_conf:
        # ── Fast path : transformer is confident, skip RAG + LLM ───────────
        log.info("✓ High confidence — routing directly from transformer")
        log.info("─" * 60)
        log.info(f"FINAL → dept      : {transformer_result['dept']}")
        log.info(f"FINAL → priority  : {transformer_result['priority']}")
        log.info(f"FINAL → confidence: high (transformer direct)")
        log.info("─" * 60)

        return RoutingResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            department=transformer_result["dept"],
            priority=transformer_result["priority"],
            confidence="high",
            reasoning=(
                f"Transformer confident: dept {dept_conf*100:.1f}%, "
                f"priority {priority_conf*100:.1f}%. RAG not invoked."
            ),
            transformer_dept=transformer_result["dept"],
            transformer_conf=dept_conf,
            transformer_top3=transformer_result["top3_dept"],
            priority_probs=transformer_result["priority_probs"],
            rag_used=False,
            rag_reason=rag_reason,
            rag_chunks=[],
            priority_chunk=None,
        )

    # ── Slow path : at least one threshold failed ────────────────────────────
    log.info(f"✗ Low confidence — dept_ok={dept_ok} priority_ok={priority_ok}")

    # Stage 2b : only retrieve dept chunks if dept was uncertain
    # If dept was confident (dept_ok=True), trust transformer directly — no FAISS needed
    if not dept_ok:
        log.info("  Retrieving dept definition chunks (dept confidence low)")
        retrieved_chunks = hybrid_retrieve(
            cleaned_text, retrieval_embedder,
            faiss_index, bm25, all_chunks, cross_encoder,
            top_n_final=4
        )
    else:
        log.info(f"  Skipping dept retrieval — transformer confident ({dept_conf*100:.1f}%)")
        retrieved_chunks = []

    # Stage 2b : only retrieve priority chunk if priority was uncertain
    if not priority_ok:
        log.info("  Retrieving priority chunk (priority confidence low)")
        priority_chunk = retrieve_priority_chunk(
            cleaned_text, retrieval_embedder,
            priority_index, priority_chunks, cross_encoder
        )
    else:
        log.info(f"  Skipping priority retrieval — transformer confident ({priority_conf*100:.1f}%)")
        priority_chunk = None

    # Stage 3 : dept = CrossEncoder top result if retrieved, else transformer direct
    #           priority = LLM reasoning if priority_chunk retrieved, else transformer direct
    if not dept_ok or not priority_ok:
        generation = generate_routing(
            cleaned_text, transformer_result, retrieved_chunks, groq_client,
            priority_chunk=priority_chunk
        )
    else:
        # Should not reach here but guard anyway
        generation = {
            "department": transformer_result["dept"],
            "priority"  : transformer_result["priority"],
            "confidence": "high",
            "reasoning" : "Transformer direct — both thresholds met.",
        }

    dept_from_rag   = generation["department"]
    dept_from_trans = transformer_result["dept"]
    agree = dept_from_rag == dept_from_trans
    log.info("─" * 60)
    log.info(f"FINAL → dept      : {dept_from_rag} {'(transformer direct)' if dept_ok else '(RAG CrossEncoder)' if agree else f'(RAG overrides transformer: {dept_from_trans})'}")
    log.info(f"FINAL → priority  : {generation['priority']} {'(transformer direct)' if priority_ok else '(LLM decision)'}")
    log.info(f"FINAL → confidence: {generation['confidence']}")
    log.info("─" * 60)

    return RoutingResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        department=generation["department"],
        priority=generation["priority"],
        confidence=generation["confidence"],
        reasoning=generation["reasoning"],
        transformer_dept=transformer_result["dept"],
        transformer_conf=dept_conf,
        transformer_top3=transformer_result["top3_dept"],
        priority_probs=transformer_result["priority_probs"],
        rag_used=True,
        rag_reason=rag_reason,
        rag_chunks=retrieved_chunks,
        priority_chunk=priority_chunk,
    )