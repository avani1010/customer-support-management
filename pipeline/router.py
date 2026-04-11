"""
router.py — Orchestrates all 3 stages.

Changes from old version:
- Per-class confidence thresholds (derived from V6 Cell 15 output)
  replaces single global threshold
- embedder (all-MiniLM-L6-v2) passed through instead of model.encoder
- hybrid_retrieve now returns (chunks, rag_gap) — gap fed to Stage 3
- Priority threshold set to 0.65 (3-class head has softer ceiling)
- RAG only activates for the signal that failed its threshold
"""

from dataclasses import dataclass, field
from pipeline.logger import get_logger
from pipeline.stage1_rewriter import rewrite_query
from pipeline.stage2a_transformer import transformer_predict
from pipeline.stage2b_retriever import hybrid_retrieve, retrieve_priority_chunk
from pipeline.stage3_generator import generate_routing

log = get_logger("router")

# ── Per-class confidence thresholds ───────────────────────────────────────────
# Derived from V6 Cell 15 (per-class accuracy on test set).
# Lower threshold = lower bar to trigger RAG = RAG used more often for that class.
# Classes with low train count (CS=166, Outages=109) get lower thresholds
# so RAG compensates for transformer weakness on those classes.
DEPT_THRESHOLDS = {
    "Billing and Payments":            0.72,  # F1=0.99, very confident
    "Customer Service":                0.60,  # F1=0.60, transformer weak here
    "General Inquiry":                 0.65,  # F1=0.64
    "Human Resources":                 0.65,  # F1=0.65
    "Returns and Exchanges":           0.72,  # F1=0.68
    "Sales and Pre-Sales":             0.72,  # F1=0.71
    "Service Outages and Maintenance": 0.60,  # F1=0.76 but only 109 train samples
    "Technical & IT Support":          0.72,  # F1=0.78
}
PRIORITY_THRESHOLD = 0.65   # 3-class head — natural ceiling ~0.70

# When RAG gap is below this, both transformer AND RAG are uncertain
# → tell the LLM explicitly so it can express low confidence
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
    fast_path:        bool = False   # True = transformer only, RAG skipped


def route_ticket(raw_text,
                 model, tokenizer, queue_encoder, priority_encoder, device,
                 embedder,                          # all-MiniLM-L6-v2 SentenceTransformer
                 faiss_index, bm25, all_chunks, cross_encoder,
                 priority_index, priority_chunks):

    log.info("=" * 60)
    log.info(f"NEW TICKET ({len(raw_text)} chars): {raw_text[:80]!r}")
    log.info("=" * 60)

    # ── Stage 1: clean and restructure raw ticket ──────────────────────────
    cleaned_text = rewrite_query(raw_text)

    # ── Stage 2a: transformer classifies dept + priority ───────────────────
    transformer_result = transformer_predict(
        cleaned_text, model, tokenizer, queue_encoder, priority_encoder, device
    )

    dept_conf     = transformer_result["dept_conf"]
    priority_conf = transformer_result["priority_conf"]
    predicted_dept = transformer_result["dept"]

    # Per-class threshold for the predicted department
    dept_threshold = DEPT_THRESHOLDS.get(predicted_dept, 0.70)

    dept_confident     = dept_conf     >= dept_threshold
    priority_confident = priority_conf >= PRIORITY_THRESHOLD

    log.info(f"Confidence gate — dept: {dept_conf:.3f} vs {dept_threshold} → "
             f"{'PASS' if dept_confident else 'FAIL'} | "
             f"priority: {priority_conf:.3f} vs {PRIORITY_THRESHOLD} → "
             f"{'PASS' if priority_confident else 'FAIL'}")

    # ── Fast path: both signals confident, skip RAG entirely ──────────────
    if dept_confident and priority_confident:
        log.info("⚡ FAST PATH — transformer confident on both signals, skipping RAG")
        return RoutingResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            department=predicted_dept,
            priority=transformer_result["priority"],
            confidence="high",
            reasoning=(
                f"Transformer predicted {predicted_dept} with "
                f"{dept_conf*100:.1f}% confidence and "
                f"{transformer_result['priority']} priority with "
                f"{priority_conf*100:.1f}% confidence. "
                f"Both exceed thresholds — fast path taken."
            ),
            transformer_dept=predicted_dept,
            transformer_conf=dept_conf,
            transformer_top3=transformer_result["top3_dept"],
            priority_probs=transformer_result["priority_probs"],
            rag_chunks=[],
            rag_gap=0.0,
            priority_chunk=None,
            fast_path=True,
        )

    # ── Slow path: retrieve what's needed ─────────────────────────────────
    retrieved_chunks = []
    rag_gap = 0.0

    if not dept_confident:
        # Dept uncertain → run full hybrid retrieval
        retrieved_chunks, rag_gap = hybrid_retrieve(
            cleaned_text, embedder,
            faiss_index, bm25, all_chunks, cross_encoder,
            top_n_final=4
        )
        both_uncertain = rag_gap < RAG_GAP_UNCERTAIN
        if both_uncertain:
            log.warning(f"⚠ RAG gap={rag_gap:.3f} < {RAG_GAP_UNCERTAIN} — "
                        f"both transformer and RAG uncertain on dept")

    priority_chunk = None
    if not priority_confident:
        # Priority uncertain → retrieve priority criteria
        priority_chunk = retrieve_priority_chunk(
            cleaned_text, embedder,
            priority_index, priority_chunks, cross_encoder
        )

    # ── Stage 3: LLM reads everything and decides ─────────────────────────
    generation = generate_routing(
        cleaned_text, transformer_result, retrieved_chunks,
        priority_chunk=priority_chunk,
        rag_gap=rag_gap,
        dept_confident=dept_confident,
        priority_confident=priority_confident,
    )

    agree = generation["department"] == predicted_dept
    log.info("─" * 60)
    log.info(f"FINAL → dept      : {generation['department']}")
    log.info(f"FINAL → priority  : {generation['priority']}")
    log.info(f"FINAL → confidence: {generation['confidence']}")
    log.info(f"FINAL → agreement : "
             f"{'✓ transformer agrees' if agree else '✗ Groq overrides transformer'}")
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
        fast_path=False,
    )
