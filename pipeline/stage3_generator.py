"""
Stage 3 — LLM Generation 
"""

import json, re
from groq import Groq
from pipeline.logger import get_logger

log = get_logger("stage3.generator")

VALID_DEPTS = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "Technical & IT Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
]


def resolve_department_from_rag(retrieved_chunks, transformer_result):
    """
    Department resolution:
    - If retrieved_chunks is non-empty (dept was uncertain), use CrossEncoder top chunk.
    - If retrieved_chunks is empty (dept was confident, retrieval skipped), use transformer directly.
    """
    if not retrieved_chunks:
        log.info(
            f"Dept resolved from transformer directly (retrieval skipped): "
            f"{transformer_result['dept']} ({transformer_result['dept_conf']*100:.1f}%)"
        )
        return transformer_result["dept"], None

    for item in retrieved_chunks:
        dept = item["chunk"].get("dept", "")
        if dept in VALID_DEPTS:
            log.info(
                f"Dept resolved from RAG top chunk: {dept} "
                f"(CE={item['ce_score']:.3f}) — transformer was "
                f"{transformer_result['dept']} ({transformer_result['dept_conf']*100:.1f}%)"
            )
            return dept, item["ce_score"]

    log.warning("No valid dept in RAG chunks — falling back to transformer")
    return transformer_result["dept"], None


def build_priority_prompt(cleaned_text, transformer_result, priority_chunk):
    """
    Focused prompt: only asks the LLM to decide priority.
    Gives it the ticket, the transformer's priority signal, and the
    escalation criteria chunk to reason against.
    """
    transformer_prio_block = (
            f"TRANSFORMER PRIORITY PREDICTION:\n"
            f"  Predicted : {transformer_result['priority']} "
            f"({transformer_result['priority_conf']*100:.1f}% confidence)\n"
            f"  All probs : "
            + "  ".join(
        f"{k}: {v*100:.1f}%"
        for k, v in sorted(
            transformer_result["priority_probs"].items(),
            key=lambda x: -x[1]
        )
    )
    )

    if priority_chunk:
        pchunk = priority_chunk["chunk"]
        criteria_block = (
            f"Section={pchunk['section']} | CE={priority_chunk['ce_score']:.3f}\n"
            f"{pchunk['raw_body'][:600]}"
        )
    else:
        criteria_block = "No priority criteria chunk retrieved."

    return f"""You are a support ticket priority classifier.

Your only job is to assign the correct priority level (high, medium, or low) to this ticket.

You have two inputs:
1. Transformer prediction — a fine-tuned model's priority signal from ticket language patterns
2. Priority escalation criteria — the official rules defining what makes a ticket high/medium/low

Apply the criteria to this specific ticket. If the criteria clearly match a different level
than the transformer predicted, override it. If the criteria are ambiguous, trust the transformer.

---
TICKET:
{cleaned_text}

---
SOURCE 1 — TRANSFORMER PRIORITY SIGNAL:
{transformer_prio_block}

---
SOURCE 2 — PRIORITY ESCALATION CRITERIA:
{criteria_block}

---
VALID PRIORITIES: high, medium, low

Respond in this EXACT JSON format and nothing else:
{{
  "priority": "<high|medium|low>",
  "confidence": "<high|medium|low>",
  "reasoning": "<1-2 sentences citing which specific criteria justify this priority level>"
}}"""


def generate_routing(cleaned_text, transformer_result, retrieved_chunks,
                     groq_client, priority_chunk=None):
    """
    Stage 3 entry point.

    Department: resolved deterministically from CrossEncoder top chunk.
    Priority  : resolved by LLM reasoning against escalation criteria.
    """
    # ── Department: CrossEncoder decides, no LLM needed ───────────────────
    department, ce_score = resolve_department_from_rag(retrieved_chunks, transformer_result)

    # ── Priority: LLM reasons against criteria chunk ───────────────────────
    ce_str = f"{ce_score:.3f}" if ce_score is not None else "fallback"
    log.info(
        f"Stage 3 — dept from RAG: {department} (CE={ce_str}) "
        f"| calling LLM for priority only"
    )

    prompt = build_priority_prompt(cleaned_text, transformer_result, priority_chunk)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,   # much smaller — only returning priority JSON now
    )
    raw = response.choices[0].message.content.strip()
    log.debug(f"Groq raw response: {raw[:200]!r}")

    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in response")
        result = json.loads(match.group())

        if result.get("priority") not in ("high", "medium", "low"):
            log.warning(f"Invalid priority '{result.get('priority')}' — falling back to transformer")
            result["priority"]  = transformer_result["priority"]
            result["reasoning"] = result.get("reasoning", "") + " [priority fallback to transformer]"

        log.info(
            f"Stage 3 done — dept={department} "
            f"priority={result['priority']} confidence={result['confidence']}"
        )
        return {
            "department": department,
            "priority"  : result["priority"],
            "confidence": result.get("confidence", "medium"),
            "reasoning" : result.get("reasoning", ""),
        }

    except Exception as e:
        log.error(f"JSON parse failed: {e} — using transformer fallback for priority")
        return {
            "department": department,
            "priority"  : transformer_result["priority"],
            "confidence": "low",
            "reasoning" : "LLM priority parse failed — fallback to transformer prediction.",
        }