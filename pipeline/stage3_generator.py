"""
Stage 3 — LLM Generation

Changes from old version:
- Priority derived entirely from ticket text + criteria doc
  Transformer priority prediction is shown but explicitly marked UNRELIABLE
  so LLM does not anchor on it
- Priority criteria chunk always present (router now always retrieves it)
- Instruction block restructured: when dept is confident, LLM is told to
  trust dept but still reason priority independently
- When dept RAG ran, LLM is told to use RAG as primary dept signal and
  treat transformer as a secondary soft signal
- RAG gap warning retained — tells LLM when retrieval itself is uncertain
- priority_confident param kept for signature compatibility but always False
"""

import json, re
import os
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

PRIORITY_RULES = """
HIGH — assign when ANY of these apply:
  - Multiple users or entire organisation blocked from working
  - Revenue actively being lost (checkout down, payments failing live)
  - Financial transaction already occurred incorrectly (fraud, double charge)
  - Business-critical system completely unavailable, no workaround
  - Security incident in progress (breach, ransomware, unauthorised access)
  - Deal closing imminently, customer at risk of going to competitor
  - Return window about to expire, label not received
  - New employee cannot start due to missing access

MEDIUM — assign when the issue is real but not immediately blocking:
  - Single user affected, organisation still functional
  - Service degraded but partially working
  - Billing discrepancy exists but no money lost yet
  - Integration failing for one workflow, partial workaround exists
  - Recurring problem that needs resolution but is manageable
  - Active prospect evaluating but not yet at decision stage

LOW — assign when:
  - General question, no active problem
  - Informational or administrative request, no stated deadline
  - Early-stage inquiry, exploring options
  - Minor issue with a workaround, not blocking work
  - Post-incident review once service is restored
  - Documentation or guidance request with no urgency stated

Key signals for HIGH: "urgent", "critical", "all users", "cannot work",
"system down", "breach", "no workaround", "revenue at risk", "imminently"
Key signals for LOW: "no rush", "when you have a chance", "just wondering",
"for future reference", "exploring", "minor", "workaround in place"
Default to MEDIUM when ticket describes a real problem affecting one person
with no explicit urgency or explicit low-priority framing.
"""


def build_generation_prompt(cleaned_text, transformer_result, retrieved_chunks,
                            priority_chunk=None, rag_gap=0.0,
                            dept_confident=False, priority_confident=False):

    top3_lines = "\n".join(
        f"  {i+1}. {r['dept']} — {r['prob']*100:.1f}%"
        for i, r in enumerate(transformer_result["top3_dept"])
    )

    # Transformer block — priority labelled unreliable
    transformer_block = (
        f"TRANSFORMER PREDICTION (fine-tuned DistilBERT on synthetic data):\n"
        f"  Department : {transformer_result['dept']} "
        f"({transformer_result['dept_conf']*100:.1f}%) "
        f"[{'HIGH CONFIDENCE — dept RAG skipped' if dept_confident else 'uncertain — dept RAG retrieved'}]\n"
        f"  Priority   : {transformer_result['priority']} "
        f"({transformer_result['priority_conf']*100:.1f}%) "
        f"[UNRELIABLE — do not use for priority decision]\n"
        f"  Top-3 dept :\n{top3_lines}"
    )

    # Dept RAG block
    if retrieved_chunks:
        rag_lines = []
        for i, r in enumerate(retrieved_chunks):
            chunk = r["chunk"]
            rag_lines.append(
                f"[Chunk {i+1}] Department={chunk['dept']} | "
                f"Section={chunk['section']} | CE={r['ce_score']:.3f}\n"
                f"{chunk['raw_body'][:400]}..."
            )
        rag_block = "\n\n".join(rag_lines)

        if rag_gap < 0.15:
            rag_block = (
                    f"NOTE: RAG gap={rag_gap:.3f} — retrieved chunks are closely "
                    f"scored across departments. Retrieval is uncertain. "
                    f"Weight the ticket text heavily in your decision.\n\n"
                    + rag_block
            )
    else:
        rag_block = (
            "Dept RAG skipped — transformer confidence was >= 90%, "
            "department is almost certainly correct."
        )

    # Priority block — always present
    if priority_chunk:
        pchunk = priority_chunk["chunk"]
        priority_block = (
            f"Section={pchunk['section']} | CE={priority_chunk['ce_score']:.3f}\n"
            f"{pchunk['raw_body'][:500]}..."
        )
    else:
        priority_block = "Priority criteria not retrieved — use the rules above."

    # Instruction block — dept and priority framed separately and clearly
    if dept_confident:
        dept_instruction = (
            "DEPARTMENT: The transformer is >= 90% confident. "
            "Accept its department prediction unless the ticket text clearly "
            "contradicts it."
        )
    elif retrieved_chunks and rag_gap >= 0.15:
        dept_instruction = (
            "DEPARTMENT: The transformer was uncertain. "
            "Use the RAG routing rules below as your primary signal. "
            "The transformer top-3 is a soft secondary signal only."
        )
    else:
        dept_instruction = (
            "DEPARTMENT: Both transformer and RAG are uncertain. "
            "Reason directly from the ticket text and the routing rules. "
            "Set confidence to 'low'."
        )

    priority_instruction = (
        "PRIORITY: Ignore the transformer's priority entirely — it is trained "
        "on noisy labels and is unreliable. "
        "Derive priority solely from the ticket text against the priority rules "
        "provided. Apply the rules strictly."
    )

    return f"""You are an expert support ticket routing system.
Assign every ticket to exactly one department and one priority level.

{dept_instruction}

{priority_instruction}

---
TICKET:
{cleaned_text}

---
SOURCE 1 — TRANSFORMER:
{transformer_block}

---
SOURCE 2 — DEPT ROUTING RULES (RAG):
{rag_block}

---
SOURCE 3 — PRIORITY CRITERIA:
{priority_block}

---
PRIORITY RULES SUMMARY:
{PRIORITY_RULES}

---
VALID DEPARTMENTS (choose EXACTLY one):
{chr(10).join(f"- {d}" for d in VALID_DEPTS)}

VALID PRIORITIES: high, medium, low

---
Respond in this EXACT JSON format and nothing else:
{{
  "department": "<exact department name from the list above>",
  "priority": "<high|medium|low>",
  "confidence": "<high|medium|low>",
  "reasoning": "<2-3 sentences explaining your department and priority decisions based on the ticket text and evidence above>"
}}"""


def generate_routing(cleaned_text, transformer_result, retrieved_chunks, groq_client,
                     priority_chunk=None, rag_gap=0.0,
                     dept_confident=False, priority_confident=False):

    log.info(
        f"Stage 3 — transformer={transformer_result['dept']} "
        f"({transformer_result['dept_conf']*100:.1f}%)"
        + (f" | RAG top={retrieved_chunks[0]['chunk']['dept']} "
           f"(CE={retrieved_chunks[0]['ce_score']:.3f})"
           if retrieved_chunks else " | dept RAG skipped")
    )

    prompt = build_generation_prompt(
        cleaned_text, transformer_result, retrieved_chunks,
        priority_chunk=priority_chunk,
        rag_gap=rag_gap,
        dept_confident=dept_confident,
        priority_confident=priority_confident,
    )

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    log.debug(f"GPT raw response: {raw[:200]!r}")

    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in response")
        result = json.loads(match.group())

        if result.get("department") not in VALID_DEPTS:
            log.warning(
                f"Invalid dept '{result.get('department')}' — "
                f"falling back to transformer"
            )
            result["department"] = transformer_result["dept"]
            result["reasoning"] = (
                    result.get("reasoning", "") + " [dept fallback to transformer]"
            )

        if result.get("priority") not in ("high", "medium", "low"):
            log.warning(
                f"Invalid priority '{result.get('priority')}' — defaulting to medium"
            )
            result["priority"] = "medium"

        log.info(
            f"Stage 3 done — dept={result['department']} "
            f"priority={result['priority']} confidence={result['confidence']}"
        )
        return result

    except Exception as e:
        log.error(f"JSON parse failed: {e} — using transformer fallback")
        return {
            "department": transformer_result["dept"],
            "priority":   "medium",
            "confidence": "low",
            "reasoning":  "LLM parse failed — fallback to transformer dept, priority defaulted to medium.",
        }