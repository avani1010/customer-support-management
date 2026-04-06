"""
Stage 3 — LLM Generation

Changes from old version:
- generate_routing now accepts rag_gap, dept_confident, priority_confident
- Prompt tells LLM explicitly when both transformer AND RAG are uncertain
- Prompt tells LLM when only one signal (dept or priority) is uncertain
- This prevents the LLM from false-high-confidence on ambiguous tickets
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


def build_generation_prompt(cleaned_text, transformer_result, retrieved_chunks,
                             priority_chunk=None, rag_gap=0.0,
                             dept_confident=False, priority_confident=False):

    top3_lines = "\n".join(
        f"  {i+1}. {r['dept']} — {r['prob']*100:.1f}%"
        for i, r in enumerate(transformer_result["top3_dept"])
    )
    transformer_block = (
        f"TRANSFORMER PREDICTION (fine-tuned DistilBERT):\n"
        f"  Department : {transformer_result['dept']} "
        f"({transformer_result['dept_conf']*100:.1f}%) "
        f"[{'CONFIDENT' if dept_confident else 'UNCERTAIN — below threshold'}]\n"
        f"  Priority   : {transformer_result['priority']} "
        f"({transformer_result['priority_conf']*100:.1f}%) "
        f"[{'CONFIDENT' if priority_confident else 'UNCERTAIN — below threshold'}]\n"
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

        # Warn the LLM if RAG is also uncertain
        if rag_gap < 0.15:
            rag_block = (
                f"⚠ NOTE: RAG gap={rag_gap:.3f} is small — "
                f"retrieved chunks are similarly ranked across departments. "
                f"RAG is also uncertain. Use the top chunk as a soft signal only.\n\n"
                + rag_block
            )
    else:
        rag_block = "RAG retrieval skipped — transformer was confident on department."

    # Priority block
    if priority_chunk:
        pchunk = priority_chunk["chunk"]
        priority_block = (
            f"Section={pchunk['section']} | CE={priority_chunk['ce_score']:.3f}\n"
            f"{pchunk['raw_body'][:400]}..."
        )
    else:
        priority_block = (
            "Priority retrieval skipped — transformer was confident on priority."
            if priority_confident
            else "No priority chunk retrieved."
        )

    # Context-aware instruction block
    if not dept_confident and not priority_confident:
        instruction_note = (
            "⚠ BOTH transformer and RAG are uncertain. "
            "Reason carefully from the ticket text itself. "
            "Set confidence to 'low' unless the evidence is clear."
        )
    elif not dept_confident:
        instruction_note = (
            "Department is uncertain — use RAG chunks to determine the correct department. "
            "Priority was confident from transformer — trust it unless priority criteria contradicts."
        )
    elif not priority_confident:
        instruction_note = (
            "Department is confident from transformer — trust it. "
            "Priority is uncertain — use the priority criteria chunk to determine the correct level."
        )
    else:
        instruction_note = "Both signals were confident."

    return f"""You are an expert support ticket routing system.
Assign every ticket to exactly one department and one priority level.

{instruction_note}

You have three evidence sources:
1. Transformer classifier — fine-tuned DistilBERT, confidence scores shown
2. Dept compliance chunks — routing rules defining what each department handles
3. Priority criteria chunk — universal escalation rules (HIGH / MEDIUM / LOW)

Where sources agree, that strengthens the decision.
Where they disagree, reason about which is more specific to this ticket.

---
TICKET:
{cleaned_text}

---
SOURCE 1 — TRANSFORMER:
{transformer_block}

---
SOURCE 2 — DEPT ROUTING RULES:
{rag_block}

---
SOURCE 3 — PRIORITY CRITERIA:
{priority_block}

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
  "reasoning": "<2-3 sentences explaining why both transformer and compliance evidence support this decision>"
}}"""


def generate_routing(cleaned_text, transformer_result, retrieved_chunks, groq_client,
                     priority_chunk=None, rag_gap=0.0,
                     dept_confident=False, priority_confident=False):

    log.info(
        f"Stage 3 — transformer={transformer_result['dept']} "
        f"({transformer_result['dept_conf']*100:.1f}%)"
        + (f" | RAG top={retrieved_chunks[0]['chunk']['dept']} "
           f"(CE={retrieved_chunks[0]['ce_score']:.3f})"
           if retrieved_chunks else " | RAG skipped")
    )

    prompt = build_generation_prompt(
        cleaned_text, transformer_result, retrieved_chunks,
        priority_chunk=priority_chunk,
        rag_gap=rag_gap,
        dept_confident=dept_confident,
        priority_confident=priority_confident,
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    log.debug(f"Groq raw response: {raw[:200]!r}")

    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in response")
        result = json.loads(match.group())

        if result.get("department") not in VALID_DEPTS:
            log.warning(f"Invalid dept '{result.get('department')}' — "
                        f"falling back to transformer")
            result["department"] = transformer_result["dept"]
            result["reasoning"] = (result.get("reasoning", "")
                                    + " [dept fallback to transformer]")

        if result.get("priority") not in ("high", "medium", "low"):
            log.warning(f"Invalid priority '{result.get('priority')}' — "
                        f"falling back to transformer")
            result["priority"] = transformer_result["priority"]

        log.info(f"Stage 3 done — dept={result['department']} "
                 f"priority={result['priority']} confidence={result['confidence']}")
        return result

    except Exception as e:
        log.error(f"JSON parse failed: {e} — using transformer fallback")
        return {
            "department": transformer_result["dept"],
            "priority":   transformer_result["priority"],
            "confidence": "low",
            "reasoning":  "LLM parse failed — fallback to transformer prediction.",
        }
