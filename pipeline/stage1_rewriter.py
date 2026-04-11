"""
Stage 1 — Query Rewriting
Groq LLM cleans the raw ticket before it enters the transformer or retriever.
"""
import json, re
import os

from groq import Groq

from pipeline.logger import get_logger


log = get_logger("stage1.rewriter")
from dotenv import load_dotenv
load_dotenv()

def rewrite_query(raw_text: str) -> str:
    log.info(f"Stage 1 — rewriting ticket ({len(raw_text)} chars)")
    log.debug(f"Raw input: {raw_text[:120]!r}")

    prompt = (
        "You are a customer-support triage assistant. "
        "Given a raw support ticket, return ONLY valid JSON with these keys:\n"
        "  structured_body: cleaned, professional rewrite of the ticket\n"
        "  subject: concise 6-10 word subject line\n"
        "  urgency_signals: list of urgency phrases found\n"
        "  tech_keywords: list of technical terms found\n"
        "  explanation: one sentence explaining the likely department\n"
        "Return nothing outside the JSON object.\n"
        "Preserve the customer's original vocabulary as much as possible.\n "
        "Do not substitute synonyms for domain-specific terms the customer used.\n "
        "IMPORTANT: In structured_body, rewrite only what the customer said.\n"
        "Do NOT add words like 'technical', 'billing', 'outage', or any "
        "department-specific vocabulary that was not in the original ticket."
        f"TICKET:\n{raw_text.strip()}"
    )

    # response = groq_client.chat.completions.create(
    #     model="llama-3.3-70b-versatile",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.0,
    #     max_tokens=500,
    # )
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()

    # Try to parse JSON and extract structured_body
    def _extract_body_from_llm(raw_str, fallback):
        """Robustly extract structured_body even from malformed LLM JSON."""
        # Strip markdown fences
        s = re.sub(r"^```(?:json)?\s*", "", raw_str, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```\s*$", "", s).strip()

        # Attempt 1: valid JSON parse
        try:
            obj = json.loads(s)
            return obj.get("structured_body", fallback).strip(), obj
        except Exception:
            pass

        # Attempt 2: extract outermost {...} and retry (handles trailing garbage)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj.get("structured_body", fallback).strip(), obj
            except Exception:
                pass

        # Attempt 3: regex-extract just the structured_body value directly
        # handles truncated/unclosed JSON like: "structured_body": "Some text
        m2 = re.search(
            r'"structured_body"\s*:\s*"(.*?)(?:"|$)',
            s, re.DOTALL | re.IGNORECASE
        )
        if m2:
            body = m2.group(1).strip().rstrip('",')
            if len(body) > 5:
                log.warning("Partial JSON — extracted structured_body via regex")
                return body, {}

        # Attempt 4: fall back to original raw ticket text (NOT the broken LLM output)
        log.warning("All JSON extraction attempts failed — using original ticket text")
        return fallback.strip(), {}

    cleaned, parsed = _extract_body_from_llm(raw, raw_text)
    if parsed:
        log.info(f"Stage 1 done — subject: {parsed.get('subject','?')!r}")
        log.debug(f"Urgency signals : {parsed.get('urgency_signals', [])}")
        log.debug(f"Tech keywords   : {parsed.get('tech_keywords', [])}")
        log.debug(f"Explanation     : {parsed.get('explanation','')}")
    else:
        log.warning(f"Stage 1 — JSON parse failed, using cleaned fallback")

    log.debug(f"Cleaned text: {cleaned[:120]!r}")
    return cleaned