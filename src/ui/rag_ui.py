# =============================================================================
#  rag_ui.py  —  Customer Support Management · Ticket Routing UI
#  Gradio 6.6.0  |  DistilBERT (no token_type_ids)  |  Groq LLaMA-3.3-70b
#  Models: Rarry/queue  |  Rarry/Priority  |  Rarry/RAG_Ticket_Trial
#
#  Setup:
#    pip install groq
#    Add GROQ_API_KEY=<your_key> to secrets.env
#    Get a free key at: https://console.groq.com
# =============================================================================

import os
import re
import json
import pickle
import traceback
import warnings
warnings.filterwarnings("ignore")

import torch
import faiss
import gradio as gr

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from groq import Groq
from collections import defaultdict
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
QUEUE_REPO_ID    = "Rarry/queue"
PRIORITY_REPO_ID = "Rarry/Priority"
RAG_REPO_ID      = "Rarry/RAG_Ticket_Trial"
EMBED_MODEL_ID   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"

load_dotenv("secrets.env")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — load everything once
# ─────────────────────────────────────────────────────────────────────────────
print(f"[startup] Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(QUEUE_REPO_ID, token=HF_TOKEN)

queue_model = AutoModelForSequenceClassification.from_pretrained(
    QUEUE_REPO_ID, token=HF_TOKEN).to(DEVICE).eval()

priority_model = AutoModelForSequenceClassification.from_pretrained(
    PRIORITY_REPO_ID, token=HF_TOKEN).to(DEVICE).eval()

with open(hf_hub_download(QUEUE_REPO_ID,    "queue_encoder.pkl",    token=HF_TOKEN), "rb") as f:
    queue_encoder = pickle.load(f)
with open(hf_hub_download(PRIORITY_REPO_ID, "priority_encoder.pkl", token=HF_TOKEN), "rb") as f:
    priority_encoder = pickle.load(f)

queue_id2label    = {i: l for i, l in enumerate(queue_encoder.classes_)}
priority_id2label = {i: l for i, l in enumerate(priority_encoder.classes_)}

print("[startup] Classifiers loaded.")
print("  Queue labels   :", queue_id2label)
print("  Priority labels:", priority_id2label)

index = faiss.read_index(
    hf_hub_download(RAG_REPO_ID, "rag_index.faiss", token=HF_TOKEN))
with open(hf_hub_download(RAG_REPO_ID, "rag_metadata.pkl", token=HF_TOKEN), "rb") as f:
    _meta = pickle.load(f)

corpus_texts      = _meta["texts"]
corpus_queues     = _meta["queues"]
corpus_priorities = _meta["priorities"]
print(f"[startup] FAISS: {index.ntotal} vectors | {len(corpus_texts)} metadata rows")

embedder = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
print("[startup] Embedder ready.")

llm_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
print(f"[startup] LLM: {'Groq (' + GROQ_MODEL + ') ready' if llm_client else 'no GROQ_API_KEY — LLM step skipped'}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def llm_preprocess(raw_text: str) -> dict:
    """Groq LLaMA-3.3 cleans and structures the ticket. Gracefully skipped if no key."""
    fallback = {
        "structured_body": raw_text,
        "subject":         "Support Ticket",
        "urgency_signals": [],
        "tech_keywords":   [],
        "explanation":     "LLM preprocessing skipped — no GROQ_API_KEY set.",
    }
    if llm_client is None:
        return fallback

    system_prompt = (
        "You are a customer-support triage assistant. "
        "Given a raw support ticket, return ONLY valid JSON with these keys:\n"
        "  structured_body  : cleaned, professional rewrite of the ticket\n"
        "  subject          : concise 6-10 word subject line\n"
        "  urgency_signals  : list of urgency phrases found\n"
        "  tech_keywords    : list of technical terms found\n"
        "  explanation      : one sentence explaining the likely department\n"
        "Return nothing outside the JSON object."
    )
    try:
        resp = llm_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": raw_text},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$",          "", content)
        d = json.loads(content)
        return {
            "structured_body": d.get("structured_body", raw_text),
            "subject":         d.get("subject",         "Support Ticket"),
            "urgency_signals": d.get("urgency_signals", []),
            "tech_keywords":   d.get("tech_keywords",   []),
            "explanation":     d.get("explanation",     ""),
        }
    except Exception as e:
        print(f"[llm_preprocess] Error (falling back to raw text): {e}")
        traceback.print_exc()
        fallback["explanation"] = f"LLM error: {e}"
        return fallback


def retrieve_similar_tickets(query_text: str, k: int = 5) -> list:
    vec = embedder.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).reshape(1, -1)
    scores, indices = index.search(vec, k)
    return [
        {
            "text":       corpus_texts[idx][:300],
            "queue":      corpus_queues[idx],
            "priority":   corpus_priorities[idx],
            "similarity": round(float(s), 4),
        }
        for s, idx in zip(scores[0], indices[0])
    ]


def transformer_predict(text: str) -> dict:
    """
    Run both classifiers.
    DistilBERT does not use token_type_ids — must pop before forward pass.
    (From V4.ipynb cell 44: inputs.pop('token_type_ids', None))
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)

    inputs.pop("token_type_ids", None)   # DistilBERT fix

    with torch.no_grad():
        q_logits = queue_model(**inputs).logits
        p_logits = priority_model(**inputs).logits

    q = torch.softmax(q_logits, dim=-1).cpu().numpy()[0]
    p = torch.softmax(p_logits, dim=-1).cpu().numpy()[0]

    return {
        "queue_pred":     queue_id2label[int(q.argmax())],
        "queue_conf":     float(q.max()),
        "queue_probs":    {queue_id2label[i]: round(float(v), 4) for i, v in enumerate(q)},
        "priority_pred":  priority_id2label[int(p.argmax())],
        "priority_conf":  float(p.max()),
        "priority_probs": {priority_id2label[i]: round(float(v), 4) for i, v in enumerate(p)},
    }


def ensemble_predict(text: str, retrieved: list, base_alpha: float = 0.7) -> dict:
    """
    Dynamic-alpha ensemble (V4.ipynb cell 46):
      alpha = base_alpha + (transformer_conf - 0.5) * 0.4, clamped [0.4, 0.95]
    """
    t = transformer_predict(text)

    rag_scores = defaultdict(float)
    for r in retrieved:
        rag_scores[r["queue"]] += r["similarity"]
    total     = sum(rag_scores.values()) or 1.0
    rag_probs = {label: score / total for label, score in rag_scores.items()}

    alpha = base_alpha + (t["queue_conf"] - 0.5) * 0.4
    alpha = max(0.4, min(0.95, alpha))

    combined = {}
    for i, label in queue_id2label.items():
        t_prob   = float(t["queue_probs"][label])
        rag_prob = rag_probs.get(label, 0.0)
        combined[label] = alpha * t_prob + (1 - alpha) * rag_prob

    final_queue = max(combined, key=combined.get)
    rag_queue   = max(rag_probs, key=rag_probs.get) if rag_probs else final_queue

    pv = defaultdict(float)
    for r in retrieved:
        pv[r["priority"]] += r["similarity"]
    rag_priority = max(pv, key=pv.get) if pv else t["priority_pred"]

    return {
        "final_queue":    final_queue,
        "final_priority": t["priority_pred"],
        "transformer":    t,
        "rag_queue":      rag_queue,
        "rag_priority":   rag_priority,
        "agreement":      final_queue == rag_queue,
        "alpha_used":     round(alpha, 3),
        "all_scores":     dict(sorted(combined.items(), key=lambda x: -x[1])),
    }


def full_pipeline(raw_text: str) -> dict:
    print(f"\n[pipeline] ── New ticket ──────────────────────────────────────")
    print(f"[pipeline] Input: {raw_text[:100]!r}")

    llm_result = llm_preprocess(raw_text)
    clean_text = llm_result["structured_body"]
    print(f"[pipeline] LLM done. Subject: {llm_result['subject']!r}")

    retrieved = retrieve_similar_tickets(clean_text, k=5)
    print(f"[pipeline] RAG done. Top: {retrieved[0]['queue']} (sim={retrieved[0]['similarity']})")

    ens = ensemble_predict(clean_text, retrieved)
    print(
        f"[pipeline] Ensemble: final={ens['final_queue']} | "
        f"transformer={ens['transformer']['queue_pred']} {ens['transformer']['queue_conf']:.1%} | "
        f"rag={ens['rag_queue']} | agree={ens['agreement']} | alpha={ens['alpha_used']}"
    )
    print(f"[pipeline] Priority: {ens['final_priority']}")

    return {
        "raw_text":  raw_text,
        "llm":       llm_result,
        "retrieved": retrieved,
        "ensemble":  ens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PE = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def render_routing_card(r: dict) -> str:
    ens   = r["ensemble"]
    q, p  = ens["final_queue"], ens["final_priority"]
    qc    = ens["transformer"]["queue_conf"]
    alpha = ens["alpha_used"]
    ag    = "✅ RAG agrees" if ens["agreement"] else f"⚠️ RAG suggested **{ens['rag_queue']}**"
    return "\n".join([
        "## 🎯 Routing Decision", "",
        "| Field | Value |", "|---|---|",
        f"| **Department** | {q} |",
        f"| **Priority** | {PE.get(p.lower(),'⚪')} {p.capitalize()} |",
        f"| **Transformer Confidence** | `{qc:.1%}` |",
        f"| **Ensemble α** | `{alpha}` (higher = more transformer weight) |",
        f"| **RAG Validation** | {ag} |",
    ])


def render_dept_bars(r: dict) -> str:
    scores = r["ensemble"]["all_scores"]
    top    = next(iter(scores.values()))
    fq     = r["ensemble"]["final_queue"]
    lines  = ["## 📊 Department Confidence (Ensemble)", ""]
    for label, prob in scores.items():
        pct = prob / max(top, 1e-9)
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        m   = "  ◀ predicted" if label == fq else ""
        lines.append(f"`{label:<34}` `{bar}` `{prob:.1%}`{m}")
        lines.append("")
    return "\n".join(lines)


def render_prio_bars(r: dict) -> str:
    probs = r["ensemble"]["transformer"]["priority_probs"]
    sp    = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    fp    = r["ensemble"]["final_priority"]
    lines = ["## 🚦 Priority Confidence", ""]
    for label, prob in sp:
        bar = "█" * int(prob * 28) + "░" * (28 - int(prob * 28))
        m   = "  ◀ predicted" if label == fp else ""
        lines.append(f"{PE.get(label.lower(),'⚪')} `{label:<8}` `{bar}` `{prob:.1%}`{m}")
        lines.append("")
    return "\n".join(lines)


def render_llm(r: dict) -> str:
    llm = r["llm"]
    urg = ", ".join(llm["urgency_signals"]) if llm["urgency_signals"] else "_none detected_"
    kw  = ", ".join(llm["tech_keywords"])   if llm["tech_keywords"]   else "_none detected_"
    return "\n".join([
        "## 🤖 LLM Pre-Processing", "",
        f"**Subject:** {llm['subject']}", "",
        f"**Urgency signals:** {urg}", "",
        f"**Technical keywords:** {kw}", "",
        "**Cleaned ticket body:**", "",
        f"> {llm['structured_body']}", "",
        "---", "",
        f"💬 _{llm['explanation'] or 'n/a'}_",
    ])


def render_rag(r: dict) -> str:
    lines = ["## 🔍 RAG — Top 5 Similar Tickets", ""]
    for i, rt in enumerate(r["retrieved"], 1):
        bar = "█" * int(rt["similarity"] * 20)
        lines += [
            f"**{i}. {rt['queue']}** · _{rt['priority'].capitalize()} priority_"
            f" · sim `{rt['similarity']:.4f}` `{bar}`",
            "",
            f"> {rt['text']}...",
            "",
        ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO EVENT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

EMPTY = "_(Submit a ticket above to see results)_"


def process_ticket(user_message: str, history: list):
    print(f"\n[UI] process_ticket called | msg={user_message!r}")

    if not user_message or not user_message.strip():
        print("[UI] Empty input — skipping.")
        return history, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY

    history = history or []

    try:
        result = full_pipeline(user_message.strip())
    except Exception as e:
        print(f"[UI] Pipeline exception: {e}")
        traceback.print_exc()
        err_md = f"**⚠️ Pipeline error:** {e}"
        history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": err_md},
        ]
        return history, err_md, err_md, err_md, err_md, err_md

    ens  = result["ensemble"]
    q, p = ens["final_queue"], ens["final_priority"]
    qc   = ens["transformer"]["queue_conf"]
    ag   = (
        "✅ RAG agrees."
        if ens["agreement"]
        else f"⚠️ RAG suggested **{ens['rag_queue']}** — ensemble overrides."
    )
    expl = result["llm"]["explanation"]

    reply = (
        "Ticket analysed and routed!\n\n"
        f"🏢 **Department:** {q}\n"
        f"{PE.get(p.lower(),'⚪')} **Priority:** {p.capitalize()}"
        f"  ·  Transformer confidence `{qc:.1%}`\n\n"
        f"{ag}"
        + (f"\n\n_{expl}_" if expl else "")
        + "\n\n_Full breakdown in the panels on the right →_"
    )

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    print("[UI] process_ticket done ✓")

    return (
        history,
        render_routing_card(result),
        render_dept_bars(result),
        render_prio_bars(result),
        render_llm(result),
        render_rag(result),
    )


def clear_all():
    print("[UI] clear_all called.")
    return [], EMPTY, EMPTY, EMPTY, EMPTY, EMPTY


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { font-family: 'Inter', 'Segoe UI', sans-serif; }

#header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 12px; padding: 20px 28px;
    margin-bottom: 8px; color: #fff;
}
#header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
#header p  { margin: 4px 0 0; font-size: 0.88rem; opacity: 0.85; }

#dept-bars .prose,
#prio-bars .prose {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
}

#llm-panel .prose blockquote  { border-left: 4px solid #2d6a9f; padding-left: 10px; color: #dde; }
#rag-panel .prose blockquote  { border-left: 4px solid #10b981; padding-left: 10px; color: #dde; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE TICKETS
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "My internet isn't working and I can't log in since yesterday — urgent deadline today.",
    "I think I was charged twice for my subscription last month. Can someone look into this?",
    "We need to update payroll records for three new hires who started this week.",
    "The checkout page keeps crashing when customers try to pay with Visa. We're losing sales.",
    "Can you tell me your standard service hours and how to escalate a complaint?",
    "Our Salesforce CRM has been down since 9am, the entire sales team is blocked.",
]

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Customer Support Ticket Router") as demo:

    gr.HTML("""
    <div id="header">
      <h1>🎫 Customer Support Ticket Router</h1>
      <p>Fine-tuned DistilBERT &nbsp;·&nbsp; FAISS RAG retrieval
         &nbsp;·&nbsp; Dynamic-α Ensemble &nbsp;·&nbsp; Groq LLaMA-3.3 preprocessing</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: chat + input ────────────────────────────────────────────────
        with gr.Column(scale=5):

            chatbot = gr.Chatbot(
                label="Ticket Conversation",
                height=440,
                show_label=True,
            )

            msg_input = gr.Textbox(
                placeholder="Describe your issue in plain language…",
                label="Your ticket",
                lines=3,
                max_lines=8,
                show_label=True,
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Submit Ticket", variant="primary",  scale=3)
                clear_btn  = gr.Button("🗑️ Clear",          variant="secondary", scale=1)

            gr.Examples(
                examples=EXAMPLES,
                inputs=msg_input,
                label="💡 Quick examples — click to load",
                examples_per_page=6,
            )

        # ── RIGHT: results panels ─────────────────────────────────────────────
        with gr.Column(scale=6):

            routing_card = gr.Markdown(value=EMPTY)

            with gr.Tabs():
                with gr.Tab("📊 Department Confidence"):
                    dept_bars = gr.Markdown(value=EMPTY, elem_id="dept-bars")
                with gr.Tab("🚦 Priority Confidence"):
                    prio_bars = gr.Markdown(value=EMPTY, elem_id="prio-bars")
                with gr.Tab("🤖 LLM Pre-Processing"):
                    llm_panel = gr.Markdown(value=EMPTY, elem_id="llm-panel")
                with gr.Tab("🔍 RAG Retrieved Tickets"):
                    rag_panel = gr.Markdown(value=EMPTY, elem_id="rag-panel")

    OUTS = [chatbot, routing_card, dept_bars, prio_bars, llm_panel, rag_panel]

    submit_btn.click(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=None, outputs=msg_input)

    msg_input.submit(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=None, outputs=msg_input)

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=None, outputs=msg_input)

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, share=True)
