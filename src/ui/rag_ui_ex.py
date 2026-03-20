# =============================================================================
#  rag_ui.py  —  Customer Support Management · Ticket Routing UI
#  Gradio 6.6.0  |  DistilBERT (no token_type_ids)  |  Groq LLaMA-3.3-70b
#  Models: Rarry/queue  |  Rarry/Priority  |  Rarry/RAG_Ticket_Trial
#
#  Setup:
#    pip install groq
#    (NEW) pip install captum plotly
#    Add GROQ_API_KEY=<your_key> to secrets.env (optional)
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
from collections import defaultdict, Counter
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



def _tokenize_for_model(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)
    inputs.pop("token_type_ids", None)   # DistilBERT fix
    return inputs



def transformer_predict(text: str) -> dict:
    """
    Run both classifiers.
    DistilBERT does not use token_type_ids — must pop before forward pass.
    (From V4.ipynb cell 44: inputs.pop('token_type_ids', None))
    """
    inputs = _tokenize_for_model(text)


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
        "queue_logits":   q_logits.detach().cpu().numpy()[0].tolist(),
        "priority_logits": p_logits.detach().cpu().numpy()[0].tolist(),
    }



def ensemble_predict(text: str, retrieved: list, base_alpha: float = 0.7) -> dict:
    """
    Queue: dynamic-alpha ensemble + Fix 1 (RAG-strong reduces alpha) + Fix 2 (RAG veto).
    Priority: transformer+RAG ensemble (same as the last fix we added).
    """
    t = transformer_predict(text)

    # -------------------------
    # RAG vote for QUEUE
    # -------------------------
    rag_scores = defaultdict(float)
    for r in retrieved:
        rag_scores[r["queue"]] += r["similarity"]
    total_q = sum(rag_scores.values()) or 1.0
    rag_probs = {label: score / total_q for label, score in rag_scores.items()}

    rag_queue = max(rag_probs, key=rag_probs.get) if rag_probs else t["queue_pred"]
    rag_max = max(rag_probs.values()) if rag_probs else 0.0
    top_sim = max((r["similarity"] for r in retrieved), default=0.0)

    # -------------------------
    # Dynamic alpha for QUEUE (original)
    # -------------------------
    alpha = base_alpha + (t["queue_conf"] - 0.5) * 0.4
    alpha = max(0.4, min(0.95, alpha))

    # -------------------------
    # Fix 1: If RAG vote is very concentrated, trust RAG more (reduce alpha)
    # -------------------------
    rag_strength = max(0.0, min(1.0, (rag_max - 0.60) / 0.40))  # 0 when <=0.60, 1 when >=1.0
    alpha_min_when_rag_strong = 0.25
    alpha = (1 - rag_strength) * alpha + rag_strength * alpha_min_when_rag_strong

    # -------------------------
    # Combine QUEUE: transformer + RAG
    # -------------------------
    combined = {}
    for _, label in queue_id2label.items():
        t_prob = float(t["queue_probs"].get(label, 0.0))
        r_prob = float(rag_probs.get(label, 0.0))
        combined[label] = alpha * t_prob + (1 - alpha) * r_prob

    sorted_combined = sorted(combined.items(), key=lambda x: -x[1])

    # Default final/runners from ensemble
    final_queue = sorted_combined[0][0] if sorted_combined else t["queue_pred"]
    runner_up_queue = sorted_combined[1][0] if len(sorted_combined) > 1 else final_queue

    # -------------------------
    # Fix 2: RAG veto rule (unanimous RAG + strong similarity beats medium transformer)
    # -------------------------
    veto_triggered = False
    if (
        rag_probs
        and rag_max >= 0.95
        and top_sim >= 0.45
        and rag_queue != t["queue_pred"]
        and t["queue_conf"] < 0.85
    ):
        veto_triggered = True
        final_queue = rag_queue
        runner_up_queue = t["queue_pred"]

    # -------------------------
    # RAG vote for PRIORITY + priority ensemble
    # -------------------------
    pv = defaultdict(float)
    for r in retrieved:
        pv[r["priority"]] += r["similarity"]

    total_p = sum(pv.values()) or 1.0
    rag_priority_probs = {label: score / total_p for label, score in pv.items()}
    rag_priority = max(pv, key=pv.get) if pv else t["priority_pred"]

    alpha_p = base_alpha + (t["priority_conf"] - 0.5) * 0.4
    alpha_p = max(0.4, min(0.95, alpha_p))

    combined_p = {}
    for _, label in priority_id2label.items():
        t_prob = float(t["priority_probs"].get(label, 0.0))
        r_prob = float(rag_priority_probs.get(label, 0.0))
        combined_p[label] = alpha_p * t_prob + (1 - alpha_p) * r_prob

    final_priority = max(combined_p, key=combined_p.get) if combined_p else t["priority_pred"]

    return {
        "final_queue":     final_queue,
        "runner_up_queue": runner_up_queue,
        "final_priority":  final_priority,

        "transformer":     t,

        "rag_queue":       rag_queue,
        "rag_priority":    rag_priority,

        "rag_probs":       dict(sorted(rag_probs.items(), key=lambda x: -x[1])),
        "rag_priority_probs": dict(sorted(rag_priority_probs.items(), key=lambda x: -x[1])),

        "agreement":       final_queue == rag_queue,

        "alpha_used":      round(alpha, 3),
        "priority_alpha_used": round(alpha_p, 3),

        "all_scores":      dict(sorted_combined),
        "priority_scores": dict(sorted(combined_p.items(), key=lambda x: -x[1])),

        # Debug fields (optional; safe to keep even if UI ignores them)
        "rag_max":         round(float(rag_max), 4),
        "top_sim":         round(float(top_sim), 4),
        "veto_triggered":  bool(veto_triggered),
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
        "raw_text":   raw_text,
        "clean_text": clean_text,
        "llm":        llm_result,
        "retrieved":  retrieved,
        "ensemble":   ens,
    }



# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY — Integrated Gradients & Occlusion-style perturbations
# ─────────────────────────────────────────────────────────────────────────────


STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","of","to","in","on","for","from",
    "with","without","at","by","as","is","are","was","were","be","been","being","it","this","that","these",
    "those","we","i","you","they","he","she","our","their","my","your","me","us","them","can","could","should",
    "would","please","help","need","needed","urgent","thanks","thank"
}


def _get_embedding_layer(model):
    # DistilBERT: model.distilbert.embeddings
    for base in ("distilbert", "bert", "roberta", "deberta", "electra", "albert"):
        if hasattr(model, base):
            m = getattr(model, base)
            if hasattr(m, "embeddings"):
                return m.embeddings
    return None


def _merge_wordpieces(tokens, scores):
    merged = []
    cur_tok, cur_score = "", 0.0
    for t, s in zip(tokens, scores):
        if t in ("[CLS]","[SEP]","[PAD]"):
            continue
        if t.startswith("##"):
            cur_tok += t[2:]
            cur_score += float(s)
        else:
            if cur_tok:
                merged.append((cur_tok, cur_score))
            cur_tok, cur_score = t, float(s)
    if cur_tok:
        merged.append((cur_tok, cur_score))
    return merged


def _normalize_signed(vals, eps=1e-9):
    m = max(abs(v) for v in vals) if vals else 1.0
    m = max(m, eps)
    return [float(v) / m for v in vals]


def _rgba_for_score(x):
    # x in [-1,1], red = positive support, blue = negative support
    x = max(-1.0, min(1.0, float(x)))
    if x >= 0:
        r, g, b = 255, int(255 * (1 - 0.55 * x)), int(255 * (1 - 0.55 * x))
        a = 0.10 + 0.35 * x
    else:
        x = -x
        r, g, b = int(255 * (1 - 0.55 * x)), int(255 * (1 - 0.55 * x)), 255
        a = 0.10 + 0.35 * x
    return f"rgba({r},{g},{b},{a:.3f})"


def _render_token_heat(tokens_scores, title, subtitle=None, max_tokens=80):
    items = tokens_scores[:max_tokens]
    chips = []
    for tok, sc in items:
        bg = _rgba_for_score(sc)
        safe = tok.replace("<","&lt;").replace(">","&gt;")
        chips.append(
            f"<span class='chip' style='background:{bg}' title='{sc:+.3f}'>{safe}</span>"
        )
    sub = f"<div class='sub'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='heatbox'>"
        f"<div class='htitle'>{title}</div>"
        f"{sub}"
        f"<div class='chips'>{''.join(chips)}</div>"
        f"</div>"
    )


def _try_import_captum():
    try:
        from captum.attr import LayerIntegratedGradients
        return LayerIntegratedGradients
    except Exception:
        return None


def integrated_gradients_explain(text: str, model, target_idx: int):
    LayerIntegratedGradients = _try_import_captum()
    if LayerIntegratedGradients is None:
        return None, "Captum is not installed. Run: pip install captum"


    emb_layer = _get_embedding_layer(model)
    if emb_layer is None:
        return None, "Could not locate the transformer embedding layer for attribution."


    inputs = _tokenize_for_model(text)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]


    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id


    baseline_ids = input_ids.clone()
    special = torch.zeros_like(baseline_ids, dtype=torch.bool)
    if cls_id is not None:
        special |= (baseline_ids == cls_id)
    if sep_id is not None:
        special |= (baseline_ids == sep_id)
    baseline_ids[~special] = pad_id


    def forward_func(input_ids_, attention_mask_):
        out = model(input_ids=input_ids_, attention_mask=attention_mask_)
        return out.logits


    lig = LayerIntegratedGradients(forward_func, emb_layer)


    try:
        attr = lig.attribute(
            inputs=(input_ids, attention_mask),
            baselines=(baseline_ids, attention_mask),
            target=target_idx,
            n_steps=16,
        )
        if isinstance(attr, (tuple, list)):
            attr = attr[0]
        token_attr = attr.sum(dim=-1).detach().cpu().squeeze(0)  # [T]
        toks = tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().squeeze(0).tolist())
        scores = token_attr.tolist()
        merged = _merge_wordpieces(toks, scores)
        merged_norm = [(t, s) for t, s in merged]
        merged_norm = [(t, s) for t, s in zip([t for t,_ in merged_norm], _normalize_signed([s for _,s in merged_norm]))]
        merged_abs_sorted = sorted(merged_norm, key=lambda x: -abs(x[1]))
        return {"sorted": merged_abs_sorted, "ordered": merged_norm}, None
    except Exception as e:
        return None, f"Attribution error: {e}"


def _simple_words(text: str):
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-']{2,}", text.lower())


def occlusion_drop_report(text: str, top_words: list, base_pred: dict) -> list:
    reports = []
    base_q = base_pred["queue_conf"]
    base_p = base_pred["priority_conf"]
    for w in top_words:
        occluded = re.sub(rf"\b{re.escape(w)}\b", " ", text, flags=re.IGNORECASE)
        occluded = re.sub(r"\s+", " ", occluded).strip()
        if not occluded:
            continue
        pred2 = transformer_predict(occluded)
        reports.append({
            "word": w,
            "queue_drop": max(0.0, base_q - pred2["queue_conf"]),
            "prio_drop":  max(0.0, base_p - pred2["priority_conf"]),
            "new_queue_conf": pred2["queue_conf"],
            "new_prio_conf": pred2["priority_conf"],
        })
    return reports


def _pick_top_support_words(attr_sorted, k=3):
    out = []
    for tok, sc in attr_sorted:
        w = tok.lower().strip("`\"'.,:;!?()[]{}")
        if not w or w in STOPWORDS:
            continue
        if not re.search(r"[a-z]", w):
            continue
        if sc <= 0:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= k:
            break
    return out


def extract_competitor_keywords(retrieved: list, competitor: str, k=8):
    texts = [r["text"] for r in retrieved if r["queue"] == competitor]
    if not texts:
        return []
    words = []
    for t in texts:
        words += _simple_words(t)
    counts = Counter([w for w in words if w not in STOPWORDS and len(w) >= 4])
    return [w for w,_ in counts.most_common(k)]



# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY — Radar charts (Plotly) for Dept + Priority
# ─────────────────────────────────────────────────────────────────────────────


def _try_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except Exception:
        return None


def build_dept_radar_fig(r: dict, top_k: int = 7):
    go = _try_import_plotly()
    if go is None:
        return None


    ens = r["ensemble"]
    scores = ens["all_scores"]
    labels = list(scores.keys())[:max(3, min(top_k, len(scores)))]


    comb   = [scores[l] for l in labels]
    trans  = [ens["transformer"]["queue_probs"].get(l, 0.0) for l in labels]
    ragp   = [ens["rag_probs"].get(l, 0.0) for l in labels]


    labels_c = labels + [labels[0]]
    comb_c   = comb   + [comb[0]]
    trans_c  = trans  + [trans[0]]
    rag_c    = ragp   + [ragp[0]]


    vmax = max(comb_c + trans_c + rag_c) if labels else 1.0


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=trans_c, theta=labels_c, fill="toself", name="Transformer"))
    fig.add_trace(go.Scatterpolar(r=rag_c,   theta=labels_c, fill="toself", name="RAG vote"))
    fig.add_trace(go.Scatterpolar(r=comb_c,  theta=labels_c, fill="toself", name="Ensemble"))


    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, vmax * 1.05])),
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
        title="Department radar (top classes)"
    )
    return fig


def build_priority_radar_fig(r: dict):
    go = _try_import_plotly()
    if go is None:
        return None


    ens = r["ensemble"]
    probs = ens["transformer"]["priority_probs"]
    labels = list(sorted(probs.keys()))
    trans  = [probs[l] for l in labels]


    pv = defaultdict(float)
    for rt in r["retrieved"]:
        pv[rt["priority"]] += rt["similarity"]
    total = sum(pv.values()) or 1.0
    ragp = [pv.get(l, 0.0) / total for l in labels]


    comb = trans[:]


    labels_c = labels + [labels[0]]
    trans_c  = trans  + [trans[0]]
    rag_c    = ragp   + [ragp[0]]
    comb_c   = comb   + [comb[0]]


    vmax = max(trans_c + rag_c + comb_c) if labels else 1.0


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=trans_c, theta=labels_c, fill="toself", name="Transformer"))
    fig.add_trace(go.Scatterpolar(r=rag_c,   theta=labels_c, fill="toself", name="RAG vote"))
    fig.add_trace(go.Scatterpolar(r=comb_c,  theta=labels_c, fill="toself", name="Final"))


    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, vmax * 1.05])),
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        title="Priority radar"
    )
    return fig



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


def render_rag_vote_breakdown(r: dict) -> str:
    rag_probs = r["ensemble"]["rag_probs"]
    if not rag_probs:
        return "## 🧾 RAG vote breakdown\n\n_(No RAG votes available.)_"
    lines = ["## 🧾 RAG vote breakdown (similarity-weighted)", ""]
    top = next(iter(rag_probs.values()))
    for label, prob in list(rag_probs.items())[:10]:
        pct = prob / max(top, 1e-9)
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        lines.append(f"`{label:<34}` `{bar}` `{prob:.1%}`")
        lines.append("")
    return "\n".join(lines)


def render_decision_trace(r: dict) -> str:
    ens = r["ensemble"]
    t   = ens["transformer"]


    def topk(d, k=3):
        return sorted(d.items(), key=lambda x: -x[1])[:k]


    t_top = topk(t["queue_probs"], 3)
    r_top = topk(ens["rag_probs"], 3) if ens["rag_probs"] else []
    e_top = list(ens["all_scores"].items())[:3]


    p_top = topk(t["priority_probs"], 3)


    lines = [
        "## 🧾 Decision trace",
        "",
        f"**Input used:**",
        "",
        f"> {r['clean_text']}",
        "",
        "| Stage | Key fields |",
        "|---|---|",
        f"| α (dynamic) | `{ens['alpha_used']}` |",
        f"| Transformer top-3 (dept) | " + ", ".join([f"`{k}` {v:.1%}" for k,v in t_top]) + " |",
        f"| RAG vote top-3 (dept) | " + (", ".join([f"`{k}` {v:.1%}" for k,v in r_top]) if r_top else "_n/a_") + " |",
        f"| Ensemble top-3 (dept) | " + ", ".join([f"`{k}` {v:.1%}" for k,v in e_top]) + " |",
        f"| Transformer top-3 (priority) | " + ", ".join([f"`{k}` {v:.1%}" for k,v in p_top]) + " |",
    ]
    return "\n".join(lines)


def render_what_would_change(r: dict, keyphrases: dict) -> str:
    ens = r["ensemble"]
    predicted = ens["final_queue"]
    runner = ens["runner_up_queue"]


    all_scores = list(ens["all_scores"].items())
    s_pred = all_scores[0][1] if all_scores else 0.0
    s_run  = all_scores[1][1] if len(all_scores) > 1 else 0.0
    margin = s_pred - s_run


    comp_kw = extract_competitor_keywords(r["retrieved"], runner, k=8)


    sugg_remove = keyphrases.get("support_words", [])
    sugg_add = comp_kw[:5]


    lines = [
        "## 🔁 What would change it? (heuristic)",
        "",
        f"**Current pick:** `{predicted}`",
        f"**Nearest competitor:** `{runner}`",
        f"**Score margin (ensemble):** `{margin:.3f}`",
        "",
        "### If you removed these words",
        "",
        ("Try removing/avoiding: " + ", ".join([f"`{w}`" for w in sugg_remove]) if sugg_remove else "_No strong support-words identified._"),
        "",
        "### If you added more detail for the competitor",
        "",
        ("Terms that appear in competitor-like retrieved tickets: " + ", ".join([f"`{w}`" for w in sugg_add]) if sugg_add else
         "_No competitor-labelled exemplars in the top-5 retrieval. Add details that clearly match the competitor department’s scope (systems, processes, products, error messages, accounts, etc.)._"),
        "",
        "_Note: this is a quick sensitivity probe (no LLM); it’s meant to show what the model is relying on, not prescribe how users must write tickets._"
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
        return history, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, "", "", EMPTY, EMPTY, None, None


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
        return history, err_md, err_md, err_md, err_md, err_md, err_md, err_md, "", "", err_md, err_md, None, None


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


    clean_text = result["clean_text"]
    tpred = result["ensemble"]["transformer"]


    keyphrases_state = {"support_words": []}


    q_idx = None
    for i, lab in queue_id2label.items():
        if lab == tpred["queue_pred"]:
            q_idx = i
            break
    queue_html = ""
    queue_err = None
    if q_idx is not None:
        q_attr, queue_err = integrated_gradients_explain(clean_text, queue_model, q_idx)
        if q_attr:
            keyphrases_state["support_words"] = _pick_top_support_words(q_attr["sorted"], k=3)
            ordered = q_attr["ordered"]
            queue_html = _render_token_heat(
                ordered,
                title=f"Department attribution (Transformer → {tpred['queue_pred']})",
                subtitle="Red supports the predicted class; blue pushes against it (Integrated Gradients)."
            )
        else:
            queue_html = _render_token_heat([], "Department attribution unavailable", subtitle=queue_err or "Unknown error")
    else:
        queue_html = _render_token_heat([], "Department attribution unavailable", subtitle="Could not resolve predicted class index.")


    p_idx = None
    for i, lab in priority_id2label.items():
        if lab == tpred["priority_pred"]:
            p_idx = i
            break
    prio_html = ""
    prio_err = None
    if p_idx is not None:
        p_attr, prio_err = integrated_gradients_explain(clean_text, priority_model, p_idx)
        if p_attr:
            ordered = p_attr["ordered"]
            prio_html = _render_token_heat(
                ordered,
                title=f"Priority attribution (Transformer → {tpred['priority_pred']})",
                subtitle="Same colour logic as above."
            )
        else:
            prio_html = _render_token_heat([], "Priority attribution unavailable", subtitle=prio_err or "Unknown error")
    else:
        prio_html = _render_token_heat([], "Priority attribution unavailable", subtitle="Could not resolve predicted class index.")


    occl_md = "## 🧪 Occlusion check (remove top words)\n\n"
    if keyphrases_state["support_words"]:
        reports = occlusion_drop_report(clean_text, keyphrases_state["support_words"], tpred)
        if reports:
            occl_md += "| Removed word | Dept conf drop | New dept conf | Priority conf drop | New priority conf |\n|---|---:|---:|---:|---:|\n"
            for rep in reports:
                occl_md += (
                    f"| `{rep['word']}` | {rep['queue_drop']:.1%} | {rep['new_queue_conf']:.1%} "
                    f"| {rep['prio_drop']:.1%} | {rep['new_prio_conf']:.1%} |\n"
                )
        else:
            occl_md += "_No occlusion results (text became empty or perturbation failed)._"
    else:
        occl_md += "_No strong support-words found (or attribution unavailable). Install Captum to enable._"


    change_md = render_what_would_change(result, keyphrases_state)
    trace_md = render_decision_trace(result)


    radar_dept = build_dept_radar_fig(result, top_k=7)
    radar_prio = build_priority_radar_fig(result)


    return (
        history,
        render_routing_card(result),
        render_dept_bars(result),
        render_prio_bars(result),
        render_llm(result),
        render_rag(result),
        render_rag_vote_breakdown(result),
        trace_md,
        queue_html,
        prio_html,
        occl_md,
        change_md,
        radar_dept,
        radar_prio,
    )



def clear_all():
    print("[UI] clear_all called.")
    return [], EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, "", "", EMPTY, EMPTY, None, None



# ─────────────────────────────────────────────────────────────────────────────
# UI — Custom horizontal tab strip (Chrome-like) + panel switching
# ─────────────────────────────────────────────────────────────────────────────

TAB_KEYS = ["dept", "prio", "radar", "keyphr", "change", "trace", "llm", "rag"]

def set_active_panel(selected: str):
    if selected not in TAB_KEYS:
        selected = "dept"

    def b(sel_key):
        return gr.update(variant="primary" if selected == sel_key else "secondary")

    def v(sel_key):
        return gr.update(visible=(selected == sel_key))

    return (
        selected,
        b("dept"), b("prio"), b("radar"), b("keyphr"), b("change"), b("trace"), b("llm"), b("rag"),
        v("dept"), v("prio"), v("radar"), v("keyphr"), v("change"), v("trace"), v("llm"), v("rag"),
    )



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


.heatbox { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
           border-radius: 12px; padding: 14px 14px; }
.htitle { font-weight: 700; margin-bottom: 6px; }
.sub { opacity: 0.8; font-size: 0.85rem; margin-bottom: 10px; }
.chips { line-height: 2.3; }
.chip  { display: inline-block; padding: 2px 8px; border-radius: 999px;
         margin: 3px 6px 3px 0; border: 1px solid rgba(255,255,255,0.08);
         font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
         font-size: 0.86rem; }


/* ─────────────────────────────────────────────────────────────────────────────
   UI TWEAK — Horizontal "Chrome-like" tab strip (scrollable)
   ───────────────────────────────────────────────────────────────────────────── */

#results-tabstrip {
    display: flex;
    flex-wrap: nowrap;
    gap: 8px;
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
}

#results-tabstrip::-webkit-scrollbar { height: 8px; }
#results-tabstrip::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.18); border-radius: 99px; }
#results-tabstrip::-webkit-scrollbar-track { background: rgba(255,255,255,0.06); border-radius: 99px; }

#results-tabstrip button {
    flex: 0 0 auto;
    min-width: max-content;
    border-radius: 12px !important;
    padding: 8px 12px !important;
}
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
         &nbsp;·&nbsp; Dynamic-α Ensemble &nbsp;·&nbsp; Optional Groq LLaMA-3.3 preprocessing
         &nbsp;·&nbsp; Explainability (IG + Occlusion + Trace + Radar)</p>
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

            # ─────────────────────────────────────────────────────────────────
            # CUSTOM TAB STRIP (horizontal scroll, no overflow menu)
            # ─────────────────────────────────────────────────────────────────
            active_panel = gr.State("dept")

            with gr.Row(elem_id="results-tabstrip"):
                btn_dept   = gr.Button("📊 Department", variant="primary")
                btn_prio   = gr.Button("🚦 Priority", variant="secondary")
                btn_radar  = gr.Button("🕸️ Radar", variant="secondary")
                btn_key    = gr.Button("🧩 Key phrases", variant="secondary")
                btn_change = gr.Button("🔁 What would change it?", variant="secondary")
                btn_trace  = gr.Button("🧾 Decision trace", variant="secondary")
                btn_llm    = gr.Button("🤖 LLM", variant="secondary")
                btn_rag    = gr.Button("🔍 RAG", variant="secondary")

            # Panels (only one visible at a time)
            panel_dept   = gr.Column(visible=True)
            panel_prio   = gr.Column(visible=False)
            panel_radar  = gr.Column(visible=False)
            panel_key    = gr.Column(visible=False)
            panel_change = gr.Column(visible=False)
            panel_trace  = gr.Column(visible=False)
            panel_llm    = gr.Column(visible=False)
            panel_rag    = gr.Column(visible=False)


            with panel_dept:
                dept_bars = gr.Markdown(value=EMPTY, elem_id="dept-bars")

            with panel_prio:
                prio_bars = gr.Markdown(value=EMPTY, elem_id="prio-bars")

            with panel_radar:
                radar_dept = gr.Plot(value=None)
                radar_prio = gr.Plot(value=None)

            with panel_key:
                key_q_html = gr.HTML(value="")
                key_p_html = gr.HTML(value="")
                occl_md    = gr.Markdown(value=EMPTY)

            with panel_change:
                change_md = gr.Markdown(value=EMPTY)

            with panel_trace:
                trace_md = gr.Markdown(value=EMPTY)

            with panel_llm:
                llm_panel = gr.Markdown(value=EMPTY, elem_id="llm-panel")

            with panel_rag:
                rag_panel = gr.Markdown(value=EMPTY, elem_id="rag-panel")
                rag_vote  = gr.Markdown(value=EMPTY)

            tab_outputs = [
                active_panel,
                btn_dept, btn_prio, btn_radar, btn_key, btn_change, btn_trace, btn_llm, btn_rag,
                panel_dept, panel_prio, panel_radar, panel_key, panel_change, panel_trace, panel_llm, panel_rag
            ]

            btn_dept.click(fn=lambda: set_active_panel("dept"),   inputs=[], outputs=tab_outputs)
            btn_prio.click(fn=lambda: set_active_panel("prio"),   inputs=[], outputs=tab_outputs)
            btn_radar.click(fn=lambda: set_active_panel("radar"), inputs=[], outputs=tab_outputs)
            btn_key.click(fn=lambda: set_active_panel("keyphr"),  inputs=[], outputs=tab_outputs)
            btn_change.click(fn=lambda: set_active_panel("change"), inputs=[], outputs=tab_outputs)
            btn_trace.click(fn=lambda: set_active_panel("trace"), inputs=[], outputs=tab_outputs)
            btn_llm.click(fn=lambda: set_active_panel("llm"),     inputs=[], outputs=tab_outputs)
            btn_rag.click(fn=lambda: set_active_panel("rag"),     inputs=[], outputs=tab_outputs)


    OUTS = [
        chatbot,
        routing_card,
        dept_bars,
        prio_bars,
        llm_panel,
        rag_panel,
        rag_vote,
        trace_md,
        key_q_html,
        key_p_html,
        occl_md,
        change_md,
        radar_dept,
        radar_prio,
    ]


    submit_btn.click(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)


    msg_input.submit(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)


    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=OUTS,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)



# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, share=True)
