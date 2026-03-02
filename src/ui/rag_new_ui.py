import os
import re
import json
import pickle
import traceback
import warnings
from dataclasses import dataclass
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import faiss
import gradio as gr

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq
except Exception:
    Groq = None


# =========================
# App config (constants)
# =========================
APP_TITLE = "Customer Support Ticket Router"

# Multitask model (dept + priority)
MULTITASK_REPO_ID = "Nethra19/multitask-ticket-model"

# RAG repo (can be overridden by env var RAG_REPO_ID)
RAG_REPO_ID = "Rarry/RAG_Ticket_Trial"

# Embeddings + optional LLM
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

# Load env
load_dotenv("secrets.env")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

print(f"[startup] Device: {DEVICE}")


# =========================
# Multitask model definition
# =========================
class MultiTaskModel(nn.Module):
    """
    DistilBERT encoder + two linear heads.
    forward() returns {"logits": (queue_logits, priority_logits)}.
    """

    def __init__(self, repo_id: str, num_queue_labels: int, num_priority_labels: int, hf_token: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(repo_id, token=hf_token)
        hidden_size = self.encoder.config.hidden_size
        self.queue_classifier = nn.Linear(hidden_size, num_queue_labels)
        self.priority_classifier = nn.Linear(hidden_size, num_priority_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]
        queue_logits = self.queue_classifier(pooled_output)
        priority_logits = self.priority_classifier(pooled_output)
        return {"logits": (queue_logits, priority_logits)}


def load_label_encoder(path: str):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# =========================
# Load multitask tokenizer + model + heads + encoders
# =========================
print("[startup] Loading multitask tokenizer + heads + label encoders...")

tokenizer = AutoTokenizer.from_pretrained(MULTITASK_REPO_ID, token=HF_TOKEN)

queue_encoder_path = hf_hub_download(
    repo_id=MULTITASK_REPO_ID,
    filename="queue_encoder.pkl",
    token=HF_TOKEN,
)
priority_encoder_path = hf_hub_download(
    repo_id=MULTITASK_REPO_ID,
    filename="priority_encoder.pkl",
    token=HF_TOKEN,
)

queue_encoder = load_label_encoder(queue_encoder_path)
priority_encoder = load_label_encoder(priority_encoder_path)

queue_id_to_label = {i: label for i, label in enumerate(queue_encoder.classes_)}
priority_id_to_label = {i: label for i, label in enumerate(priority_encoder.classes_)}

heads_path = hf_hub_download(
    repo_id=MULTITASK_REPO_ID,
    filename="heads.pt",
    token=HF_TOKEN,
)

multitask_model = MultiTaskModel(
    repo_id=MULTITASK_REPO_ID,
    num_queue_labels=len(queue_encoder.classes_),
    num_priority_labels=len(priority_encoder.classes_),
    hf_token=HF_TOKEN,
).to(DEVICE)

try:
    heads_state = torch.load(heads_path, map_location=DEVICE, weights_only=True)
except TypeError:
    heads_state = torch.load(heads_path, map_location=DEVICE)

multitask_model.queue_classifier.load_state_dict(heads_state["queue_classifier"])
multitask_model.priority_classifier.load_state_dict(heads_state["priority_classifier"])
multitask_model.eval()

print("[startup] Multitask model loaded.")
print("[startup] Queue labels:", queue_id_to_label)
print("[startup] Priority labels:", priority_id_to_label)


# =========================
# RAG: FAISS index + metadata (robust filenames + graceful fallback)
# =========================
print("[startup] Loading FAISS RAG index + metadata...")

RAG_REPO_ID = os.getenv("RAG_REPO_ID", RAG_REPO_ID)

def try_hf_download(repo_id: str, filename_candidates: list[str], hf_token: str | None):
    last_error = None
    for filename in filename_candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token, repo_type="model")
        except Exception as e:
            last_error = e
    raise last_error

faiss_index = None
rag_metadata = None
corpus_texts: list[str] = []
corpus_queues: list[str] = []
corpus_priorities: list[str] = []

try:
    faiss_path = try_hf_download(
        repo_id=RAG_REPO_ID,
        filename_candidates=["rag_index.faiss"],
        hf_token=HF_TOKEN,
    )
    metadata_path = try_hf_download(
        repo_id=RAG_REPO_ID,
        filename_candidates=["rag_metadata.pkl"],
        hf_token=HF_TOKEN,
    )

    faiss_index = faiss.read_index(faiss_path)
    with open(metadata_path, "rb") as f:
        rag_metadata = pickle.load(f)

    corpus_texts = rag_metadata.get("texts") or rag_metadata.get("corpus_texts") or []
    corpus_queues = rag_metadata.get("queues") or rag_metadata.get("corpus_queues") or []
    corpus_priorities = rag_metadata.get("priorities") or rag_metadata.get("corpus_priorities") or []

    print(f"[startup] FAISS: {faiss_index.ntotal} vectors, {len(corpus_texts)} texts.")
except Exception as e:
    print(f"[startup][WARN] RAG disabled: could not load repo '{RAG_REPO_ID}'.")
    print(f"[startup][WARN] Reason: {e}")
    faiss_index = None


# =========================
# Embedder
# =========================
embedder = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
print("[startup] Embedder ready.")


# =========================
# Optional Groq LLM
# =========================
llm_client = None
if Groq is not None and GROQ_KEY:
    llm_client = Groq(api_key=GROQ_KEY)
    print(f"[startup] LLM Groq {GROQ_MODEL_ID} ready.")
else:
    print("[startup] No GROQ_API_KEY (LLM preprocessing skipped).")


# =========================
# Text + LLM preprocessing
# =========================
def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def llm_preprocess(raw_text: str) -> dict:
    raw_text = str(raw_text or "").strip()
    fallback = {
        "structured_body": raw_text,
        "subject": "Support Ticket",
        "urgency_signals": [],
        "tech_keywords": [],
        "explanation": "LLM preprocessing skipped (no GROQ_API_KEY set).",
    }
    if llm_client is None:
        return fallback

    system_prompt = (
        "You are a customer-support triage assistant. "
        "Given a raw support ticket, return ONLY valid JSON with these keys:\n"
        "  structured_body: cleaned, professional rewrite of the ticket\n"
        "  subject: concise 6-10 word subject line\n"
        "  urgency_signals: list of urgency phrases found\n"
        "  tech_keywords: list of technical terms found\n"
        "  explanation: one sentence explaining the likely department\n"
        "Return nothing outside the JSON object."
    )

    try:
        resp = llm_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.2,
            max_tokens=500,
        )

        content = (resp.choices[0].message.content or "").strip()

        # Handle totally empty responses
        if not content:
            fallback["explanation"] = "LLM returned empty response; using raw text."
            return fallback

        # Strip common fenced blocks
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"\s*```$", "", content).strip()

        # If model wrapped JSON in extra text, extract first JSON object
        if not content.startswith("{"):
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                content = m.group(0).strip()

        d = json.loads(content)

        return {
            "structured_body": d.get("structured_body", raw_text),
            "subject": d.get("subject", "Support Ticket"),
            "urgency_signals": d.get("urgency_signals", []) or [],
            "tech_keywords": d.get("tech_keywords", []) or [],
            "explanation": d.get("explanation", "") or "",
        }

    except Exception as e:
        # Don’t spam giant tracebacks for an optional step; just fall back.
        print("[llm_preprocess] Error; falling back:", e)
        fallback["explanation"] = f"LLM error: {e}"
        return fallback



# =========================
# RAG retrieval
# =========================
def retrieve_similar_tickets(query_text: str, k: int = 5) -> list[dict]:
    if faiss_index is None:
        return []

    vec = embedder.encode(query_text, normalize_embeddings=True, convert_to_numpy=True).reshape(1, -1)
    scores, indices = faiss_index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append(
            {
                "text": corpus_texts[idx][:300] if idx < len(corpus_texts) else "",
                "queue": corpus_queues[idx] if idx < len(corpus_queues) else "",
                "priority": corpus_priorities[idx] if idx < len(corpus_priorities) else "",
                "similarity": round(float(score), 4),
            }
        )
    return results


# =========================
# Multitask transformer inference
# =========================
def tokenize_for_model(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs.pop("token_type_ids", None)
    return {k: v.to(DEVICE) for k, v in inputs.items()}


@torch.no_grad()
def transformer_predict(text: str) -> dict:
    inputs = tokenize_for_model(text)
    outputs = multitask_model(**inputs)

    queue_logits, priority_logits = outputs["logits"]
    queue_probs = torch.softmax(queue_logits, dim=-1).detach().cpu().numpy()[0]
    priority_probs = torch.softmax(priority_logits, dim=-1).detach().cpu().numpy()[0]

    queue_id = int(queue_probs.argmax())
    priority_id = int(priority_probs.argmax())

    return {
        "queue_pred": queue_id_to_label[queue_id],
        "queue_conf": float(queue_probs.max()),
        "queue_probs": {queue_id_to_label[i]: round(float(v), 4) for i, v in enumerate(queue_probs)},
        "priority_pred": priority_id_to_label[priority_id],
        "priority_conf": float(priority_probs.max()),
        "priority_probs": {priority_id_to_label[i]: round(float(v), 4) for i, v in enumerate(priority_probs)},
        "queue_logits": queue_logits.detach().cpu().numpy()[0].tolist(),
        "priority_logits": priority_logits.detach().cpu().numpy()[0].tolist(),
    }


# =========================
# Ensemble
# =========================
def ensemble_predict(text: str, retrieved: list[dict], base_alpha: float = 0.7) -> dict:
    t = transformer_predict(text)

    rag_scores = defaultdict(float)
    for r in retrieved:
        rag_scores[r["queue"]] += r["similarity"]

    total_q = sum(rag_scores.values()) or 1.0
    rag_probs = {label: score / total_q for label, score in rag_scores.items()}

    rag_queue = max(rag_probs, key=rag_probs.get) if rag_probs else t["queue_pred"]
    rag_max = max(rag_probs.values()) if rag_probs else 0.0
    top_sim = max((r["similarity"] for r in retrieved), default=0.0)

    alpha = base_alpha + (t["queue_conf"] - 0.5) * 0.4
    alpha = max(0.4, min(0.95, alpha))

    rag_strength = max(0.0, min(1.0, (rag_max - 0.60) / 0.40))
    alpha_min_when_rag_strong = 0.25
    alpha = (1 - rag_strength) * alpha + rag_strength * alpha_min_when_rag_strong

    combined_queue = {}
    for _, label in queue_id_to_label.items():
        t_prob = float(t["queue_probs"].get(label, 0.0))
        r_prob = float(rag_probs.get(label, 0.0))
        combined_queue[label] = alpha * t_prob + (1 - alpha) * r_prob

    sorted_queue = sorted(combined_queue.items(), key=lambda x: -x[1])
    final_queue = sorted_queue[0][0] if sorted_queue else t["queue_pred"]
    runner_up_queue = sorted_queue[1][0] if len(sorted_queue) > 1 else final_queue

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

    priority_vote = defaultdict(float)
    for r in retrieved:
        priority_vote[r["priority"]] += r["similarity"]

    total_p = sum(priority_vote.values()) or 1.0
    rag_priority_probs = {label: score / total_p for label, score in priority_vote.items()}
    rag_priority = max(priority_vote, key=priority_vote.get) if priority_vote else t["priority_pred"]

    alpha_p = base_alpha + (t["priority_conf"] - 0.5) * 0.4
    alpha_p = max(0.4, min(0.95, alpha_p))

    combined_priority = {}
    for _, label in priority_id_to_label.items():
        t_prob = float(t["priority_probs"].get(label, 0.0))
        r_prob = float(rag_priority_probs.get(label, 0.0))
        combined_priority[label] = alpha_p * t_prob + (1 - alpha_p) * r_prob

    final_priority = max(combined_priority, key=combined_priority.get) if combined_priority else t["priority_pred"]

    return {
        "final_queue": final_queue,
        "runner_up_queue": runner_up_queue,
        "final_priority": final_priority,
        "transformer": t,
        "rag_queue": rag_queue,
        "rag_priority": rag_priority,
        "rag_probs": dict(sorted(rag_probs.items(), key=lambda x: -x[1])),
        "rag_priority_probs": dict(sorted(rag_priority_probs.items(), key=lambda x: -x[1])),
        "agreement": (final_queue == rag_queue),
        "alpha_used": round(float(alpha), 3),
        "priority_alpha_used": round(float(alpha_p), 3),
        "all_scores": dict(sorted(combined_queue.items(), key=lambda x: -x[1])),
        "priority_scores": dict(sorted(combined_priority.items(), key=lambda x: -x[1])),
        "rag_max": round(float(rag_max), 4),
        "top_sim": round(float(top_sim), 4),
        "veto_triggered": bool(veto_triggered),
    }


# =========================
# Pipeline
# =========================
def full_pipeline(raw_text: str) -> dict:
    llm_result = llm_preprocess(raw_text)
    cleaned_body = clean_text(llm_result["structured_body"])

    retrieved = retrieve_similar_tickets(cleaned_body, k=5)
    ensemble = ensemble_predict(cleaned_body, retrieved)

    return {
        "raw_text": raw_text,
        "clean_text": cleaned_body,
        "llm": llm_result,
        "retrieved": retrieved,
        "ensemble": ensemble,
    }


# =========================
# Explainability helpers (IG + occlusion)
# =========================
STOPWORDS = set(
    """
the a an and or but if then else when while of to in on for from with without at by as
is are was were be been being it this that these those we i you they he she our their my your me us them
can could should would please help need needed urgent thanks thank
""".split()
)

def get_embedding_layer(model):
    for base in ["distilbert", "bert", "roberta", "deberta", "electra", "albert"]:
        if hasattr(model, base):
            m = getattr(model, base)
            if hasattr(m, "embeddings"):
                return m.embeddings
    if hasattr(model, "encoder"):
        enc = getattr(model, "encoder")
        if hasattr(enc, "embeddings"):
            return enc.embeddings
    return None

def merge_wordpieces(tokens, scores):
    merged = []
    current_token, current_score = "", 0.0
    for tok, sc in zip(tokens, scores):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if tok.startswith("##"):
            current_token += tok[2:]
            current_score += float(sc)
        else:
            if current_token:
                merged.append((current_token, current_score))
            current_token, current_score = tok, float(sc)
    if current_token:
        merged.append((current_token, current_score))
    return merged

def normalize_signed(vals, eps=1e-9):
    m = max((abs(v) for v in vals), default=1.0)
    m = max(m, eps)
    return [float(v) / m for v in vals]

def rgba_for_score(x):
    x = max(-1.0, min(1.0, float(x)))
    if x >= 0:
        r, g, b = 255, int(255 * (1 - 0.55 * x)), int(255 * (1 - 0.55 * x))
        a = 0.10 + 0.35 * x
    else:
        x = -x
        r, g, b = int(255 * (1 - 0.55 * x)), int(255 * (1 - 0.55 * x)), 255
        a = 0.10 + 0.35 * x
    return f"rgba({r},{g},{b},{a:.3f})"

def render_token_heat(tokens_scores, title, subtitle=None, max_tokens=80):
    items = tokens_scores[:max_tokens]
    chips = []
    for tok, sc in items:
        bg = rgba_for_score(sc)
        safe = tok.replace("<", "&lt;").replace(">", "&gt;")
        chips.append(
            f"<span class='chip' style='background:{bg}'>"
            f"<span class='t'>{safe}</span> "
            f"<span class='s'>{sc:.3f}</span></span>"
        )
    sub = f"<div class='sub'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='heatbox'>"
        f"<div class='htitle'>{title}</div>"
        f"{sub}"
        f"<div class='chips'>{''.join(chips)}</div>"
        f"</div>"
    )

def try_import_captum():
    try:
        from captum.attr import LayerIntegratedGradients
        return LayerIntegratedGradients
    except Exception:
        return None

@dataclass
class SimpleOutput:
    logits: torch.Tensor

class QueueWrapper(nn.Module):
    def __init__(self, mt: MultiTaskModel):
        super().__init__()
        self.mt = mt
        self.distilbert = mt.encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.mt(input_ids=input_ids, attention_mask=attention_mask)
        queue_logits, _ = out["logits"]
        return SimpleOutput(logits=queue_logits)

class PriorityWrapper(nn.Module):
    def __init__(self, mt: MultiTaskModel):
        super().__init__()
        self.mt = mt
        self.distilbert = mt.encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.mt(input_ids=input_ids, attention_mask=attention_mask)
        _, priority_logits = out["logits"]
        return SimpleOutput(logits=priority_logits)

queue_model_for_ig = QueueWrapper(multitask_model).to(DEVICE).eval()
priority_model_for_ig = PriorityWrapper(multitask_model).to(DEVICE).eval()

def integrated_gradients_explain(text: str, model: nn.Module, target_idx: int):
    LayerIntegratedGradients = try_import_captum()
    if LayerIntegratedGradients is None:
        return None, "Captum is not installed. Run: pip install captum"

    embedding_layer = get_embedding_layer(model)
    if embedding_layer is None:
        return None, "Could not locate the transformer embedding layer for attribution."

    try:
        inputs = tokenize_for_model(text)
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

        def forward_func(iids, amask):
            out = model(input_ids=iids, attention_mask=amask)
            return out.logits

        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        attr = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            baselines=baseline_ids,
            target=target_idx,
            n_steps=16,
        )

        token_attr = attr.sum(dim=-1).detach().cpu().squeeze(0)
        toks = tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().squeeze(0).tolist())
        scores = token_attr.tolist()

        merged = merge_wordpieces(toks, scores)
        norm_vals = normalize_signed([s for _, s in merged])
        merged_norm = [(t, s) for (t, _), s in zip(merged, norm_vals)]

        sorted_abs = sorted(merged_norm, key=lambda x: -abs(x[1]))
        return {"sorted": sorted_abs, "ordered": merged_norm}, None
    except Exception as e:
        return None, f"Attribution error: {e}"

def simple_words(text: str):
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())

def occlusion_drop_report(text: str, top_words: list[str], base_pred: dict):
    reports = []
    base_q = base_pred["queue_conf"]
    base_p = base_pred["priority_conf"]

    for w in top_words:
        occluded = re.sub(rf"\b{re.escape(w)}\b", " ", text, flags=re.IGNORECASE)
        occluded = re.sub(r"\s+", " ", occluded).strip()
        if not occluded:
            continue
        pred2 = transformer_predict(occluded)
        reports.append(
            {
                "word": w,
                "queue_drop": max(0.0, base_q - pred2["queue_conf"]),
                "prio_drop": max(0.0, base_p - pred2["priority_conf"]),
                "new_queue_conf": pred2["queue_conf"],
                "new_priority_conf": pred2["priority_conf"],
            }
        )
    return reports

def pick_top_support_words(attr_sorted, k=3):
    out = []
    for tok, sc in attr_sorted:
        w = tok.lower().strip(".,!?")
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

def extract_competitor_keywords(retrieved: list[dict], competitor: str, k=8):
    texts = [r["text"] for r in retrieved if r["queue"] == competitor]
    if not texts:
        return []
    words = []
    for t in texts:
        words += simple_words(t)
    counts = Counter([w for w in words if w not in STOPWORDS and len(w) >= 4])
    return [w for w, _ in counts.most_common(k)]


# =========================
# Plotly radar (optional)
# =========================
def try_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except Exception:
        return None

def build_dept_radar_fig(result: dict, top_k: int = 7):
    go = try_import_plotly()
    if go is None:
        return None

    ensemble = result["ensemble"]
    scores = ensemble["all_scores"]

    labels = list(scores.keys())[:max(3, min(top_k, len(scores)))]
    combined = [scores[l] for l in labels]
    transformer = [ensemble["transformer"]["queue_probs"].get(l, 0.0) for l in labels]
    rag_vote = [ensemble["rag_probs"].get(l, 0.0) for l in labels]

    labels_c = labels + labels[:1]
    combined_c = combined + combined[:1]
    transformer_c = transformer + transformer[:1]
    rag_vote_c = rag_vote + rag_vote[:1]
    vmax = max(combined_c + transformer_c + rag_vote_c) if labels else 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=transformer_c, theta=labels_c, fill="toself", name="Transformer"))
    fig.add_trace(go.Scatterpolar(r=rag_vote_c, theta=labels_c, fill="toself", name="RAG vote"))
    fig.add_trace(go.Scatterpolar(r=combined_c, theta=labels_c, fill="toself", name="Ensemble"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(1.05, vmax * 1.05)])),
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
        title="Department radar (top classes)",
    )
    return fig

def build_priority_radar_fig(result: dict):
    go = try_import_plotly()
    if go is None:
        return None

    ensemble = result["ensemble"]
    labels = list(sorted(ensemble["transformer"]["priority_probs"].keys()))
    transformer = [ensemble["transformer"]["priority_probs"].get(l, 0.0) for l in labels]
    rag_vote = [ensemble["rag_priority_probs"].get(l, 0.0) for l in labels]
    combined = [float(ensemble["priority_scores"].get(l, 0.0)) for l in labels]

    labels_c = labels + labels[:1]
    transformer_c = transformer + transformer[:1]
    rag_vote_c = rag_vote + rag_vote[:1]
    combined_c = combined + combined[:1]
    vmax = max(transformer_c + rag_vote_c + combined_c) if labels else 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=transformer_c, theta=labels_c, fill="toself", name="Transformer"))
    fig.add_trace(go.Scatterpolar(r=rag_vote_c, theta=labels_c, fill="toself", name="RAG vote"))
    fig.add_trace(go.Scatterpolar(r=combined_c, theta=labels_c, fill="toself", name="Final"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(1.05, vmax * 1.05)])),
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        title="Priority radar",
    )
    return fig


# =========================
# Rendering (markdown panels)
# =========================
PRIORITY_LABEL_DISPLAY = {"high": "High", "medium": "Medium", "low": "Low"}

def render_routing_card(result: dict) -> str:
    e = result["ensemble"]
    q = e["final_queue"]
    p = e["final_priority"]
    qc = e["transformer"]["queue_conf"]
    alpha = e["alpha_used"]
    rag_msg = "RAG agrees." if e["agreement"] else f"RAG suggested: {e['rag_queue']}."
    return "\n".join(
        [
            "## Routing Decision",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Department | **{q}** |",
            f"| Priority | {PRIORITY_LABEL_DISPLAY.get(p.lower(), p.capitalize())} |",
            f"| Transformer confidence | {qc:.3f} |",
            f"| Ensemble alpha (higher = more transformer weight) | {alpha} |",
            f"| RAG validation | {rag_msg} |",
        ]
    )

def render_dept_bars(result: dict) -> str:
    # UI FIX: fenced code block => preserves line breaks + alignment
    scores = result["ensemble"]["all_scores"]
    top = next(iter(scores.values()), 1e-9)
    final_queue = result["ensemble"]["final_queue"]

    lines = ["Department confidence (ensemble)"]
    for label, prob in scores.items():
        pct = prob / max(top, 1e-9)
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        mark = "< predicted" if label == final_queue else ""
        lines.append(f"{label:<34} {bar}  {prob:.3f} {mark}".rstrip())

    return "```text\n" + "\n".join(lines) + "\n```"

def render_prio_bars(result: dict) -> str:
    # UI FIX: fenced code block => preserves line breaks + alignment
    probs = result["ensemble"]["transformer"]["priority_probs"]
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    final_priority = result["ensemble"]["final_priority"]

    lines = ["Priority confidence (transformer)"]
    for label, prob in sorted_probs:
        bar = "█" * int(prob * 28) + "░" * (28 - int(prob * 28))
        mark = "< predicted" if label == final_priority else ""
        shown = PRIORITY_LABEL_DISPLAY.get(label.lower(), label)
        lines.append(f"{shown:<8} {bar}  {prob:.3f} {mark}".rstrip())

    return "```text\n" + "\n".join(lines) + "\n```"

def render_llm_panel(result: dict) -> str:
    llm = result["llm"]
    urg = ", ".join(llm["urgency_signals"]) if llm["urgency_signals"] else "None detected."
    kw = ", ".join(llm["tech_keywords"]) if llm["tech_keywords"] else "None detected."

    return "\n".join(
        [
            "## LLM pre-processing",
            "",
            f"- **Subject:** {llm['subject']}",
            f"- **Urgency signals:** {urg}",
            f"- **Technical keywords:** {kw}",
            "",
            "### Cleaned ticket body",
            "",
            llm["structured_body"],
            "",
            "---",
            "",
            (llm.get("explanation", "") or "n/a"),
        ]
    )

def render_rag_panel(result: dict) -> str:
    if not result["retrieved"]:
        return "## RAG\n\nRAG disabled or no retrieved tickets."
    lines = ["## RAG top 5 similar tickets", ""]
    for i, rt in enumerate(result["retrieved"], 1):
        bar = "█" * int(rt["similarity"] * 20)
        lines += [
            f"**{i}.** {rt['queue']}  |  {rt['priority'].capitalize()}  |  sim={rt['similarity']:.4f}  {bar}",
            f"> {rt['text']}...",
            "",
        ]
    return "\n".join(lines)

def render_rag_vote_breakdown(result: dict) -> str:
    # UI FIX: fenced code block keeps the “bar list” readable
    rag_probs = result["ensemble"]["rag_probs"]
    if not rag_probs:
        return "## RAG vote breakdown\n\nNo RAG votes available."

    top = next(iter(rag_probs.values()), 1e-9)
    lines = ["RAG vote breakdown (similarity-weighted)"]
    for label, prob in list(rag_probs.items())[:10]:
        pct = prob / max(top, 1e-9)
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        lines.append(f"{label:<34} {bar}  {prob:.3f}".rstrip())

    return "```text\n" + "\n".join(lines) + "\n```"

def render_decision_trace(result: dict) -> str:
    e = result["ensemble"]
    t = e["transformer"]

    def top_k(d, k=3):
        return sorted(d.items(), key=lambda x: -x[1])[:k]

    ttop = top_k(t["queue_probs"], 3)
    rtop = top_k(e["rag_probs"], 3) if e["rag_probs"] else []
    etop = list(e["all_scores"].items())[:3]
    ptop = top_k(t["priority_probs"], 3)

    return "\n".join(
        [
            "## Decision trace",
            "",
            "**Input used:**",
            result["clean_text"],
            "",
            "| Stage | Key fields |",
            "|---|---|",
            f"| Queue alpha | {e['alpha_used']} |",
            f"| Transformer top-3 dept | {', '.join([f'{k} ({v:.3f})' for k,v in ttop])} |",
            f"| RAG vote top-3 dept | {', '.join([f'{k} ({v:.3f})' for k,v in rtop]) if rtop else 'n/a'} |",
            f"| Ensemble top-3 dept | {', '.join([f'{k} ({v:.3f})' for k,v in etop])} |",
            f"| Transformer top-3 priority | {', '.join([f'{k} ({v:.3f})' for k,v in ptop])} |",
        ]
    )

def render_what_would_change_it(result: dict, keyphrases: dict) -> str:
    e = result["ensemble"]
    predicted = e["final_queue"]
    runner = e["runner_up_queue"]

    all_scores = list(e["all_scores"].items())
    spred = all_scores[0][1] if all_scores else 0.0
    srun = all_scores[1][1] if len(all_scores) > 1 else 0.0
    margin = spred - srun

    competitor_keywords = extract_competitor_keywords(result["retrieved"], runner, k=8)
    remove_words = keyphrases.get("support_words", [])
    add_words = competitor_keywords[:5]

    return "\n".join(
        [
            "## What would change it? (heuristic)",
            "",
            f"**Current pick:** {predicted}",
            f"**Nearest competitor:** {runner}",
            f"**Score margin (ensemble):** {margin:.3f}",
            "",
            "### If you removed these words",
            "Try removing/avoiding: " + (", ".join(remove_words) if remove_words else "No strong support-words identified."),
            "",
            "### If you added more detail for the competitor",
            "Terms that appear in competitor-like retrieved tickets: " + (", ".join(add_words) if add_words else "No competitor-labelled exemplars in top-5 retrieval."),
            "",
            "_Note: this is a quick sensitivity probe (no LLM); it’s meant to show what the model is relying on._",
        ]
    )


# =========================
# UI glue
# =========================
EMPTY_TEXT = "Submit a ticket above to see results."

def process_ticket(user_message: str, history: list):
    if not user_message or not user_message.strip():
        return (
            history or [],
            EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT,
            EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT,
            None, None
        )

    history = history or []
    try:
        result = full_pipeline(user_message.strip())
    except Exception as e:
        traceback.print_exc()
        err_text = f"Pipeline error: {e}"
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": err_text},
        ]
        return (
            history,
            err_text, err_text, err_text, err_text, err_text, err_text, err_text,
            err_text, err_text, err_text, err_text,
            None, None
        )

    e = result["ensemble"]
    q = e["final_queue"]
    p = e["final_priority"]
    qc = e["transformer"]["queue_conf"]
    rag_note = "RAG agrees." if e["agreement"] else f"RAG suggested {e['rag_queue']} (ensemble overrides)."
    explanation = result["llm"].get("explanation", "")

    reply = "\n".join(
        [
            "Ticket analysed and routed!",
            f"- Department: {q}",
            f"- Priority: {PRIORITY_LABEL_DISPLAY.get(p.lower(), p.capitalize())}",
            f"- Transformer confidence: {qc:.3f}",
            f"- {rag_note}",
            (f"- {explanation}" if explanation else ""),
            "",
            "Full breakdown is available in the panels on the right.",
        ]
    ).strip()

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    keyphrases_state = {"support_words": []}
    transformer_pred = e["transformer"]

    # Queue IG
    queue_idx = None
    for i, lab in queue_id_to_label.items():
        if lab == transformer_pred["queue_pred"]:
            queue_idx = i
            break

    if queue_idx is not None:
        qattr, qerr = integrated_gradients_explain(result["clean_text"], queue_model_for_ig, queue_idx)
        if qattr:
            keyphrases_state["support_words"] = pick_top_support_words(qattr["sorted"], k=3)
            queue_html = render_token_heat(
                qattr["ordered"],
                title=f"Department attribution (Transformer: {transformer_pred['queue_pred']})",
                subtitle="Red supports the predicted class; blue pushes against it. (Integrated Gradients)",
            )
        else:
            queue_html = render_token_heat([], "Department attribution unavailable", subtitle=qerr or "Unknown error.")
    else:
        queue_html = render_token_heat([], "Department attribution unavailable", subtitle="Could not resolve predicted class index.")

    # Priority IG
    priority_idx = None
    for i, lab in priority_id_to_label.items():
        if lab == transformer_pred["priority_pred"]:
            priority_idx = i
            break

    if priority_idx is not None:
        pattr, perr = integrated_gradients_explain(result["clean_text"], priority_model_for_ig, priority_idx)
        if pattr:
            priority_html = render_token_heat(
                pattr["ordered"],
                title=f"Priority attribution (Transformer: {transformer_pred['priority_pred']})",
                subtitle="Same colour logic as above.",
            )
        else:
            priority_html = render_token_heat([], "Priority attribution unavailable", subtitle=perr or "Unknown error.")
    else:
        priority_html = render_token_heat([], "Priority attribution unavailable", subtitle="Could not resolve predicted class index.")

    # Occlusion check
    if keyphrases_state["support_words"]:
        reports = occlusion_drop_report(result["clean_text"], keyphrases_state["support_words"], transformer_pred)
        if reports:
            occl_lines = [
                "## Occlusion check (remove top words)",
                "",
                "| Removed word | Dept conf drop | New dept conf | Priority conf drop | New priority conf |",
                "|---|---:|---:|---:|---:|",
            ]
            for rep in reports:
                occl_lines.append(
                    f"| {rep['word']} | {rep['queue_drop']:.3f} | {rep['new_queue_conf']:.3f} | "
                    f"{rep['prio_drop']:.3f} | {rep['new_priority_conf']:.3f} |"
                )
            occl_md = "\n".join(occl_lines)
        else:
            occl_md = "## Occlusion check\n\nNo occlusion results (text became empty or perturbation failed)."
    else:
        occl_md = "## Occlusion check\n\nNo strong support-words found or attribution unavailable. (Install Captum to enable.)"

    change_md = render_what_would_change_it(result, keyphrases_state)
    trace_md = render_decision_trace(result)

    radar_dept = build_dept_radar_fig(result, top_k=7)
    radar_prio = build_priority_radar_fig(result)

    return (
        history,
        render_routing_card(result),
        render_dept_bars(result),
        render_prio_bars(result),
        render_llm_panel(result),
        render_rag_panel(result),
        render_rag_vote_breakdown(result),
        trace_md,
        queue_html,
        priority_html,
        occl_md,
        change_md,
        radar_dept,
        radar_prio,
    )

def clear_all():
    return (
        [],
        EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT,
        EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT, EMPTY_TEXT,
        None, None
    )

TAB_KEYS = ["dept", "prio", "radar", "keyphr", "change", "trace", "llm", "rag"]

def set_active_panel(selected: str):
    if selected not in TAB_KEYS:
        selected = "dept"

    def button_variant(sel):
        return gr.update(variant="primary" if selected == sel else "secondary")

    def panel_visible(sel):
        return gr.update(visible=(selected == sel))

    return (
        selected,
        button_variant("dept"),
        button_variant("prio"),
        button_variant("radar"),
        button_variant("keyphr"),
        button_variant("change"),
        button_variant("trace"),
        button_variant("llm"),
        button_variant("rag"),
        panel_visible("dept"),
        panel_visible("prio"),
        panel_visible("radar"),
        panel_visible("keyphr"),
        panel_visible("change"),
        panel_visible("trace"),
        panel_visible("llm"),
        panel_visible("rag"),
    )


# =========================
# UI styling + examples
# =========================
CSS = """
body, .gradio-container { font-family: Inter, Segoe UI, sans-serif; }
#header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
  border-radius: 12px;
  padding: 20px 28px;
  margin-bottom: 8px;
  color: #fff;
}
#header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
#header p { margin: 4px 0 0; font-size: 0.88rem; opacity: 0.85; }

#llm-panel .prose blockquote { border-left: 4px solid #2d6a9f; padding-left: 10px; color: #dde; }
#rag-panel .prose blockquote { border-left: 4px solid #10b981; padding-left: 10px; color: #dde; }

.heatbox {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 14px 14px;
  margin-bottom: 10px;
}
.htitle { font-weight: 700; margin-bottom: 6px; }
.sub { opacity: 0.8; font-size: 0.85rem; margin-bottom: 10px; }
.chips { line-height: 2.3; }
.chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  margin: 3px 6px 3px 0;
  border: 1px solid rgba(255,255,255,0.08);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.86rem;
}
.chip .s { opacity: 0.8; margin-left: 6px; }

/* Horizontal tab strip */
#results-tabstrip { display: flex; flex-wrap: nowrap; gap: 8px; overflow-x: auto; overflow-y: hidden; padding-bottom: 6px;
  scroll-behavior: smooth; -webkit-overflow-scrolling: touch;
}
#results-tabstrip::-webkit-scrollbar { height: 8px; }
#results-tabstrip::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.18); border-radius: 99px; }
#results-tabstrip::-webkit-scrollbar-track { background: rgba(255,255,255,0.06); border-radius: 99px; }
#results-tabstrip button { flex: 0 0 auto; min-width: max-content; border-radius: 12px !important; padding: 8px 12px !important; }
"""

EXAMPLES = [
    "My internet isnt working and I cant log in since yesterday urgent deadline today.",
    "I think I was charged twice for my subscription last month. Can someone look into this?",
    "We need to update payroll records for three new hires who started this week.",
    "The checkout page keeps crashing when customers try to pay with Visa. Were losing sales.",
    "Can you tell me your standard service hours and how to escalate a complaint?",
    "Our Salesforce CRM has been down since 9am, the entire sales team is blocked.",
]


# =========================
# Build Gradio app (same layout as before)
# =========================
with gr.Blocks(css=CSS, title=APP_TITLE) as demo:
    gr.HTML(
        """
        <div id="header">
          <h1>Customer Support Ticket Router</h1>
          <p>
            Multitask DistilBERT (dept + priority) &nbsp;&nbsp;•&nbsp;&nbsp;
            FAISS RAG retrieval &nbsp;&nbsp;•&nbsp;&nbsp;
            Dynamic ensemble (Fix 1 + Fix 2) &nbsp;&nbsp;•&nbsp;&nbsp;
            Optional Groq LLaMA-3.3 preprocessing &nbsp;&nbsp;•&nbsp;&nbsp;
            Explainability: IG + Occlusion + Trace + Radar
          </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="Ticket conversation", height=440, show_label=True)
            msg_input = gr.Textbox(
                placeholder="Describe your issue in plain language",
                label="Your ticket",
                lines=3,
                max_lines=8,
                show_label=True,
            )
            with gr.Row():
                submit_btn = gr.Button("Submit ticket", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            gr.Examples(examples=EXAMPLES, inputs=msg_input, label="Quick examples (click to load)", examples_per_page=6)

        with gr.Column(scale=6):
            routing_card = gr.Markdown(value=EMPTY_TEXT)

            active_panel = gr.State("dept")

            with gr.Row(elem_id="results-tabstrip"):
                btn_dept = gr.Button("Department", variant="primary")
                btn_prio = gr.Button("Priority", variant="secondary")
                btn_radar = gr.Button("Radar", variant="secondary")
                btn_key = gr.Button("Key phrases", variant="secondary")
                btn_change = gr.Button("What would change it?", variant="secondary")
                btn_trace = gr.Button("Decision trace", variant="secondary")
                btn_llm = gr.Button("LLM", variant="secondary")
                btn_rag = gr.Button("RAG", variant="secondary")

            panel_dept = gr.Column(visible=True)
            panel_prio = gr.Column(visible=False)
            panel_radar = gr.Column(visible=False)
            panel_key = gr.Column(visible=False)
            panel_change = gr.Column(visible=False)
            panel_trace = gr.Column(visible=False)
            panel_llm = gr.Column(visible=False)
            panel_rag = gr.Column(visible=False)

            with panel_dept:
                dept_bars = gr.Markdown(value=EMPTY_TEXT, elem_id="dept-bars")
            with panel_prio:
                prio_bars = gr.Markdown(value=EMPTY_TEXT, elem_id="prio-bars")
            with panel_radar:
                radar_dept_plot = gr.Plot(value=None)
                radar_prio_plot = gr.Plot(value=None)
            with panel_key:
                key_q_html = gr.HTML(value=EMPTY_TEXT)
                key_p_html = gr.HTML(value=EMPTY_TEXT)
                occl_md = gr.Markdown(value=EMPTY_TEXT)
            with panel_change:
                change_md = gr.Markdown(value=EMPTY_TEXT)
            with panel_trace:
                trace_md = gr.Markdown(value=EMPTY_TEXT)
            with panel_llm:
                llm_panel = gr.Markdown(value=EMPTY_TEXT, elem_id="llm-panel")
            with panel_rag:
                rag_panel = gr.Markdown(value=EMPTY_TEXT, elem_id="rag-panel")
                rag_vote = gr.Markdown(value=EMPTY_TEXT)

            tab_outputs = [
                active_panel,
                btn_dept, btn_prio, btn_radar, btn_key, btn_change, btn_trace, btn_llm, btn_rag,
                panel_dept, panel_prio, panel_radar, panel_key, panel_change, panel_trace, panel_llm, panel_rag,
            ]

            btn_dept.click(fn=lambda: set_active_panel("dept"), inputs=[], outputs=tab_outputs)
            btn_prio.click(fn=lambda: set_active_panel("prio"), inputs=[], outputs=tab_outputs)
            btn_radar.click(fn=lambda: set_active_panel("radar"), inputs=[], outputs=tab_outputs)
            btn_key.click(fn=lambda: set_active_panel("keyphr"), inputs=[], outputs=tab_outputs)
            btn_change.click(fn=lambda: set_active_panel("change"), inputs=[], outputs=tab_outputs)
            btn_trace.click(fn=lambda: set_active_panel("trace"), inputs=[], outputs=tab_outputs)
            btn_llm.click(fn=lambda: set_active_panel("llm"), inputs=[], outputs=tab_outputs)
            btn_rag.click(fn=lambda: set_active_panel("rag"), inputs=[], outputs=tab_outputs)

    outputs = [
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
        radar_dept_plot,
        radar_prio_plot,
    ]

    submit_btn.click(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=outputs,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)

    msg_input.submit(
        fn=process_ticket,
        inputs=[msg_input, chatbot],
        outputs=outputs,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=outputs,
    ).then(fn=lambda: "", inputs=[], outputs=msg_input)


if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, share=True)
