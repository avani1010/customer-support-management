import re
import torch
import torch.nn as nn
import plotly.graph_objects as go
from collections import Counter

PE = {"high": "🔴", "medium": "🟡", "low": "🟢"}
PCOLOR = {"high": "#f87171", "medium": "#fbbf24", "low": "#4ade80"}
PBG    = {"high": "rgba(248,113,113,.15)", "medium": "rgba(251,191,36,.12)", "low": "rgba(74,222,128,.12)"}
PBORDER= {"high": "rgba(248,113,113,.35)", "medium": "rgba(251,191,36,.3)", "low": "rgba(74,222,128,.3)"}
EMPTY  = "_(Submit a ticket above to see results)_"
STOPWORDS = {
    "the","a","an","and","or","but","if","then","when","of","to","in","on",
    "for","from","with","at","by","as","is","are","was","were","be","been",
    "it","this","that","we","i","you","they","he","she","our","my","your",
    "can","could","should","would","please","help","need","thanks","thank"
}

_FONT_STACK = "IBM Plex Sans,system-ui,-apple-system,sans-serif"
_MONO_STACK = "IBM Plex Mono,monospace"

_BG0   = "#0f172a"
_BG1   = "#1e293b"
_BG2   = "#273549"
_BORDER= "#334155"
_TEXT1 = "#f1f5f9"
_TEXT2 = "#cbd5e1"
_TEXT3 = "#94a3b8"
_TEXT4 = "#64748b"
_ACCENT= "#6366f1"
_ACCENT2="#8b5cf6"


def _wrap(inner):
    return (
        f'<div style="background:{_BG0}!important;color:{_TEXT2}!important;'
        f'font-family:{_FONT_STACK}!important;'
        f'font-size:14px!important;line-height:1.5!important;padding:6px 2px!important">'
        f'{inner}</div>'
    )


def _section(label):
    return (
        f'<div style="font-size:10.5px!important;font-weight:700!important;'
        f'text-transform:uppercase!important;letter-spacing:.12em!important;'
        f'color:{_TEXT4}!important;margin:20px 0 10px!important;'
        f'padding-bottom:7px!important;border-bottom:1px solid {_BORDER}!important;'
        f'display:flex!important;align-items:center!important;gap:6px!important">'
        f'<span style="width:3px!important;height:12px!important;'
        f'background:linear-gradient(180deg,{_ACCENT},{_ACCENT2})!important;'
        f'border-radius:99px!important;display:inline-block!important;flex-shrink:0!important"></span>'
        f'{label}</div>'
    )


def _bar_row(label, pct, color, tag=""):
    filled = min(max(pct, 0), 1)
    return (
        f'<div style="margin-bottom:12px!important">'
        f'<div style="display:flex!important;justify-content:space-between!important;'
        f'align-items:center!important;margin-bottom:5px!important">'
        f'<div style="display:flex!important;align-items:center!important;gap:6px!important">'
        f'<span style="font-size:12px!important;font-weight:500!important;color:{_TEXT3}!important;'
        f'font-family:{_MONO_STACK}!important">{label}</span>'
        f'{tag}</div>'
        f'<span style="font-size:12px!important;font-weight:700!important;color:{_TEXT1}!important;'
        f'font-family:{_MONO_STACK}!important">{pct*100:.1f}%</span></div>'
        f'<div style="background:{_BORDER}!important;border-radius:99px!important;height:9px!important;'
        f'overflow:hidden!important;position:relative!important">'
        f'<div style="width:{filled*100:.1f}%!important;height:100%!important;'
        f'background:{color}!important;border-radius:99px!important;'
        f'transition:width .4s ease!important"></div>'
        f'</div></div>'
    )


def _tag(text, bg, color):
    return (
        f'<span style="display:inline-block!important;margin-left:6px!important;'
        f'font-size:10px!important;font-weight:700!important;padding:1px 7px!important;'
        f'border-radius:99px!important;background:{bg}!important;color:{color}!important;'
        f'font-family:{_FONT_STACK}!important">{text}</span>'
    )


def _card(inner, border_color=None, bg=None):
    bc = border_color or _BORDER
    bg_ = bg or _BG1
    return (
        f'<div style="background:{bg_}!important;border:1px solid {bc}!important;'
        f'border-radius:8px!important;padding:12px 14px!important;margin-bottom:8px!important">'
        f'{inner}</div>'
    )


def _stat_box(label, value, sub="", value_size="18px"):
    sub_html = (
        f'<div style="font-size:11px!important;color:{_TEXT4}!important;'
        f'margin-top:5px!important;font-family:{_MONO_STACK}!important">{sub}</div>'
        if sub else ""
    )
    return (
        f'<div style="background:{_BG1}!important;border:1px solid {_BORDER}!important;'
        f'border-radius:12px!important;padding:16px 18px!important;'
        f'box-shadow:0 1px 6px rgba(0,0,0,.25)!important">'
        f'<div style="font-size:10px!important;font-weight:700!important;text-transform:uppercase!important;'
        f'letter-spacing:.12em!important;color:{_TEXT4}!important;margin-bottom:7px!important;'
        f'font-family:{_MONO_STACK}!important">{label}</div>'
        f'<div style="font-size:{value_size}!important;font-weight:700!important;color:{_TEXT1}!important;'
        f'line-height:1.2!important;font-family:{_FONT_STACK}!important">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


class _SimpleOut:
    def __init__(self, logits): self.logits = logits

class QueueWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mt = m
        self.distilbert = m.encoder
    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.mt.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _SimpleOut(self.mt.queue_classifier(out.last_hidden_state[:, 0]))

class PriorityWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mt = m
        self.distilbert = m.encoder
    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.mt.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _SimpleOut(self.mt.priority_classifier(out.last_hidden_state[:, 0]))


def _merge_wordpieces(tokens, scores):
    merged, cur_tok, cur_sc = [], "", 0.0
    for tok, sc in zip(tokens, scores):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]: continue
        if tok.startswith("##"):
            cur_tok += tok[2:]; cur_sc += float(sc)
        else:
            if cur_tok: merged.append((cur_tok, cur_sc))
            cur_tok, cur_sc = tok, float(sc)
    if cur_tok: merged.append((cur_tok, cur_sc))
    return merged

def _norm_signed(vals, eps=1e-9):
    m = max((abs(v) for v in vals), default=1.0)
    return [float(v) / max(m, eps) for v in vals]

def _rgba(x):
    x = max(-1.0, min(1.0, float(x)))
    if x >= 0:
        a = 0.15 + 0.45 * x
        return f"rgba(99,102,241,{a:.3f})"
    else:
        x = -x
        a = 0.15 + 0.45 * x
        return f"rgba(239,68,68,{a:.3f})"

def render_token_heat(tokens_scores, title, subtitle=None, max_tokens=80):
    chips = "".join(
        f'<span style="display:inline-block!important;padding:2px 8px!important;'
        f'border-radius:99px!important;margin:2px 3px 2px 0!important;'
        f'background:{_rgba(sc)}!important;font-family:{_MONO_STACK}!important;'
        f'font-size:12px!important;color:{_TEXT1}!important;border:1px solid rgba(255,255,255,.08)!important"'
        f'title="{sc:+.3f}">{tok.replace("<","&lt;").replace(">","&gt;")}</span>'
        for tok, sc in tokens_scores[:max_tokens]
    )
    sub = (f'<div style="color:{_TEXT4}!important;font-size:11px!important;'
           f'margin-bottom:8px!important">{subtitle}</div>') if subtitle else ""
    inner = (
        f'<div style="background:{_BG1}!important;border:1px solid {_BORDER}!important;'
        f'border-radius:8px!important;padding:14px!important;margin-bottom:10px!important">'
        f'<div style="font-weight:600!important;font-size:13px!important;'
        f'color:{_TEXT1}!important;margin-bottom:4px!important">{title}</div>'
        f'{sub}'
        f'<div style="line-height:2.6!important">{chips}</div>'
        f'</div>'
    )
    return _wrap(inner)

def ig_explain(text, ig_model, target_idx, tokenizer, device):
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        return None, "Captum not installed — run: pip install captum"
    emb_layer = None
    for attr in ["distilbert", "bert", "roberta", "encoder"]:
        m = getattr(ig_model, attr, None)
        if m and hasattr(m, "embeddings"):
            emb_layer = m.embeddings; break
    if emb_layer is None:
        return None, "Could not locate embedding layer"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        baseline = ids.clone()
        special = torch.zeros_like(baseline, dtype=torch.bool)
        pad_id = tokenizer.pad_token_id or 0
        if tokenizer.cls_token_id: special |= (baseline == tokenizer.cls_token_id)
        if tokenizer.sep_token_id: special |= (baseline == tokenizer.sep_token_id)
        baseline[~special] = pad_id
        def fwd(iids, amask): return ig_model(input_ids=iids, attention_mask=amask).logits
        lig = LayerIntegratedGradients(fwd, emb_layer)
        attr = lig.attribute(inputs=ids, additional_forward_args=(mask,),
                             baselines=baseline, target=target_idx, n_steps=16)
        raw = attr.sum(dim=-1).detach().cpu().squeeze(0).tolist()
        toks = tokenizer.convert_ids_to_tokens(ids.detach().cpu().squeeze(0).tolist())
        merged = _merge_wordpieces(toks, raw)
        normed = _norm_signed([s for _, s in merged])
        ordered = [(t, s) for (t, _), s in zip(merged, normed)]
        return {"ordered": ordered, "sorted": sorted(ordered, key=lambda x: -abs(x[1]))}, None
    except Exception as e:
        return None, f"Attribution error: {e}"

def pick_top_support_words(attr_sorted, k=3):
    out = []
    for tok, sc in attr_sorted:
        w = tok.lower().strip("`\"'.,!?()[]{};:")
        if not w or w in STOPWORDS or not re.search(r"[a-z]", w) or sc <= 0: continue
        if w not in out: out.append(w)
        if len(out) >= k: break
    return out

def occlusion_drop(text, words, base_q_conf, base_p_conf,
                   model, tokenizer, queue_encoder, priority_encoder, device):
    from pipeline.stage2a_transformer import transformer_predict
    reports = []
    for w in words:
        occ = re.sub(rf"\b{re.escape(w)}\b", " ", text, flags=re.IGNORECASE)
        occ = re.sub(r"\s+", " ", occ).strip()
        if not occ: continue
        pred = transformer_predict(occ, model, tokenizer, queue_encoder, priority_encoder, device)
        reports.append({
            "word": w,
            "queue_drop": max(0.0, base_q_conf - pred["dept_conf"]),
            "prio_drop":  max(0.0, base_p_conf - pred["priority_conf"]),
            "new_queue_conf": pred["dept_conf"],
            "new_prio_conf":  pred["priority_conf"],
        })
    return reports

def extract_competitor_kw(rag_chunks, competitor, k=8):
    texts = [r["chunk"]["raw_body"] for r in rag_chunks if r["chunk"]["dept"] == competitor]
    if not texts: return []
    words = [w for t in texts for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t.lower())]
    counts = Counter([w for w in words if w not in STOPWORDS and len(w) >= 4])
    return [w for w, _ in counts.most_common(k)]


def build_dept_radar(result):
    top3 = result.transformer_top3
    labels = [r["dept"] for r in top3]
    t_vals = [r["prob"] for r in top3]
    rag_scores = {}
    for r in result.rag_chunks:
        d = r["chunk"]["dept"]
        rag_scores[d] = rag_scores.get(d, 0) + max(0, r["ce_score"])
    total = sum(rag_scores.values()) or 1.0
    r_vals = [rag_scores.get(l, 0.0) / total for l in labels]
    lc = labels + labels[:1]
    tc = t_vals + t_vals[:1]
    rc = r_vals + r_vals[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=tc, theta=lc, fill="toself", name="Transformer",
        line=dict(color="#818cf8", width=2.5),
        fillcolor="rgba(99,102,241,0.18)",
        marker=dict(size=6, color="#6366f1", symbol="circle"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=rc, theta=lc, fill="toself", name="RAG vote",
        line=dict(color="#fb923c", width=2.5),
        fillcolor="rgba(251,146,60,0.12)",
        marker=dict(size=6, color="#f97316", symbol="diamond"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, max(tc + rc) * 1.2 or 1],
                gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)",
                tickfont=dict(size=9, color="#64748b"), tickformat=".0%",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Sans, system-ui"),
                linecolor="rgba(255,255,255,.1)", gridcolor="rgba(255,255,255,.06)",
            ),
            bgcolor="#0f172a",
        ),
        showlegend=True, margin=dict(l=28, r=28, t=56, b=24), height=380,
        title=dict(text="Department Signal Comparison",
                   font=dict(size=13, color="#e2e8f0", family="IBM Plex Sans, system-ui"), x=0.5),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="IBM Plex Sans, system-ui", color="#94a3b8"),
        legend=dict(font=dict(size=11, color="#94a3b8", family="IBM Plex Sans"),
                    bgcolor="rgba(30,41,59,.8)", bordercolor="rgba(255,255,255,.08)", borderwidth=1,
                    x=0.5, xanchor="center", y=-0.05, orientation="h"),
    )
    return fig

def build_prio_radar(result):
    probs = result.priority_probs if hasattr(result, "priority_probs") else {}
    labels = ["high", "medium", "low"]
    vals = [probs.get(l, 0.0) for l in labels]
    lc = labels + labels[:1]
    vc = vals + vals[:1]
    prio = result.priority.lower() if hasattr(result, "priority") else "medium"
    line_color = {"high": "#f87171", "medium": "#fbbf24", "low": "#4ade80"}.get(prio, "#818cf8")
    fill_color = {"high": "rgba(248,113,113,0.18)", "medium": "rgba(251,191,36,0.15)",
                  "low": "rgba(74,222,128,0.15)"}.get(prio, "rgba(99,102,241,0.18)")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=lc, fill="toself", name="Priority",
        line=dict(color=line_color, width=2.5), fillcolor=fill_color,
        marker=dict(size=8, color=line_color),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(vc) * 1.2 or 1],
                            gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)",
                            tickfont=dict(size=9, color="#64748b"), tickformat=".0%"),
            angularaxis=dict(tickfont=dict(size=13, color="#94a3b8", family="IBM Plex Sans, system-ui"),
                             linecolor="rgba(255,255,255,.1)", gridcolor="rgba(255,255,255,.06)"),
            bgcolor="#0f172a",
        ),
        showlegend=False, margin=dict(l=28, r=28, t=56, b=24), height=340,
        title=dict(text="Priority Signal Distribution",
                   font=dict(size=13, color="#e2e8f0", family="IBM Plex Sans, system-ui"), x=0.5),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="IBM Plex Sans, system-ui", color="#94a3b8"),
    )
    return fig


def render_routing_card(r):
    agree = r.transformer_dept == r.department
    prio  = r.priority.lower()
    pc    = PCOLOR.get(prio, _TEXT3)
    pbg   = PBG.get(prio, "rgba(99,102,241,.12)")
    pbd   = PBORDER.get(prio, "rgba(99,102,241,.3)")

    dept_box = _stat_box("Department", r.department, value_size="15px")
    prio_box = _stat_box("Priority",
        f'<span style="display:inline-flex!important;align-items:center!important;'
        f'gap:5px!important;padding:4px 12px!important;border-radius:99px!important;'
        f'background:{pbg}!important;color:{pc}!important;font-size:14px!important;'
        f'font-weight:700!important;border:1px solid {pbd}!important">'
        f'{PE.get(prio,"⚪")} {r.priority.capitalize()}</span>')
    conf_box = _stat_box("LLM Confidence", r.confidence.capitalize(),
                         sub=f"Transformer: {r.transformer_conf*100:.1f}%")

    grid = (
        f'<div style="display:grid!important;grid-template-columns:1fr 1fr 1fr!important;'
        f'gap:10px!important;margin-bottom:10px!important">'
        f'{dept_box}{prio_box}{conf_box}</div>'
    )

    if agree:
        agree_html = (
            f'<div style="display:flex!important;align-items:center!important;gap:8px!important;'
            f'padding:9px 14px!important;border-radius:8px!important;font-size:13px!important;'
            f'font-weight:500!important;background:rgba(74,222,128,.1)!important;'
            f'color:#4ade80!important;border:1px solid rgba(74,222,128,.25)!important;'
            f'margin-bottom:8px!important">'
            f'✅ Transformer and Groq agree</div>'
        )
    else:
        agree_html = (
            f'<div style="display:flex!important;align-items:center!important;gap:8px!important;'
            f'padding:9px 14px!important;border-radius:8px!important;font-size:13px!important;'
            f'font-weight:500!important;background:rgba(251,191,36,.1)!important;'
            f'color:#fbbf24!important;border:1px solid rgba(251,191,36,.25)!important;'
            f'margin-bottom:8px!important">'
            f'⚠️ Transformer predicted <strong style="color:{_TEXT1}!important">{r.transformer_dept}</strong>'
            f' — Groq overrides</div>'
        )

    reasoning = (
        f'<div style="background:{_BG1}!important;border-left:3px solid {_ACCENT}!important;'
        f'border-radius:0 6px 6px 0!important;padding:10px 14px!important;'
        f'font-size:13px!important;color:{_TEXT2}!important;line-height:1.6!important">'
        f'{r.reasoning}</div>'
    )

    return _wrap(grid + agree_html + reasoning)


def render_evidence_html(r):
    top3 = r.transformer_top3
    dept_bars = ""
    for item in top3:
        is_final = item["dept"] == r.department
        is_trans = item["dept"] == r.transformer_dept
        tag = ""
        if is_final:
            tag = _tag("FINAL", _ACCENT, "#fff")
        elif is_trans:
            tag = _tag("transformer", "rgba(99,102,241,.25)", "#a5b4fc")
        color = _ACCENT if is_final else "#818cf8" if is_trans else _BORDER
        dept_bars += _bar_row(item["dept"], item["prob"], color, tag)

    probs = r.priority_probs if hasattr(r, "priority_probs") else {}
    prio_bars = ""
    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
        is_pred = label == r.priority
        tag = _tag("predicted", _ACCENT, "#fff") if is_pred else ""
        prio_bars += _bar_row(f"{PE.get(label,'⚪')} {label.capitalize()}", prob,
                              PCOLOR.get(label, _TEXT4), tag)

    def _clean_section(s):
        """Turn SLIDING_WINDOW_0 / OVERVIEW / ROUTING etc into readable label."""
        s = re.sub(r"_\d+$", "", str(s))          # strip trailing _0, _1 …
        s = s.replace("_", " ").strip()
        return s.title() if s else "Excerpt"

    def _extract_body(raw, max_chars=220):
        """Strip boilerplate header and return the most informative excerpt."""
        text = str(raw)
        # Drop 'DEPARTMENT: X ROUTING SCOPE…' header lines
        text = re.sub(r"DEPARTMENT:.*?(?=\n|OVERVIEW|ROUTING|$)", "", text, flags=re.DOTALL|re.IGNORECASE)
        # Drop section banners like ==== or ----
        text = re.sub(r"[=\-]{4,}.*?[=\-]{4,}", " ", text, flags=re.DOTALL)
        # Drop remaining ALL-CAPS banner words at the start
        text = re.sub(r"^\s*(ROUTING SCOPE.*?\n|OVERVIEW.*?\n)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        # Take the first meaningful sentence block
        if len(text) > max_chars:
            # Try to cut at a sentence boundary
            cut = text.rfind(". ", 0, max_chars)
            text = text[:cut + 1] if cut > 60 else text[:max_chars]
        return text.replace("<","&lt;").replace(">","&gt;")

    rag_html = ""
    for i, item in enumerate(r.rag_chunks, 1):
        chunk = item["chunk"]
        score_color = "#4ade80" if item["ce_score"] > 0 else "#fbbf24"
        body = _extract_body(chunk["raw_body"])
        section_label = _clean_section(chunk["section"])
        rag_html += _card(
            f'<div style="display:flex!important;justify-content:space-between!important;'
            f'align-items:flex-start!important;margin-bottom:6px!important">'
            f'<div><div style="font-size:13px!important;font-weight:600!important;'
            f'color:{_TEXT1}!important">{i}. {chunk["dept"]}</div>'
            f'<div style="font-size:11px!important;color:{_TEXT4}!important;'
            f'font-family:monospace!important;margin-top:2px!important">{section_label}</div></div>'
            f'<span style="font-size:11px!important;font-weight:700!important;padding:2px 8px!important;'
            f'border-radius:99px!important;background:{_BG2}!important;color:{score_color}!important;'
            f'white-space:nowrap!important;flex-shrink:0!important;margin-left:8px!important">'
            f'CE {item["ce_score"]:.3f}</span></div>'
            f'<div style="font-size:12px!important;color:{_TEXT3}!important;line-height:1.6!important;'
            f'border-left:2px solid {_BORDER}!important;padding-left:10px!important">'
            f'{body}…</div>'
        )

    if r.priority_chunk:
        pchunk = r.priority_chunk["chunk"]
        body = _extract_body(pchunk["raw_body"])
        p_section_label = _clean_section(pchunk["section"])
        rag_html += _card(
            f'<div style="display:flex!important;justify-content:space-between!important;'
            f'align-items:flex-start!important;margin-bottom:6px!important">'
            f'<div><div style="font-size:13px!important;font-weight:600!important;'
            f'color:#a5b4fc!important">Priority Rule</div>'
            f'<div style="font-size:11px!important;color:{_TEXT4}!important;'
            f'font-family:monospace!important;margin-top:2px!important">{p_section_label}</div></div>'
            f'<span style="font-size:11px!important;font-weight:700!important;padding:2px 8px!important;'
            f'border-radius:99px!important;background:{_BG2}!important;color:{_TEXT3}!important;'
            f'white-space:nowrap!important;flex-shrink:0!important;margin-left:8px!important">'
            f'CE {r.priority_chunk["ce_score"]:.3f}</span></div>'
            f'<div style="font-size:12px!important;color:{_TEXT3}!important;line-height:1.6!important;'
            f'border-left:2px solid rgba(165,180,252,.4)!important;padding-left:10px!important">'
            f'{body}…</div>',
            border_color="rgba(165,180,252,.3)", bg=_BG1
        )

    dept_scores = {}
    for item in r.rag_chunks:
        d = item["chunk"]["dept"]
        dept_scores[d] = dept_scores.get(d, 0.0) + item["ce_score"]
    vote_html = ""
    if dept_scores:
        min_s = min(dept_scores.values())
        shifted = {d: s - min_s for d, s in dept_scores.items()}
        total = sum(shifted.values()) or 1.0
        for label, score in sorted(shifted.items(), key=lambda x: -x[1]):
            vote_html += _bar_row(label, score/total, _ACCENT)

    return _wrap(
        _section("Department Confidence") + dept_bars +
        _section("Priority Confidence") + prio_bars +
        _section("Retrieved Evidence") + rag_html +
        _section("RAG Vote Breakdown") + vote_html
    )


def render_explanation_html(r):
    def mono(t):
        return (f'<span style="display:inline-block!important;font-family:{_MONO_STACK}!important;'
                f'font-size:11px!important;background:{_BG2}!important;color:#a5b4fc!important;'
                f'padding:2px 6px!important;border-radius:4px!important;margin:1px!important">{t}</span>')

    top3_html = " ".join(mono(f'{x["dept"]} {x["prob"]:.1%}') for x in r.transformer_top3)
    rag_top = (
        mono(f'{r.rag_chunks[0]["chunk"]["dept"]} {r.rag_chunks[0]["ce_score"]:.3f}')
        if r.rag_chunks else f'<span style="color:{_TEXT4}!important">no results</span>'
    )

    rows = [
        ("Stage 1", "Groq Rewrite", f'<span style="color:#4ade80!important">✅ cleaned</span>'),
        ("Stage 2a", "Transformer top-3", top3_html),
        ("Stage 2b", "RAG top chunk", rag_top),
        ("Stage 3", "Final decision",
         mono(r.department) + mono(r.priority) + mono(f"conf={r.confidence}")),
    ]

    def th(t):
        return (f'<th style="text-align:left!important;font-size:11px!important;'
                f'font-weight:700!important;text-transform:uppercase!important;'
                f'letter-spacing:.08em!important;color:{_TEXT4}!important;'
                f'padding:6px 12px!important;border-bottom:1px solid {_BORDER}!important">{t}</th>')

    def td(t):
        return (f'<td style="padding:8px 12px!important;border-bottom:1px solid {_BG2}!important;'
                f'vertical-align:top!important;color:{_TEXT2}!important;font-size:13px!important">{t}</td>')

    table_rows = "".join(f'<tr>{td(mono(s))}{td(n)}{td(v)}</tr>' for s, n, v in rows)
    # If Stage 1 returned raw JSON (parse fallback), extract structured_body
    def _extract_cleaned(text):
        import json as _json, re as _re
        t = text.strip()
        # Strip markdown fences
        t = _re.sub(r"^```(?:json)?\s*", "", t, flags=_re.IGNORECASE).strip()
        t = _re.sub(r"\s*```$", "", t).strip()
        try:
            obj = _json.loads(t)
            if isinstance(obj, dict) and "structured_body" in obj:
                return obj["structured_body"]
        except Exception:
            pass
        return text
    cleaned = _extract_cleaned(r.cleaned_text).replace("<","&lt;").replace(">","&gt;")

    cleaned_block = (
        f'<div style="font-size:12px!important;color:{_TEXT2}!important;background:{_BG1}!important;'
        f'border-left:3px solid {_BORDER}!important;padding:8px 12px!important;'
        f'border-radius:0 6px 6px 0!important;margin-bottom:14px!important;'
        f'line-height:1.6!important">{cleaned}</div>'
    )
    table = (
        f'<table style="width:100%!important;border-collapse:collapse!important;'
        f'font-size:13px!important;background:{_BG1}!important;border-radius:8px!important;overflow:hidden!important">'
        f'<thead style="background:{_BG2}!important"><tr>{th("Stage")}{th("Step")}{th("Result")}</tr></thead>'
        f'<tbody>{table_rows}</tbody></table>'
    )

    return _wrap(_section("Cleaned Input") + cleaned_block + _section("Pipeline Trace") + table)


def render_attribution_html(queue_html, prio_html, occl_rows):
    occl = ""
    if occl_rows:
        def th(t):
            return (f'<th style="text-align:left!important;font-size:11px!important;'
                    f'font-weight:700!important;text-transform:uppercase!important;'
                    f'letter-spacing:.08em!important;color:{_TEXT4}!important;'
                    f'padding:6px 10px!important;border-bottom:1px solid {_BORDER}!important">{t}</th>')
        def td_o(t, color=None):
            c = color or _TEXT2
            return (f'<td style="padding:7px 10px!important;border-bottom:1px solid {_BG2}!important;'
                    f'font-size:12px!important;color:{c}!important">{t}</td>')
        def _row(rep):
            word  = rep["word"]
            qdrop = rep["queue_drop"]
            qnew  = rep["new_queue_conf"]
            pdrop = rep["prio_drop"]
            pnew  = rep["new_prio_conf"]
            w = (f'<span style="font-family:{_MONO_STACK}!important;background:{_BG2}!important;'
                 f'color:#a5b4fc!important;padding:1px 6px!important;border-radius:4px!important">{word}</span>')
            return (
                f'<tr>{td_o(w)}'
                + td_o(f"-{qdrop:.1%}", "#f87171")
                + td_o(f"{qnew:.1%}")
                + td_o(f"-{pdrop:.1%}", "#f87171")
                + td_o(f"{pnew:.1%}")
                + "</tr>"
            )
        rows = "".join(_row(rep) for rep in occl_rows)
        occl = (
            _section("Occlusion Check") +
            f'<table style="width:100%!important;border-collapse:collapse!important;'
            f'background:{_BG1}!important;border-radius:8px!important;overflow:hidden!important">'
            f'<thead style="background:{_BG2}!important"><tr>'
            f'{th("Word")}{th("Dept Δ")}{th("New dept")}{th("Prio Δ")}{th("New prio")}'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )
    else:
        occl = (_section("Occlusion Check") +
                f'<p style="color:{_TEXT4}!important;font-size:13px!important">'
                f'Install captum to enable IG + occlusion.</p>')
    return _wrap(queue_html + prio_html + occl)


def render_sensitivity_html(r, support_words):
    runner_up = r.transformer_top3[1]["dept"] if len(r.transformer_top3) > 1 else r.department
    margin = (r.transformer_top3[0]["prob"] - r.transformer_top3[1]["prob"]
              if len(r.transformer_top3) > 1 else 1.0)
    comp_kw = extract_competitor_kw(r.rag_chunks, runner_up)

    def pill(w, bg, color):
        return (f'<span style="display:inline-block!important;margin:3px 3px 3px 0!important;'
                f'padding:3px 10px!important;border-radius:99px!important;font-size:12px!important;'
                f'font-weight:500!important;background:{bg}!important;color:{color}!important;'
                f'font-family:{_MONO_STACK}!important">{w}</span>')

    support_pills = "".join(pill(w, "rgba(99,102,241,.25)", "#a5b4fc") for w in support_words) \
                    or f'<span style="color:{_TEXT4}!important;font-size:13px!important">None identified</span>'
    comp_pills = "".join(pill(w, "rgba(251,191,36,.15)", "#fbbf24") for w in comp_kw[:5]) \
                 or f'<span style="color:{_TEXT4}!important;font-size:13px!important">None found</span>'

    def note(t):
        return f'<div style="font-size:11px!important;color:{_TEXT4}!important;margin-top:8px!important">{t}</div>'

    stat_grid = (
        f'<div style="display:grid!important;grid-template-columns:1fr 1fr 1fr!important;'
        f'gap:12px!important;background:{_BG1}!important;border:1px solid {_BORDER}!important;'
        f'border-radius:8px!important;padding:14px!important;margin-bottom:10px!important">'
        f'<div><div style="font-size:11px!important;color:{_TEXT4}!important;margin-bottom:3px!important">Prediction</div>'
        f'<div style="font-weight:700!important;font-size:14px!important;color:{_TEXT1}!important">{r.department}</div></div>'
        f'<div><div style="font-size:11px!important;color:{_TEXT4}!important;margin-bottom:3px!important">Nearest competitor</div>'
        f'<div style="font-weight:700!important;font-size:14px!important;color:{_TEXT1}!important">{runner_up}</div></div>'
        f'<div><div style="font-size:11px!important;color:{_TEXT4}!important;margin-bottom:3px!important">Score margin</div>'
        f'<div style="font-weight:700!important;font-size:14px!important;color:{_TEXT1}!important">{margin:.3f}</div></div>'
        f'</div>'
    )
    support_box = (
        f'<div style="background:{_BG1}!important;border:1px solid {_BORDER}!important;'
        f'border-radius:8px!important;padding:14px!important;margin-bottom:10px!important">'
        f'{support_pills}{note("Removing these words would reduce transformer confidence")}</div>'
    )
    comp_box = (
        f'<div style="background:{_BG1}!important;border:1px solid {_BORDER}!important;'
        f'border-radius:8px!important;padding:14px!important;margin-bottom:10px!important">'
        f'{comp_pills}{note("Heuristic probe — not prescriptive")}</div>'
    )

    return _wrap(
        _section("Current Decision") + stat_grid +
        _section("Words Supporting Current Prediction") + support_box +
        _section(f"Terms That Would Push Toward {runner_up}") + comp_box
    )


def render_chat_reply(result):
    agree = result.transformer_dept == result.department
    agree_line = "\u2705 Transformer agrees." if agree else (
        "\u26a0\ufe0f Transformer said **" + result.transformer_dept + "** \u2014 Groq overrides."
    )
    prio_emoji = PE.get(result.priority.lower(), "\u26aa")
    return (
        "Ticket routed!\n\n"
        f"\U0001f3e2 **Department:** {result.department}\n"
        f"{prio_emoji} **Priority:** {result.priority.capitalize()}"
        f"  \u00b7  Transformer: `{result.transformer_conf*100:.1f}%`\n\n"
        f"{agree_line}\n\n"
        f"_{result.reasoning}_"
    )