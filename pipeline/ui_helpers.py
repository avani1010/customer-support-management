import re
import json as _json
import torch
import torch.nn as nn
import plotly.graph_objects as go
from collections import Counter

# ── Priority constants ─────────────────────────────────────────────────────
PE     = {"high": "🔴", "medium": "🟡", "low": "🟢"}
PCOLOR = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
PBG    = {"high": "rgba(239,68,68,.14)",  "medium": "rgba(245,158,11,.12)", "low": "rgba(34,197,94,.12)"}
PBORDER= {"high": "rgba(239,68,68,.4)",   "medium": "rgba(245,158,11,.35)", "low": "rgba(34,197,94,.35)"}
EMPTY  = "_(Submit a ticket above to see results)_"

STOPWORDS = {
    "the","a","an","and","or","but","if","then","when","of","to","in","on",
    "for","from","with","at","by","as","is","are","was","were","be","been",
    "it","this","that","we","i","you","they","he","she","our","my","your",
    "can","could","should","would","please","help","need","thanks","thank"
}

_F  = "Inter,system-ui,sans-serif"
_FM = "JetBrains Mono,Fira Code,monospace"

_BG0    = "#060d1a"
_BG1    = "#0d1829"
_BG2    = "#142035"
_BG3    = "#1a2840"
_BORDER = "#1e3154"
_BORDER2= "#243c64"
_T1 = "#f0f6ff"
_T2 = "#9db4d4"
_T3 = "#5a7a9e"
_T4 = "#3a5578"

_BLUE   = "#3b82f6"
_INDIGO = "#6366f1"
_CYAN   = "#06b6d4"
_GREEN  = "#22c55e"
_AMBER  = "#f59e0b"
_RED    = "#ef4444"
_PURPLE = "#a855f7"


def _w(inner, pad="8px 0"):
    return (f'<div style="background:{_BG0};color:{_T2};font-family:{_F};'
            f'font-size:13px;line-height:1.6;padding:{pad}">{inner}</div>')

def _mono(t, color="#93c5fd", bg=_BG2, size="11px"):
    return (f'<span style="font-family:{_FM};font-size:{size};background:{bg};'
            f'color:{color};padding:2px 7px;border-radius:4px;letter-spacing:.02em">{t}</span>')

def _badge(text, color, bg, size="10px"):
    return (f'<span style="font-size:{size};font-weight:700;padding:2px 8px;'
            f'border-radius:99px;background:{bg};color:{color};'
            f'letter-spacing:.06em;font-family:{_F};white-space:nowrap">{text}</span>')

def _label(text):
    return (f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.12em;color:{_T4};margin-bottom:6px;font-family:{_FM}">{text}</div>')

def _divider(label=""):
    inner = (f'<span style="padding:0 10px;background:{_BG0};color:{_T4};'
             f'font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.1em">{label}</span>'
             if label else "")
    return (f'<div style="display:flex;align-items:center;margin:18px 0 12px;gap:0">'
            f'<div style="flex:1;height:1px;background:{_BORDER}"></div>'
            f'{inner}'
            f'<div style="flex:1;height:1px;background:{_BORDER}"></div></div>')

def _card(inner, accent=None, pad="14px 16px", mb="10px"):
    border = f"2px solid {accent}" if accent else f"1px solid {_BORDER}"
    left   = f"border-left:{border};" if accent else f"border:{border};"
    return (f'<div style="background:{_BG1};{left}border-radius:8px;'
            f'padding:{pad};margin-bottom:{mb}">{inner}</div>')

def _stat(label, value, sub="", vc=_T1, vs="18px"):
    sub_h = (f'<div style="font-size:11px;color:{_T3};margin-top:5px;font-family:{_FM}">{sub}</div>'
             if sub else "")
    return (f'<div style="background:{_BG2};border:1px solid {_BORDER};border-radius:10px;padding:14px 16px">'
            f'{_label(label)}'
            f'<div style="font-size:{vs};font-weight:700;color:{vc};line-height:1.2;font-family:{_F}">{value}</div>'
            f'{sub_h}</div>')

def _bar(label, pct, color, tag="", height="7px"):
    filled = min(max(float(pct), 0), 1)
    return (f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<span style="font-size:12px;color:{_T2};font-family:{_FM}">{label}</span>{tag}</div>'
            f'<span style="font-size:12px;font-weight:700;color:{_T1};font-family:{_FM}">'
            f'{filled*100:.1f}%</span></div>'
            f'<div style="background:{_BORDER};border-radius:99px;height:{height};overflow:hidden">'
            f'<div style="width:{filled*100:.1f}%;height:100%;background:{color};border-radius:99px;'
            f'transition:width .4s ease"></div></div></div>')

def _tag(text, bg, color):
    return (f'<span style="margin-left:6px;font-size:10px;font-weight:700;padding:1px 6px;'
            f'border-radius:99px;background:{bg};color:{color};font-family:{_F}">{text}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# Flow node helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flow_node(number, label, status, detail="", color=_BLUE, skipped=False):
    if skipped:
        bg      = _BG1
        border  = f"1px solid {_BORDER}"
        n_bg    = _BG2
        n_color = _T4
        l_color = _T4
        icon    = "⊘"
    else:
        r_int   = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bg      = f"rgba({r_int[0]},{r_int[1]},{r_int[2]},.07)"
        border  = f"1px solid rgba({r_int[0]},{r_int[1]},{r_int[2]},.35)"
        n_bg    = color
        n_color = "#fff"
        l_color = _T1
        icon    = "✓"

    detail_h = (f'<div style="font-size:11px;color:{_T3};margin-top:3px;font-family:{_FM}">'
                f'{detail}</div>') if detail else ""

    return (f'<div style="background:{bg};border:{border};border-radius:8px;'
            f'padding:10px 12px;display:flex;align-items:flex-start;gap:10px">'
            f'<div style="width:22px;height:22px;border-radius:50%;background:{n_bg};'
            f'color:{n_color};font-size:11px;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;flex-shrink:0;font-family:{_FM}">{icon}</div>'
            f'<div style="min-width:0">'
            f'<div style="font-size:12px;font-weight:600;color:{l_color}">{label}</div>'
            f'{detail_h}</div></div>')


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Attribution utilities
# ─────────────────────────────────────────────────────────────────────────────

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

def _rgba_attr(x):
    x = max(-1.0, min(1.0, float(x)))
    if x >= 0:
        a = 0.15 + 0.50 * x
        return f"rgba(59,130,246,{a:.3f})"
    else:
        a = 0.15 + 0.50 * (-x)
        return f"rgba(239,68,68,{a:.3f})"

def render_token_heat(tokens_scores, title, subtitle=None, max_tokens=80):
    chips = "".join(
        f'<span style="display:inline-block;padding:2px 9px;border-radius:99px;'
        f'margin:2px 3px 2px 0;background:{_rgba_attr(sc)};font-family:{_FM};'
        f'font-size:12px;color:{_T1};border:1px solid rgba(255,255,255,.06)"'
        f'title="{sc:+.3f}">{tok.replace("<","&lt;").replace(">","&gt;")}</span>'
        for tok, sc in tokens_scores[:max_tokens]
    )
    sub = (f'<div style="color:{_T3};font-size:11px;margin-bottom:8px">{subtitle}</div>') if subtitle else ""
    return _w(_card(
        f'<div style="font-weight:600;font-size:13px;color:{_T1};margin-bottom:6px">{title}</div>'
        f'{sub}<div style="line-height:2.8">{chips}</div>'
    ))

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


# ─────────────────────────────────────────────────────────────────────────────
# Radar plots
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _radar_layout(fig, title, rmax):
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, rmax],
                            gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.05)",
                            tickfont=dict(size=9, color=_T4), tickformat=".0%"),
            angularaxis=dict(tickfont=dict(size=11, color=_T2, family=_F),
                             linecolor="rgba(255,255,255,.07)", gridcolor="rgba(255,255,255,.04)"),
            bgcolor=_BG0,
        ),
        showlegend=True, margin=dict(l=24, r=24, t=48, b=24), height=360,
        title=dict(text=title, font=dict(size=12, color=_T1, family=_F), x=0.5),
        paper_bgcolor=_BG0, plot_bgcolor=_BG0,
        font=dict(family=_F, color=_T2),
        legend=dict(font=dict(size=10, color=_T2), bgcolor="rgba(13,24,41,.9)",
                    bordercolor=_BORDER, borderwidth=1, x=0.5, xanchor="center",
                    y=-0.08, orientation="h"),
    )

def build_dept_radar(result):
    top3   = result.transformer_top3
    labels = [r["dept"] for r in top3]
    t_vals = [r["prob"] for r in top3]
    lc     = labels + labels[:1]
    tc     = t_vals + t_vals[:1]
    fig    = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=tc, theta=lc, fill="toself", name="Transformer",
        line=dict(color=_BLUE, width=2),
        fillcolor="rgba(59,130,246,0.14)",
        marker=dict(size=6, color=_BLUE),
    ))
    dept_skip  = getattr(result, "dept_rag_skipped", False)
    radial_max = max(tc) * 1.25 or 1
    if not dept_skip and result.rag_chunks:
        rag_scores = {}
        for r in result.rag_chunks:
            d = r["chunk"]["dept"]
            rag_scores[d] = rag_scores.get(d, 0) + max(0, r["ce_score"])
        total  = sum(rag_scores.values()) or 1.0
        r_vals = [rag_scores.get(l, 0.0) / total for l in labels]
        rc     = r_vals + r_vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=rc, theta=lc, fill="toself", name="RAG vote",
            line=dict(color=_AMBER, width=2),
            fillcolor="rgba(245,158,11,0.12)",
            marker=dict(size=6, color=_AMBER, symbol="diamond"),
        ))
        radial_max = max(tc + rc) * 1.25 or 1
    _radar_layout(fig, "Department signals", radial_max)
    return fig

def build_prio_radar(result):
    probs  = getattr(result, "priority_probs", {})
    labels = ["high", "medium", "low"]
    vals   = [probs.get(l, 0.0) for l in labels]
    lc, vc = labels + labels[:1], vals + vals[:1]
    prio   = getattr(result, "priority", "medium").lower()
    lc_map = {"high": _RED, "medium": _AMBER, "low": _GREEN}
    lcolor = lc_map.get(prio, _BLUE)
    r_int  = _hex_to_rgb(lcolor)
    fig    = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=lc, fill="toself", name="Priority probs",
        line=dict(color=lcolor, width=2),
        fillcolor=f"rgba({r_int[0]},{r_int[1]},{r_int[2]},0.18)",
        marker=dict(size=8, color=lcolor),
    ))
    _radar_layout(fig, "Priority signal", max(vc) * 1.25 or 1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_routing_card(r):
    prio       = r.priority.lower()
    pc         = PCOLOR.get(prio, _T2)
    pbg        = PBG.get(prio, "rgba(99,102,241,.12)")
    pbd        = PBORDER.get(prio, "rgba(99,102,241,.35)")
    agree      = r.transformer_dept == r.department
    dept_skip  = getattr(r, "dept_rag_skipped", False)
    rag_gap    = getattr(r, "rag_gap", 0.0)
    prio_probs = getattr(r, "priority_probs", {})
    prio_conf  = prio_probs.get(prio, 0.0)

    # Outcome header
    agree_badge = (
        f'<span style="font-size:10px;font-family:{_FM};color:{_GREEN};'
        f'background:rgba(34,197,94,.1);padding:2px 8px;border-radius:4px">'
        f'✓ transformer confirmed</span>'
        if agree else
        f'<span style="font-size:10px;font-family:{_FM};color:{_AMBER};'
        f'background:rgba(245,158,11,.1);padding:2px 8px;border-radius:4px">'
        f'↻ LLM override — was {r.transformer_dept}</span>'
    )

    outcome = (
        f'<div style="display:flex;align-items:stretch;gap:10px;margin-bottom:14px">'
        f'<div style="flex:1;background:{_BG2};border:1px solid {_BORDER};border-radius:10px;'
        f'padding:16px;min-width:0">'
        f'{_label("Routed to")}'
        f'<div style="font-size:16px;font-weight:700;color:{_T1};line-height:1.3;word-break:break-word">'
        f'{r.department}</div>'
        f'<div style="margin-top:8px">{agree_badge}</div>'
        f'</div>'
        f'<div style="width:128px;flex-shrink:0;background:{pbg};border:1px solid {pbd};'
        f'border-radius:10px;padding:16px;display:flex;flex-direction:column;'
        f'align-items:center;justify-content:center;text-align:center">'
        f'{_label("Priority")}'
        f'<div style="font-size:24px;font-weight:800;color:{pc};line-height:1">'
        f'{PE.get(prio,"⚪")} {r.priority.capitalize()}</div>'
        f'<div style="font-size:10px;font-family:{_FM};color:{_T3};margin-top:6px">'
        f'conf: {r.confidence}</div>'
        f'</div></div>'
    )

    # Gate banner
    if dept_skip:
        gate = (
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
            f'border-radius:8px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.2);'
            f'margin-bottom:10px">'
            f'<span style="font-size:18px">⚡</span>'
            f'<div><div style="font-size:12px;font-weight:700;color:{_GREEN}">FAST PATH</div>'
            f'<div style="font-size:10px;color:{_T3};font-family:{_FM}">'
            f'Transformer dept conf {r.transformer_conf*100:.1f}% ≥ 90% — dept RAG skipped. '
            f'Priority chunk always retrieved → Stage 3 LLM always runs.</div></div></div>'
        )
    else:
        unc   = rag_gap < 0.15
        gcol  = _RED if unc else _AMBER
        glbl  = "UNCERTAIN — both sources weak" if unc else "SLOW PATH"
        gsub  = (f"Dept conf {r.transformer_conf*100:.1f}% < 90% — dept RAG triggered"
                 + (f" | RAG gap {rag_gap:.3f} < 0.15 — LLM given low confidence signal" if unc else ""))
        gate  = (
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
            f'border-radius:8px;background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);'
            f'margin-bottom:10px">'
            f'<span style="font-size:18px">🔀</span>'
            f'<div><div style="font-size:12px;font-weight:700;color:{gcol}">{glbl}</div>'
            f'<div style="font-size:10px;color:{_T3};font-family:{_FM}">{gsub}</div></div></div>'
        )

    # Pipeline flow
    s1 = _flow_node(1, "Stage 1 — LLM rewrite", "done",
                    detail="Groq normalises raw ticket text", color=_BLUE)
    s2a = _flow_node(2, "Stage 2a — DistilBERT V6", "done",
                     detail=(f"dept: {r.transformer_dept} {r.transformer_conf*100:.0f}% "
                             f"| priority: {prio} {prio_conf*100:.0f}%"), color=_INDIGO)

    if dept_skip:
        s2b_dept = _flow_node("⊘", "Stage 2b — Dept RAG", "done",
                              detail="skipped — transformer confident", skipped=True)
    else:
        tc = r.rag_chunks[0] if r.rag_chunks else None
        s2b_dept = _flow_node(3, "Stage 2b — Dept RAG", "done",
                              detail=(f"top: {tc['chunk']['dept']} CE {tc['ce_score']:.3f}"
                                      if tc else "no results"), color=_CYAN)

    pchunk = getattr(r, "priority_chunk", None)
    s2b_prio = _flow_node(4, "Stage 2b — Priority chunk", "done",
                          detail=(f"section: {pchunk['chunk']['section']} CE {pchunk['ce_score']:.3f}"
                                  if pchunk else "not retrieved"), color=_PURPLE)

    agree_note = "confirms transformer" if agree else f"overrides → was {r.transformer_dept}"
    s3 = _flow_node(5, "Stage 3 — LLM \n(always runs for priority) \n(runs for dept when dept RAG invoked)", "done",
                    detail=f"dept: {r.department} ({agree_note}) | priority: {r.priority} ({r.confidence})",
                    color=_INDIGO)

    flow = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px">'
            f'<div>{s1}</div><div>{s2a}</div></div>'
            + gate +
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px">'
            f'<div>{s2b_dept}</div><div>{s2b_prio}</div></div>'
            f'<div style="text-align:center;color:{_BORDER2};font-size:12px;padding:2px 0">↓</div>'
            + s3
    )

    # Reasoning
    reasoning = (
        f'<div style="background:{_BG1};border-left:3px solid {_INDIGO};'
        f'border-radius:0 8px 8px 0;padding:12px 14px;margin-top:10px;'
        f'font-size:13px;color:{_T2};line-height:1.7">'
        f'{_label("LLM reasoning")}'
        f'{r.reasoning}</div>'
    )

    return _w(outcome + _divider("pipeline flow") + flow + reasoning)


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_evidence_html(r):
    dept_skip = getattr(r, "dept_rag_skipped", False)

    dept_bars = ""
    for item in r.transformer_top3:
        is_final = item["dept"] == r.department
        is_trans = item["dept"] == r.transformer_dept
        tag   = (_tag("FINAL",       _INDIGO, "#fff")              if is_final else
                 _tag("transformer", "rgba(99,102,241,.25)", "#93c5fd") if is_trans else "")
        color = _INDIGO if is_final else _BLUE if is_trans else _BORDER
        dept_bars += _bar(item["dept"], item["prob"], color, tag)

    prio_bars = ""
    probs = getattr(r, "priority_probs", {})
    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
        is_pred = label == r.priority
        tag   = _tag("selected", _INDIGO, "#fff") if is_pred else ""
        prio_bars += _bar(f'{PE.get(label,"⚪")} {label.capitalize()}', prob,
                          PCOLOR.get(label, _T3), tag)

    def _clean_sec(s):
        s = re.sub(r"_\d+$", "", str(s)).replace("_", " ").strip()
        return s.title() or "Excerpt"

    def _excerpt(raw, n=220):
        t = str(raw)
        t = re.sub(r"DEPARTMENT:.*?(?=\n|OVERVIEW|ROUTING|$)", "", t, flags=re.DOTALL|re.IGNORECASE)
        t = re.sub(r"[=\-]{4,}.*?[=\-]{4,}", " ", t, flags=re.DOTALL)
        t = re.sub(r"^\s*(ROUTING SCOPE.*?\n|OVERVIEW.*?\n)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > n:
            cut = t.rfind(". ", 0, n)
            t = t[:cut+1] if cut > 60 else t[:n]
        return t.replace("<","&lt;").replace(">","&gt;")

    if dept_skip:
        rag_dept_html = (
            f'<div style="text-align:center;padding:20px;color:{_T4};font-size:13px;'
            f'background:{_BG1};border:1px dashed {_BORDER};border-radius:8px;margin-bottom:8px">'
            f'⚡ Dept RAG skipped — transformer confidence ≥ 90% threshold</div>'
        )
    else:
        rag_dept_html = ""
        for i, item in enumerate(r.rag_chunks, 1):
            chunk   = item["chunk"]
            ce      = item["ce_score"]
            ce_c    = _GREEN if ce > 0.5 else _AMBER if ce > 0 else _T3
            body    = _excerpt(chunk["raw_body"])
            sec     = _clean_sec(chunk["section"])
            is_top  = (i == 1)
            rag_dept_html += _card(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                f'<div><span style="font-size:13px;font-weight:700;color:{_T1}">{i}. {chunk["dept"]}</span>'
                + (_badge("  TOP MATCH", _CYAN, "rgba(6,182,212,.15)", "9px") if is_top else "") +
                f'<div style="font-size:10px;color:{_T4};font-family:{_FM};margin-top:2px">{sec}</div></div>'
                f'<span style="font-family:{_FM};font-size:12px;font-weight:700;color:{ce_c}">'
                f'CE {ce:.3f}</span></div>'
                f'<div style="font-size:12px;color:{_T3};line-height:1.65;'
                f'border-left:2px solid {_BORDER};padding-left:10px">{body}…</div>',
                accent=_CYAN if is_top else None
            )

    pchunk = getattr(r, "priority_chunk", None)
    if pchunk:
        pc_body = _excerpt(pchunk["chunk"]["raw_body"])
        pc_sec  = _clean_sec(pchunk["chunk"]["section"])
        prio_chunk_html = _card(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<div><span style="font-size:13px;font-weight:700;color:{_PURPLE}">Priority criteria</span>'
            f'<div style="font-size:10px;color:{_T4};font-family:{_FM};margin-top:2px">'
            f'section: {pc_sec}</div></div>'
            f'<span style="font-family:{_FM};font-size:12px;font-weight:700;color:{_T2}">'
            f'CE {pchunk["ce_score"]:.3f}</span></div>'
            f'<div style="font-size:12px;color:{_T3};line-height:1.65;'
            f'border-left:2px solid rgba(168,85,247,.4);padding-left:10px">{pc_body}…</div>',
            accent="rgba(168,85,247,.4)"
        )
    else:
        prio_chunk_html = f'<div style="color:{_T4};font-size:13px;padding:8px">Not retrieved.</div>'

    vote_html = ""
    if not dept_skip:
        dept_scores = {}
        for item in r.rag_chunks:
            d = item["chunk"]["dept"]
            dept_scores[d] = dept_scores.get(d, 0.0) + item["ce_score"]
        if dept_scores:
            mn      = min(dept_scores.values())
            shifted = {d: s - mn for d, s in dept_scores.items()}
            total   = sum(shifted.values()) or 1.0
            for label, score in sorted(shifted.items(), key=lambda x: -x[1]):
                vote_html += _bar(label, score/total, _CYAN)

    return _w(
        _divider("transformer — dept confidence") + dept_bars +
        _divider("transformer — priority confidence") + prio_bars +
        _divider("dept rag chunks") + rag_dept_html +
        _divider("priority chunk (always retrieved)") + prio_chunk_html +
        ((_divider("rag vote breakdown") + vote_html) if vote_html else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_explanation_html(r):
    dept_skip  = getattr(r, "dept_rag_skipped", False)
    rag_gap    = getattr(r, "rag_gap", 0.0)
    prio_probs = getattr(r, "priority_probs", {})
    prio_conf  = prio_probs.get(r.priority.lower(), 0.0)
    agree      = r.transformer_dept == r.department

    def ok(t):  return f'<span style="color:{_GREEN}">{t}</span>'
    def warn(t): return f'<span style="color:{_AMBER}">{t}</span>'
    def info(t): return f'<span style="color:{_CYAN}">{t}</span>'
    def muted(t): return f'<span style="color:{_T4};font-size:10px">{t}</span>'

    def _clean_text(text):
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s*```$", "", t).strip()
        try:
            obj = _json.loads(t)
            if isinstance(obj, dict) and "structured_body" in obj:
                return obj["structured_body"]
        except Exception:
            pass
        return text

    cleaned = _clean_text(r.cleaned_text).replace("<","&lt;").replace(">","&gt;")
    cleaned_block = (
        f'<div style="font-size:12px;color:{_T2};background:{_BG1};'
        f'border-left:3px solid {_BORDER};padding:10px 14px;border-radius:0 8px 8px 0;'
        f'margin-bottom:16px;line-height:1.7">{cleaned}</div>'
    )

    top3_txt = "  ".join(_mono(f'{x["dept"]} {x["prob"]:.0%}') for x in r.transformer_top3)

    if dept_skip:
        rag_dept_txt = ok("⊘ skipped — transformer conf ≥ 90%")
        dept_src     = ok(f"✓ {r.department}") + " " + muted("(transformer direct)")
    else:
        tc = r.rag_chunks[0] if r.rag_chunks else None
        rag_dept_txt = (info(f'{tc["chunk"]["dept"]} CE={tc["ce_score"]:.3f}')
                        if tc else warn("no results"))
        if agree:
            dept_src = ok(f"✓ {r.department}") + " " + muted("(transformer + RAG agree)")
        else:
            dept_src = warn(f"↻ {r.department}") + " " + muted(f"(LLM overrides {r.transformer_dept})")

    pchunk = getattr(r, "priority_chunk", None)
    prio_chunk_txt = (info(f'section={pchunk["chunk"]["section"]} CE={pchunk["ce_score"]:.3f}')
                      if pchunk else warn("not retrieved"))

    gap_note = ""
    if not dept_skip and rag_gap > 0:
        gap_note = " " + muted(f"(RAG gap={rag_gap:.3f}" + (" ⚠ both uncertain" if rag_gap < 0.15 else "") + ")")

    rows = [
        ("Stage 1",  "Groq rewrite",       ok("✓ normalised")),
        ("Stage 2a", "Transformer dept",    _mono(f'{r.transformer_dept} {r.transformer_conf*100:.1f}%') + gap_note),
        ("Stage 2a", "Transformer priority", _mono(f'{r.priority} {prio_conf*100:.1f}%')),
        ("Stage 2a", "Top-3 candidates",    top3_txt),
        ("Stage 2b", "Dept RAG",            rag_dept_txt),
        ("Stage 2b", "Priority chunk",      prio_chunk_txt),
        ("Stage 3",  "Dept decision",       dept_src),
        ("Stage 3",  "Priority decision",   ok(f"✓ {r.priority}") + " " + muted(f"(LLM · conf={r.confidence})")),
    ]

    def th(t):
        return (f'<th style="text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:{_T4};padding:6px 12px;'
                f'border-bottom:1px solid {_BORDER};font-family:{_FM}">{t}</th>')
    def td(t, mono_f=False):
        return (f'<td style="padding:8px 12px;border-bottom:1px solid {_BG2};'
                f'vertical-align:top;color:{_T2};font-size:12px;'
                f'font-family:{_FM if mono_f else _F}">{t}</td>')

    table_rows = "".join(
        f'<tr>{td(_mono(s), mono_f=True)}{td(n)}{td(v)}</tr>'
        for s, n, v in rows
    )
    table = (
        f'<table style="width:100%;border-collapse:collapse;background:{_BG1};'
        f'border-radius:8px;overflow:hidden;border:1px solid {_BORDER}">'
        f'<thead style="background:{_BG2}"><tr>{th("Stage")}{th("Step")}{th("Result")}</tr></thead>'
        f'<tbody>{table_rows}</tbody></table>'
    )

    return _w(_divider("cleaned input") + cleaned_block + _divider("pipeline trace") + table)


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_attribution_html(queue_html, prio_html, occl_rows):
    if occl_rows:
        def th(t):
            return (f'<th style="text-align:left;font-size:10px;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:.1em;color:{_T4};'
                    f'padding:6px 10px;border-bottom:1px solid {_BORDER};font-family:{_FM}">{t}</th>')
        def td_o(t, color=None):
            return (f'<td style="padding:7px 10px;border-bottom:1px solid {_BG2};'
                    f'font-size:12px;color:{color or _T2}">{t}</td>')
        def _row(rep):
            w = (f'<span style="font-family:{_FM};background:{_BG2};color:#93c5fd;'
                 f'padding:1px 6px;border-radius:4px">{rep["word"]}</span>')
            return (f'<tr>{td_o(w)}'
                    + td_o(f'-{rep["queue_drop"]:.1%}', _RED)
                    + td_o(f'{rep["new_queue_conf"]:.1%}')
                    + td_o(f'-{rep["prio_drop"]:.1%}', _RED)
                    + td_o(f'{rep["new_prio_conf"]:.1%}') + '</tr>')
        rows = "".join(_row(rep) for rep in occl_rows)
        occl = (
                _divider("occlusion check") +
                f'<table style="width:100%;border-collapse:collapse;background:{_BG1};'
                f'border-radius:8px;overflow:hidden;border:1px solid {_BORDER}">'
                f'<thead style="background:{_BG2}"><tr>'
                f'{th("Word")}{th("Dept Δ")}{th("New dept")}{th("Prio Δ")}{th("New prio")}'
                f'</tr></thead><tbody>{rows}</tbody></table>'
        )
    else:
        occl = (
                _divider("occlusion check") +
                f'<div style="padding:16px;text-align:center;color:{_T4};font-size:13px;'
                f'background:{_BG1};border:1px dashed {_BORDER};border-radius:8px">'
                f'Install captum to enable IG + occlusion analysis.</div>'
        )
    return _w(queue_html + prio_html + occl)


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_sensitivity_html(r, support_words):
    dept_skip = getattr(r, "dept_rag_skipped", False)
    runner_up = r.transformer_top3[1]["dept"] if len(r.transformer_top3) > 1 else r.department
    margin    = (r.transformer_top3[0]["prob"] - r.transformer_top3[1]["prob"]
                 if len(r.transformer_top3) > 1 else 1.0)
    comp_kw   = extract_competitor_kw(r.rag_chunks, runner_up) if not dept_skip else []

    def pill(w, bg, tc):
        return (f'<span style="display:inline-block;margin:3px 3px 3px 0;padding:3px 10px;'
                f'border-radius:99px;font-size:12px;font-weight:500;background:{bg};'
                f'color:{tc};font-family:{_FM}">{w}</span>')

    support_pills = (
            "".join(pill(w, "rgba(59,130,246,.2)", "#93c5fd") for w in support_words)
            or f'<span style="color:{_T4};font-size:13px">None identified</span>'
    )
    comp_pills = (
            "".join(pill(w, "rgba(245,158,11,.15)", _AMBER) for w in comp_kw[:5])
            or f'<span style="color:{_T4};font-size:13px">'
               f'{"Dept RAG not invoked" if dept_skip else "None found"}</span>'
    )

    mc = _GREEN if margin > 0.3 else _AMBER if margin > 0.1 else _RED
    stats = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px">'
        f'{_stat("Predicted", r.department, vs="13px")}'
        f'{_stat("Nearest competitor", runner_up, vs="13px")}'
        f'{_stat("Score margin", f"{margin:.3f}", vc=mc, vs="18px")}'
        f'</div>'
    )

    return _w(
        _divider("decision stability") + stats +
        _divider("words supporting current prediction") +
        f'<div style="background:{_BG1};border:1px solid {_BORDER};border-radius:8px;'
        f'padding:14px;margin-bottom:10px">{support_pills}'
        f'<div style="font-size:11px;color:{_T4};margin-top:8px">'
        f'Removing these words would reduce transformer confidence in this department</div></div>'
        + _divider(f"terms pushing toward {runner_up}") +
        f'<div style="background:{_BG1};border:1px solid {_BORDER};border-radius:8px;'
        f'padding:14px">{comp_pills}'
        f'<div style="font-size:11px;color:{_T4};margin-top:8px">'
        f'Vocabulary from {runner_up} definition chunks — heuristic only</div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT REPLY
# ─────────────────────────────────────────────────────────────────────────────

def render_chat_reply(result):
    prio       = result.priority.lower()
    prio_emoji = PE.get(prio, "⚪")
    agree      = result.transformer_dept == result.department
    dept_skip  = getattr(result, "dept_rag_skipped", False)
    prio_probs = getattr(result, "priority_probs", {})
    prio_conf  = prio_probs.get(prio, 0.0)

    if dept_skip:
        path = "⚡ **Fast path** — transformer confident (≥90%), dept RAG skipped."
    elif agree:
        path = "🔀 **Slow path** — dept RAG confirmed transformer prediction."
    else:
        path = f"🔀 **Slow path** — LLM overrides transformer (`{result.transformer_dept}` → `{result.department}`)."

    return (
        "**Ticket routed.**\n\n"
        f"🏢 **Department:** {result.department}\n"
        f"{prio_emoji} **Priority:** {result.priority.capitalize()} · conf `{result.confidence}`\n\n"
        f"Transformer: dept `{result.transformer_conf*100:.1f}%` | priority `{prio_conf*100:.1f}%`\n\n"
        f"{path}\n\n"
        f"_{result.reasoning}_"
    )