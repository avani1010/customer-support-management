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
    lc = labels + labels[:1]
    tc = t_vals + t_vals[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=tc, theta=lc, fill="toself", name="Transformer",
        line=dict(color="#818cf8", width=2.5),
        fillcolor="rgba(99,102,241,0.18)",
        marker=dict(size=6, color="#6366f1", symbol="circle"),
    ))
    # Only add RAG trace if RAG was actually invoked
    if getattr(result, "rag_used", False) and result.rag_chunks:
        rag_scores = {}
        for r in result.rag_chunks:
            d = r["chunk"]["dept"]
            rag_scores[d] = rag_scores.get(d, 0) + max(0, r["ce_score"])
        total = sum(rag_scores.values()) or 1.0
        r_vals = [rag_scores.get(l, 0.0) / total for l in labels]
        rc = r_vals + r_vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=rc, theta=lc, fill="toself", name="RAG vote",
            line=dict(color="#fb923c", width=2.5),
            fillcolor="rgba(251,146,60,0.12)",
            marker=dict(size=6, color="#f97316", symbol="diamond"),
        ))
        radial_max = max(tc + rc) * 1.2 or 1
        title_text = "Department Signal Comparison"
    else:
        radial_max = max(tc) * 1.2 or 1
        title_text = "Department Signal Comparison (transformer only — RAG not invoked)"
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, radial_max],
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
        title=dict(text=title_text,
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
    agree    = r.transformer_dept == r.department
    prio     = r.priority.lower()
    pc       = PCOLOR.get(prio, _TEXT3)
    pbg      = PBG.get(prio, "rgba(99,102,241,.12)")
    pbd      = PBORDER.get(prio, "rgba(99,102,241,.3)")
    rag_used = getattr(r, "rag_used", True)
    rag_reason = getattr(r, "rag_reason", "")

    priority_probs = getattr(r, "priority_probs", {})
    priority_conf  = priority_probs.get(r.priority.lower(), 0.0)

    dept_box = _stat_box("Department", r.department, value_size="15px")
    prio_box = _stat_box("Priority",
                         f'<span style="display:inline-flex!important;align-items:center!important;'
                         f'gap:5px!important;padding:4px 12px!important;border-radius:99px!important;'
                         f'background:{pbg}!important;color:{pc}!important;font-size:14px!important;'
                         f'font-weight:700!important;border:1px solid {pbd}!important">'
                         f'{PE.get(prio,"⚪")} {r.priority.capitalize()}</span>')

    dept_tick  = "✓" if r.transformer_conf >= 0.85 else "✗"
    prio_tick  = "✓" if priority_conf >= 0.60 else "✗"
    dept_color = "#4ade80" if r.transformer_conf >= 0.85 else "#f87171"
    prio_color = "#4ade80" if priority_conf >= 0.60 else "#f87171"

    conf_box = _stat_box(
        "Transformer confidence",
        (
            f'<span style="font-size:12px!important;display:block!important;margin-bottom:3px!important">'
            f'<span style="color:{dept_color}!important;font-weight:700!important">{dept_tick} Dept: {r.transformer_conf*100:.1f}%</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{prio_color}!important;font-weight:700!important">{prio_tick} Priority: {priority_conf*100:.1f}%</span>'
            f'</span>'
        ),
        sub="LLM priority conf: " + r.confidence.capitalize() if rag_used else "RAG not invoked"
    )

    grid = (
        f'<div style="display:grid!important;grid-template-columns:1fr 1fr 1fr!important;'
        f'gap:10px!important;margin-bottom:10px!important">'
        f'{dept_box}{prio_box}{conf_box}</div>'
    )

    dept_passed = r.transformer_conf >= 0.85
    prio_passed = priority_conf >= 0.60

    if not rag_used:
        path_html = (
            f'<div style="display:flex!important;align-items:flex-start!important;gap:8px!important;'
            f'padding:9px 14px!important;border-radius:8px!important;font-size:13px!important;'
            f'font-weight:500!important;background:rgba(74,222,128,.1)!important;'
            f'color:#4ade80!important;border:1px solid rgba(74,222,128,.25)!important;'
            f'margin-bottom:8px!important;flex-direction:column!important">'
            f'<span>⚡ Fast path — transformer only (dept + priority both confident)</span>'
            f'<span style="font-size:11px!important;font-weight:400!important;'
            f'color:{_TEXT3}!important;font-family:monospace!important">{rag_reason}</span>'
            f'</div>'
        )
        agree_html = ""
    else:
        if dept_passed:
            dept_note = (
                f'<span style="color:#4ade80!important">✓ Dept {r.transformer_conf*100:.1f}% ≥ 85% — '
                f'transformer confident, used directly (dept retrieval skipped)</span>'
            )
        else:
            dept_note = (
                    f'<span style="color:#fbbf24!important">✗ Dept {r.transformer_conf*100:.1f}% &lt; 85% — '
                    f'transformer uncertain, dept decided by RAG CrossEncoder'
                    + (f' (overrides transformer: {r.transformer_dept})' if not agree else '')
                    + '</span>'
            )

        if prio_passed:
            prio_note = (
                f'<span style="color:#4ade80!important">✓ Priority {priority_conf*100:.1f}% ≥ 60% — '
                f'transformer confident, used directly (priority retrieval skipped)</span>'
            )
        else:
            prio_note = (
                f'<span style="color:#f87171!important">✗ Priority {priority_conf*100:.1f}% &lt; 60% — '
                f'transformer uncertain, LLM applied escalation criteria</span>'
            )

        if not dept_passed and not prio_passed:
            headline = "🔀 RAG invoked for dept · RAG, then LLM invoked for priority"
        elif not dept_passed:
            headline = "🔀 RAG invoked for dept · priority from transformer (confident)"
        else:
            headline = "🎯 Dept from transformer (confident) · RAG, then LLM invoked for priority only"

        path_html = (
            f'<div style="display:flex!important;align-items:flex-start!important;'
            f'padding:10px 14px!important;border-radius:8px!important;font-size:12px!important;'
            f'background:rgba(251,191,36,.08)!important;'
            f'border:1px solid rgba(251,191,36,.2)!important;'
            f'margin-bottom:8px!important;flex-direction:column!important;gap:6px!important">'
            f'<span style="font-size:13px!important;font-weight:600!important;color:#fbbf24!important">'
            f'{headline}</span>'
            f'<span style="font-family:monospace!important;color:{_TEXT3}!important;font-size:11px!important">'
            f'{rag_reason}</span>'
            f'<div style="display:flex!important;flex-direction:column!important;gap:3px!important;'
            f'margin-top:4px!important;font-size:12px!important">'
            f'<div>🏢 <strong style="color:{_TEXT2}!important">Dept:</strong> &nbsp;{dept_note}</div>'
            f'<div>🎯 <strong style="color:{_TEXT2}!important">Priority:</strong> {prio_note}</div>'
            f'</div>'
            f'</div>'
        )
        agree_html = ""

    reasoning_label = "LLM priority reasoning" if (rag_used and not prio_passed) else "Reasoning"
    reasoning = (
        f'<div style="background:{_BG1}!important;border-left:3px solid {_ACCENT}!important;'
        f'border-radius:0 6px 6px 0!important;padding:10px 14px!important;'
        f'font-size:13px!important;color:{_TEXT2}!important;line-height:1.6!important">'
        f'<span style="font-size:11px!important;color:{_TEXT4}!important;'
        f'font-weight:600!important;text-transform:uppercase!important;'
        f'letter-spacing:.06em!important">{reasoning_label}</span><br>'
        f'{r.reasoning}</div>'
    )

    return _wrap(grid + path_html + agree_html + reasoning)


def render_evidence_html(r):
    rag_used = getattr(r, "rag_used", True)
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
        s = re.sub(r"_\d+$", "", str(s))
        s = s.replace("_", " ").strip()
        return s.title() if s else "Excerpt"

    def _extract_body(raw, max_chars=220):
        text = str(raw)
        text = re.sub(r"DEPARTMENT:.*?(?=\n|OVERVIEW|ROUTING|$)", "", text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r"[=\-]{4,}.*?[=\-]{4,}", " ", text, flags=re.DOTALL)
        text = re.sub(r"^\s*(ROUTING SCOPE.*?\n|OVERVIEW.*?\n)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            cut = text.rfind(". ", 0, max_chars)
            text = text[:cut + 1] if cut > 60 else text[:max_chars]
        return text.replace("<","&lt;").replace(">","&gt;")

    if not rag_used:
        rag_html = (
            f'<div style="padding:16px!important;border-radius:8px!important;'
            f'background:{_BG1}!important;border:1px solid {_BORDER}!important;'
            f'color:{_TEXT4}!important;font-size:13px!important;text-align:center!important">'
            f'⚡ RAG not invoked — transformer confidence exceeded threshold</div>'
        )
        vote_html = ""
    else:
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

    rag_section = _section("Retrieved Evidence") + rag_html
    vote_section = (_section("RAG Vote Breakdown") + vote_html) if vote_html else ""

    return _wrap(
        _section("Department Confidence") + dept_bars +
        _section("Priority Confidence") + prio_bars +
        rag_section + vote_section
    )


def render_explanation_html(r):
    rag_used = getattr(r, "rag_used", True)
    priority_probs = getattr(r, "priority_probs", {})
    priority_conf  = priority_probs.get(r.priority.lower(), 0.0)

    def mono(t):
        return (f'<span style="display:inline-block!important;font-family:{_MONO_STACK}!important;'
                f'font-size:11px!important;background:{_BG2}!important;color:#a5b4fc!important;'
                f'padding:2px 6px!important;border-radius:4px!important;margin:1px!important">{t}</span>')

    def note(t, color=_TEXT4):
        return (f'<span style="font-size:10px!important;color:{color}!important;'
                f'font-style:italic!important;margin-left:4px!important">{t}</span>')

    top3_html = " ".join(mono(f'{x["dept"]} {x["prob"]:.1%}') for x in r.transformer_top3)

    if rag_used:
        dept_passed = r.transformer_conf >= 0.85
        prio_passed = priority_conf >= 0.60
        rag_top_chunk = r.rag_chunks[0] if r.rag_chunks else None
        rag_top = (
            mono(f'{rag_top_chunk["chunk"]["dept"]} CE={rag_top_chunk["ce_score"]:.3f}')
            + note("→ dept source", "#4ade80")
            if rag_top_chunk else
            f'<span style="color:{_TEXT4}!important">skipped — dept was confident</span>'
        )
        dept_decided = (
            mono(r.department) + note("transformer direct — dept confident, retrieval skipped", "#4ade80")
            if dept_passed else
            mono(r.department) + note("from CrossEncoder top chunk", "#4ade80")
        )
        prio_decided = (
            mono(r.priority) + note("transformer direct — priority confident, LLM skipped", "#4ade80")
            if prio_passed else
            mono(r.priority) + mono(f"conf={r.confidence}") + note(f"LLM applied escalation criteria (priority {priority_conf*100:.1f}% < 60% threshold)", "#fb923c")
        )
        prio_chunk_val = (
            mono(f'CE={r.priority_chunk["ce_score"]:.3f}') + note(f'section={r.priority_chunk["chunk"]["section"]}', "#fb923c")
            if r.priority_chunk else
            f'<span style="color:{_TEXT4}!important">skipped — priority was confident</span>'
        )
        rows = [
            ("Stage 1",  "Groq rewrite",        f'<span style="color:#4ade80!important">✅ cleaned</span>'),
            ("Stage 2a", "Transformer dept",     mono(f'{r.transformer_dept} {r.transformer_conf*100:.1f}%') + note("✓ confident — retrieval skipped" if dept_passed else "✗ uncertain — RAG triggered", "#4ade80" if dept_passed else "#f87171")),
            ("Stage 2a", "Transformer priority", mono(f'{r.priority} {priority_conf*100:.1f}%') + note("✓ confident — LLM skipped" if prio_passed else "✗ uncertain — LLM needed", "#4ade80" if prio_passed else "#f87171")),
            ("Stage 2b", "Dept RAG chunks",      rag_top),
            ("Stage 2b", "Priority chunk",       prio_chunk_val),
            ("Stage 3",  "Dept decision",        dept_decided),
            ("Stage 3",  "Priority decision",    prio_decided),
        ]
    else:
        rows = [
            ("Stage 1",  "Groq rewrite",        f'<span style="color:#4ade80!important">✅ cleaned</span>'),
            ("Stage 2a", "Transformer dept",     mono(f'{r.transformer_dept} {r.transformer_conf*100:.1f}%') + note("✓ confident — no RAG needed", "#4ade80")),
            ("Stage 2a", "Transformer priority", mono(f'{r.priority} {priority_conf*100:.1f}%') + note("✓ confident — no LLM needed", "#4ade80")),
            ("Stage 2b", "RAG",                  f'<span style="color:#4ade80!important">⚡ skipped — both thresholds met</span>'),
            ("Stage 3",  "LLM",                  f'<span style="color:#4ade80!important">⚡ skipped — transformer used directly</span>'),
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

    def _extract_cleaned(text):
        import json as _json, re as _re
        t = text.strip()
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
    rag_used = getattr(r, "rag_used", True)
    runner_up = r.transformer_top3[1]["dept"] if len(r.transformer_top3) > 1 else r.department
    margin = (r.transformer_top3[0]["prob"] - r.transformer_top3[1]["prob"]
              if len(r.transformer_top3) > 1 else 1.0)
    comp_kw = extract_competitor_kw(r.rag_chunks, runner_up) if rag_used else []

    def pill(w, bg, color):
        return (f'<span style="display:inline-block!important;margin:3px 3px 3px 0!important;'
                f'padding:3px 10px!important;border-radius:99px!important;font-size:12px!important;'
                f'font-weight:500!important;background:{bg}!important;color:{color}!important;'
                f'font-family:{_MONO_STACK}!important">{w}</span>')

    support_pills = "".join(pill(w, "rgba(99,102,241,.25)", "#a5b4fc") for w in support_words) \
                    or f'<span style="color:{_TEXT4}!important;font-size:13px!important">None identified</span>'

    if rag_used and comp_kw:
        comp_pills = "".join(pill(w, "rgba(251,191,36,.15)", "#fbbf24") for w in comp_kw[:5])
    else:
        comp_pills = (
            f'<span style="color:{_TEXT4}!important;font-size:13px!important">'
            f'{"RAG not invoked — no competitor keywords available" if not rag_used else "None found"}'
            f'</span>'
        )

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
        f'{comp_pills}{note("Heuristic probe — not prescriptive" if rag_used else "Only available when RAG is invoked")}</div>'
    )

    return _wrap(
        _section("Current Decision") + stat_grid +
        _section("Words Supporting Current Prediction") + support_box +
        _section(f"Terms That Would Push Toward {runner_up}") + comp_box
    )


def render_chat_reply(result):
    rag_used       = getattr(result, "rag_used", True)
    rag_reason     = getattr(result, "rag_reason", "")
    prio_emoji     = PE.get(result.priority.lower(), "\u26aa")
    priority_probs = getattr(result, "priority_probs", {})
    priority_conf  = priority_probs.get(result.priority.lower(), 0.0)

    dept_tick = "\u2713" if result.transformer_conf >= 0.85 else "\u2717"
    prio_tick = "\u2713" if priority_conf >= 0.60 else "\u2717"

    conf_line = (
        f"Dept: `{result.transformer_conf*100:.1f}%` {dept_tick}  \u00b7  "
        f"Priority: `{priority_conf*100:.1f}%` {prio_tick}"
    )

    if not rag_used:
        body = (
            "Ticket routed!\n\n"
            f"\U0001f3e2 **Department:** {result.department}\n"
            f"{prio_emoji} **Priority:** {result.priority.capitalize()}\n\n"
            f"{conf_line}\n"
            f"`{rag_reason}`\n\n"
            f"\u26a1 **Fast path** — transformer confident on both dept and priority, RAG + LLM skipped.\n\n"
            f"_{result.reasoning}_"
        )
    else:
        agree     = result.transformer_dept == result.department
        dept_line = (
            f"\u2705 **Dept:** CrossEncoder top chunk agrees with transformer \u2192 `{result.department}`"
            if agree else
            f"\u26a0\ufe0f **Dept:** CrossEncoder top chunk overrides transformer "
            f"(`{result.transformer_dept}` \u2192 `{result.department}`)"
        )
        body = (
            "Ticket routed!\n\n"
            f"\U0001f3e2 **Department:** {result.department}\n"
            f"{prio_emoji} **Priority:** {result.priority.capitalize()}\n\n"
            f"{conf_line}\n"
            f"`{rag_reason}`\n\n"
            f"\U0001f500 **Slow path** — RAG invoked:\n"
            f"{dept_line}\n"
            f"\U0001f3af **Priority:** transformer uncertain (`{priority_conf*100:.1f}%`) "
            f"\u2014 LLM read escalation criteria \u2192 `{result.priority}` (conf: `{result.confidence}`)\n\n"
            f"_{result.reasoning}_"
        )
    return body