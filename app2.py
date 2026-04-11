import os, sys, warnings, traceback, html as _html, re as _re
warnings.filterwarnings("ignore")

import torch
import gradio as gr
from dotenv import load_dotenv
from groq import Groq

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))
HF_TOKEN   = os.getenv("HF_TOKEN")
GROQ_TOKEN = os.getenv("GROQ_API_KEY_3")

from pipeline.stage2a_transformer import load_transformer
from pipeline.stage2b_retriever   import load_rag_artifacts
from pipeline.router               import route_ticket
from pipeline.ui_helpers import (
    QueueWrapper, PriorityWrapper, EMPTY,
    render_token_heat, ig_explain, pick_top_support_words, occlusion_drop,
    build_dept_radar, build_prio_radar,
    render_routing_card, render_evidence_html, render_explanation_html,
    render_sensitivity_html, render_attribution_html, render_chat_reply,
)

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
groq_client = Groq(api_key=GROQ_TOKEN)

model, tokenizer, queue_encoder, priority_encoder = load_transformer(
    "Nethra19/multitask-ticket-model-v6", HF_TOKEN, device
)
faiss_index, bm25, all_chunks, embedder, cross_encoder, priority_index, priority_chunks = load_rag_artifacts(
    "Nethra19/rag-index-v6", HF_TOKEN
)

queue_id_to_label    = {i: l for i, l in enumerate(queue_encoder.classes_)}
priority_id_to_label = {i: l for i, l in enumerate(priority_encoder.classes_)}
queue_ig_model    = QueueWrapper(model).to(device).eval()
priority_ig_model = PriorityWrapper(model).to(device).eval()
print(f"✓ All components ready on {device}")

# ── Design tokens ──────────────────────────────────────────────────────────
BG0    = "#060d1a"
BG1    = "#0d1829"
BG2    = "#142035"
BORDER = "#1e3154"
T1     = "#f0f6ff"
T2     = "#9db4d4"
T3     = "#5a7a9e"
T4     = "#3a5578"
BLUE   = "#3b82f6"
INDIGO = "#6366f1"
GREEN  = "#22c55e"
AMBER  = "#f59e0b"
RED    = "#ef4444"
PURPLE = "#a855f7"
FONT   = "Inter,system-ui,sans-serif"
MONO   = "JetBrains Mono,Fira Code,monospace"

EXAMPLES = [
    "I was charged twice for my subscription last month. Please refund the duplicate payment.",
    "My internet isn't working and I can't log in — urgent deadline today.",
    "We need to update payroll records for three new hires this week.",
    "The checkout page keeps crashing when customers try to pay with Visa.",
    "Can you tell me your standard service hours and how to escalate?",
    "Our Salesforce CRM has been down since 9am, sales team is blocked.",
]

EMPTY_HTML = (
    f"<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;"
    f"height:140px;color:{T4};font-size:13px;font-family:{FONT};gap:8px;"
    f"background:{BG1};border:1px dashed {BORDER};border-radius:8px'>"
    f"<div style='font-size:28px;opacity:.4'>🎫</div>"
    f"<div>Submit a ticket to see routing results</div>"
    f"</div>"
)

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

.gradio-container {{
    background: {BG0} !important;
    font-family: {FONT} !important;
}}

/* ── App header ── */
.im-header {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}}
.im-header-icon {{
    width: 44px; height: 44px;
    background: linear-gradient(135deg, {INDIGO}, {BLUE});
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}}
.im-header-title {{
    font-size: 16px; font-weight: 700; color: {T1};
    letter-spacing: -.02em;
}}
.im-header-sub {{
    font-size: 11px; color: {T3}; font-family: {MONO};
    margin-top: 3px; letter-spacing: .03em;
}}
.im-header-live {{
    margin-left: auto; flex-shrink: 0;
    background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.3);
    border-radius: 99px; padding: 5px 14px;
    font-size: 11px; font-weight: 700; color: {GREEN};
    font-family: {MONO}; letter-spacing: .06em;
    display: flex; align-items: center; gap: 6px;
}}
.im-live-dot {{
    width: 7px; height: 7px; background: {GREEN};
    border-radius: 50%; animation: pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}

/* ── Left panel ── */
.im-left {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}
.im-left-header {{
    background: {BG2};
    padding: 12px 18px;
    border-bottom: 1px solid {BORDER};
    font-size: 11px; font-weight: 700; color: {T3};
    text-transform: uppercase; letter-spacing: .1em;
    font-family: {MONO};
    display: flex; align-items: center; gap: 8px;
}}

/* ── Chat messages ── */
.im-messages {{
    height: 300px;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    scroll-behavior: smooth;
}}
.im-messages::-webkit-scrollbar {{ width: 3px; }}
.im-messages::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}

.im-msg-user {{
    align-self: flex-end; max-width: 82%;
    background: linear-gradient(135deg, {INDIGO}, {BLUE});
    color: #fff;
    border-radius: 12px 12px 3px 12px;
    padding: 10px 14px; font-size: 13px; line-height: 1.55;
}}
.im-msg-bot {{
    align-self: flex-start; max-width: 90%;
    background: {BG2}; color: {T2};
    border: 1px solid {BORDER};
    border-radius: 3px 12px 12px 12px;
    padding: 10px 14px; font-size: 13px; line-height: 1.65;
}}
.im-msg-bot strong {{ color: {T1}; }}
.im-msg-bot code {{
    background: {BG0}; color: #93c5fd;
    padding: 1px 5px; border-radius: 3px;
    font-family: {MONO}; font-size: 11px;
}}
.im-empty-state {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 8px; color: {T4}; font-size: 13px; padding: 20px 0;
}}
.im-empty-icon {{ font-size: 32px; opacity: .3; }}

/* ── Input area ── */
.im-input-area {{
    padding: 12px 16px;
    border-top: 1px solid {BORDER};
    background: {BG0};
}}

/* ── Example tickets ── */
.im-examples-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: {T4}; font-family: {MONO};
    padding: 14px 16px 8px;
}}
.im-ex-btn {{
    width: 100%; text-align: left;
    background: {BG0}; color: {T3};
    border: none; border-top: 1px solid {BORDER};
    padding: 10px 16px; font-size: 12px; font-family: {FONT};
    cursor: pointer; transition: background .12s, color .12s;
    line-height: 1.4;
}}
.im-ex-btn:hover {{ background: {BG2}; color: {T1}; }}

/* ── Tab strip ── */
#im-tabs {{
    display: flex; gap: 4px;
    background: {BG2}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 5px;
    margin-bottom: 10px; flex-wrap: wrap;
}}
#im-tabs button {{
    border-radius: 7px; padding: 7px 13px;
    font-size: 12px; font-weight: 600;
    font-family: {FONT}; white-space: nowrap;
    background: transparent; color: {T3};
    border: 1px solid transparent;
    transition: all .15s;
}}
#im-tabs button:hover {{ background: {BG0}; color: {T2}; }}
#im-tabs button.primary {{
    background: linear-gradient(135deg, {INDIGO}, {BLUE});
    color: #fff; border-color: rgba(99,102,241,.4);
    box-shadow: 0 2px 8px rgba(99,102,241,.3);
}}

/* ── Textbox overrides ── */
textarea, input[type=text] {{
    background: {BG1} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {T1} !important;
    font-family: {FONT} !important;
    font-size: 13px !important;
}}
textarea:focus, input[type=text]:focus {{
    border-color: {INDIGO} !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
}}

/* ── Buttons ── */
button.primary {{
    background: linear-gradient(135deg, {INDIGO}, {BLUE}) !important;
    border: none !important; color: #fff !important;
    font-family: {FONT} !important; font-weight: 600 !important;
    border-radius: 8px !important;
    transition: opacity .15s !important;
}}
button.primary:hover {{ opacity: .88 !important; }}
button.secondary {{
    background: {BG2} !important; color: {T2} !important;
    border: 1px solid {BORDER} !important;
    font-family: {FONT} !important;
    border-radius: 8px !important;
}}
button.secondary:hover {{ background: {BG1} !important; color: {T1} !important; }}

/* right panel */
.im-right {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px;
}}
"""

TAB_KEYS = ["evidence", "explanation", "attribution", "radar", "sensitivity"]

def _md(text):
    safe = _html.escape(str(text))
    safe = safe.replace("\n\n", "<br><br>").replace("\n", "<br>")
    safe = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = _re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    return safe

def _render_messages(history):
    if not history:
        return (
            f"<div class='im-empty-state'>"
            f"<div class='im-empty-icon'>🎫</div>"
            f"<div style='color:{T4}'>Submit a ticket to see the routing decision</div>"
            f"</div>"
        )
    parts = []
    for turn in history:
        role    = turn.get("role", "")
        content = _md(turn.get("content", ""))
        cls     = "im-msg-user" if role == "user" else "im-msg-bot"
        parts.append(f"<div class='{cls}'>{content}</div>")
    scroll = ("<script>setTimeout(()=>{"
              "var el=document.querySelector('.im-messages');"
              "if(el)el.scrollTop=el.scrollHeight;},60)</script>")
    return "".join(parts) + scroll

def _build_chat_panel(history):
    inner = _render_messages(history)
    return (
        f'<div class="im-left">'
        f'<div class="im-left-header">'
        f'<div class="im-live-dot"></div>'
        f'Incident queue'
        f'</div>'
        f'<div class="im-messages">{inner}</div>'
        f'</div>'
    )

_EMPTY_CHAT = _build_chat_panel([])

def _set_tab(selected):
    b = lambda k: gr.update(variant="primary" if selected == k else "secondary")
    v = lambda k: gr.update(visible=(selected == k))
    return (
        selected,
        b("evidence"), b("explanation"), b("attribution"), b("radar"), b("sensitivity"),
        v("evidence"), v("explanation"), v("attribution"), v("radar"), v("sensitivity"),
    )

def process_ticket_ui(user_message, history):
    history = list(history or [])
    if not user_message or not user_message.strip():
        return (history, _build_chat_panel(history),
                EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, None, None, EMPTY_HTML)

    history.append({"role": "user", "content": user_message.strip()})

    try:
        result = route_ticket(
            user_message.strip(),
            groq_client,
            model, tokenizer, queue_encoder, priority_encoder, device,
            embedder,
            faiss_index, bm25, all_chunks, cross_encoder,
            priority_index, priority_chunks
        )
    except Exception as e:
        traceback.print_exc()
        history.append({"role": "assistant", "content": f"⚠️ Pipeline error: {e}"})
        err = (f"<div style='color:{RED};padding:16px;font-size:13px;"
               f"background:{BG1};border:1px solid rgba(239,68,68,.3);border-radius:8px'>⚠️ {e}</div>")
        return (history, _build_chat_panel(history), err, err, err, err, None, None, err)

    history.append({"role": "assistant", "content": render_chat_reply(result)})

    # Attribution
    q_idx = next((i for i, l in queue_id_to_label.items() if l == result.transformer_dept), None)
    support_words, queue_html = [], ""
    if q_idx is not None:
        q_attr, q_err = ig_explain(result.cleaned_text, queue_ig_model, q_idx, tokenizer, device)
        queue_html = (
            render_token_heat(q_attr["ordered"],
                              f"Department attribution → {result.transformer_dept}",
                              "Blue = supports this dept · Red = pushes against (Integrated Gradients)")
            if q_attr else
            render_token_heat([], "IG unavailable", subtitle=q_err)
        )
        if q_attr:
            support_words = pick_top_support_words(q_attr["sorted"], k=3)

    p_idx = next((i for i, l in priority_id_to_label.items() if l == result.priority), None)
    prio_html = ""
    if p_idx is not None:
        p_attr, p_err = ig_explain(result.cleaned_text, priority_ig_model, p_idx, tokenizer, device)
        prio_html = (
            render_token_heat(p_attr["ordered"],
                              f"Priority attribution → {result.priority}",
                              "Same colour logic as department attribution above.")
            if p_attr else
            render_token_heat([], "IG unavailable", subtitle=p_err)
        )

    occl_rows = occlusion_drop(
        result.cleaned_text, support_words,
        result.transformer_conf, 0.5,
        model, tokenizer, queue_encoder, priority_encoder, device
    )

    return (
        history,
        _build_chat_panel(history),
        render_routing_card(result),
        render_evidence_html(result),
        render_explanation_html(result),
        render_attribution_html(queue_html, prio_html, occl_rows),
        build_dept_radar(result),
        build_prio_radar(result),
        render_sensitivity_html(result, support_words),
    )

def clear_ui():
    return ([], _EMPTY_CHAT, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, None, None, EMPTY_HTML)


with gr.Blocks(css=CSS, title="Incident Router") as demo:

    chat_history = gr.State([])

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="im-header">
      <div class="im-header-icon">🎫</div>
      <div>
        <div class="im-header-title">Incident Ticket Router</div>
        <div class="im-header-sub">DistilBERT V6 · MiniLM + BGE Reranker · Llama-3.3-70b</div>
      </div>
      <div class="im-header-live">
        <span class="im-live-dot"></span> LIVE
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: ticket input + chat ──────────────────────────────────────
        with gr.Column(scale=4, min_width=300):

            chat_display = gr.HTML(value=_EMPTY_CHAT)

            with gr.Column():
                msg_input = gr.Textbox(
                    placeholder="Paste or type the incident ticket here…",
                    lines=3, max_lines=8, show_label=False,
                    container=False,
                )
                with gr.Row():
                    submit_btn = gr.Button("Route ticket →", variant="primary", scale=3)
                    clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

            # Example tickets
            gr.HTML(f'<div class="im-examples-label">Example tickets</div>')
            example_btns = []
            for ex in EXAMPLES:
                btn = gr.Button(
                    ex[:70] + ("…" if len(ex) > 70 else ""),
                    variant="secondary",
                    elem_classes=["im-ex-btn"],
                    )
                example_btns.append((btn, ex))

        # ── RIGHT: routing result + analysis tabs ──────────────────────────
        with gr.Column(scale=6, min_width=420):

            routing_card = gr.HTML(value=EMPTY_HTML)
            active_panel = gr.State("evidence")

            with gr.Row(elem_id="im-tabs"):
                btn_evidence    = gr.Button("📊 Evidence",    variant="primary")
                btn_explanation = gr.Button("🔍 Trace",       variant="secondary")
                btn_attribution = gr.Button("🧠 Attribution", variant="secondary")
                btn_radar       = gr.Button("📡 Radar",       variant="secondary")
                btn_sensitivity = gr.Button("⚖️ Sensitivity", variant="secondary")

            panel_evidence    = gr.Column(visible=True)
            panel_explanation = gr.Column(visible=False)
            panel_attribution = gr.Column(visible=False)
            panel_radar       = gr.Column(visible=False)
            panel_sensitivity = gr.Column(visible=False)

            with panel_evidence:
                evidence_html = gr.HTML(value=EMPTY_HTML)
            with panel_explanation:
                explanation_html = gr.HTML(value=EMPTY_HTML)
            with panel_attribution:
                attribution_html = gr.HTML(value=EMPTY_HTML)
            with panel_radar:
                radar_dept = gr.Plot(value=None)
                radar_prio = gr.Plot(value=None)
            with panel_sensitivity:
                sensitivity_html = gr.HTML(value=EMPTY_HTML)

            tab_outs = [
                active_panel,
                btn_evidence, btn_explanation, btn_attribution, btn_radar, btn_sensitivity,
                panel_evidence, panel_explanation, panel_attribution, panel_radar, panel_sensitivity,
            ]
            btn_evidence.click(   fn=lambda: _set_tab("evidence"),    inputs=[], outputs=tab_outs)
            btn_explanation.click(fn=lambda: _set_tab("explanation"), inputs=[], outputs=tab_outs)
            btn_attribution.click(fn=lambda: _set_tab("attribution"), inputs=[], outputs=tab_outs)
            btn_radar.click(      fn=lambda: _set_tab("radar"),       inputs=[], outputs=tab_outs)
            btn_sensitivity.click(fn=lambda: _set_tab("sensitivity"), inputs=[], outputs=tab_outs)

    OUTS = [
        chat_history, chat_display,
        routing_card,
        evidence_html, explanation_html, attribution_html,
        radar_dept, radar_prio,
        sensitivity_html,
    ]

    submit_btn.click(fn=process_ticket_ui, inputs=[msg_input, chat_history], outputs=OUTS
                     ).then(fn=lambda: "", inputs=[], outputs=msg_input)
    msg_input.submit(fn=process_ticket_ui, inputs=[msg_input, chat_history], outputs=OUTS
                     ).then(fn=lambda: "", inputs=[], outputs=msg_input)
    clear_btn.click(fn=clear_ui, inputs=[], outputs=OUTS
                    ).then(fn=lambda: "", inputs=[], outputs=msg_input)

    for _btn, _txt in example_btns:
        _btn.click(fn=lambda t=_txt: t, inputs=[], outputs=msg_input)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, share=False)