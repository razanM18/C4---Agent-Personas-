"""CedarSun ReAct agent (C4) with a persona switch and a live view of the reasoning.

    python app_c4.py        # then open the address it prints (usually http://127.0.0.1:7861)
"""
import html
import json
import os
from pathlib import Path

import gradio as gr

import react_agent as ra

NL = chr(10)
LOGO = str(Path(__file__).with_name("logo.svg"))

EXAMPLES = ["How many solar panels do I need? I have a fridge, 4 lights, a TV and 2 ACs in Beirut.",
            "What warranty do you give on the battery and the inverter?",
            "How much does a 5 kW system cost?",
            "I'd like a free site survey. My name is Rami and my email is rami@example.com",
            "My battery is making a burning smell and there is smoke.",
            "kifak, kam sene el garantie 3ala el battery?"]

PERSONA_BLURB = {
    "friendly": ("Warm and chatty", "Short, upbeat answers with everyday words. Slightly higher temperature."),
    "expert": ("Precise and technical", "Exact figures and tidy structure. Temperature near zero, so answers are repeatable."),
    "cautious": ("Careful and safety-first", "Checks facts twice and admits what it does not know. Low temperature and a repetition penalty."),
}
PERSONA_ICON = {"friendly": "😊", "expert": "🧑‍🔧", "cautious": "🛡️"}

CSS = """
:root { --sun:#F08A1C; --sun-soft:#FFF4E3; --navy:#173F73; --ink:#1f2a37; --muted:#6b7686; --line:#e8ecf2; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
body, .gradio-container { background: #fbfaf7 !important; }
footer { display: none !important; }

#hero { display:flex; align-items:center; gap:16px; padding:18px 22px; margin-bottom:14px;
        background: linear-gradient(120deg, #173F73 0%, #2A6BB5 70%); border-radius:18px; color:#fff; }
#hero img { width:56px; height:56px; background:#fff; border-radius:14px; padding:6px; }
#hero h1 { margin:0; font-size:1.35rem; font-weight:700; color:#fff; }
#hero p { margin:2px 0 0; font-size:.92rem; color:#dbe7f7; }

#persona .wrap { display:grid !important; grid-template-columns:1fr; gap:8px; }
#persona label { border:1.5px solid var(--line) !important; border-radius:14px !important; padding:11px 14px !important;
                 background:#fff !important; cursor:pointer; transition:all .15s; }
#persona label:hover { border-color:var(--sun) !important; }
#persona label.selected, #persona label:has(input:checked) { border-color:var(--sun) !important; background:var(--sun-soft) !important; }
#persona label span { font-weight:600; color:var(--ink); }

.pcard { background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px 16px; color:var(--ink); font-size:.92rem; line-height:1.5; }
.pcard .ptitle { font-weight:700; font-size:1.02rem; margin-bottom:2px; }
.pcard .pmood { color:var(--sun); font-weight:600; font-size:.85rem; margin-bottom:8px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.chip { background:var(--sun-soft); color:#8a4b00; border-radius:999px; padding:3px 10px; font-size:.78rem; font-weight:600; }
.pcard .why { color:var(--muted); font-size:.84rem; }

#chat { border:1px solid var(--line) !important; border-radius:18px !important; background:#fff !important; }
#chat .message { font-size:.98rem; line-height:1.55; }
#box textarea { border-radius:14px !important; font-size:1rem !important; }
#send, #csend { background:var(--sun) !important; border:none !important; color:#fff !important; font-weight:700; border-radius:14px !important; }
#send:hover, #csend:hover { filter:brightness(1.07); }
.chipbtn { border-radius:999px !important; font-size:.82rem !important; background:#fff !important; border:1px solid var(--line) !important; color:var(--ink) !important; }
.chipbtn:hover { border-color:var(--sun) !important; background:var(--sun-soft) !important; }

.ccard { background:#fff; border:1px solid var(--line); border-radius:16px; padding:14px 16px; height:100%; }
.ctitle { font-weight:700; margin-bottom:8px; color:var(--navy); }
.canswer { background:#f7f9fc; border-radius:12px; padding:10px 12px; line-height:1.55; color:var(--ink); white-space:pre-wrap; }
.cdetails { margin-top:10px; font-size:.85rem; } .cdetails summary { cursor:pointer; color:var(--sun); font-weight:600; }
#trace-wrap { border:1px solid var(--line); border-radius:16px; background:#fff; }
.empty { color:var(--muted); padding:6px 4px; font-size:.92rem; }
.step { border-left:3px solid var(--line); margin:0 0 12px 8px; padding:2px 0 2px 14px; }
.step h4 { margin:0 0 6px; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.row { margin:5px 0; font-size:.92rem; line-height:1.5; color:var(--ink); }
.tag { display:inline-block; min-width:92px; font-size:.74rem; font-weight:700; padding:2px 9px; border-radius:8px; margin-right:6px; text-align:center; }
.t-thought { background:#eef3fb; color:#2A6BB5; }
.t-action { background:#fff0dc; color:#b35f00; }
.t-obs { background:#e7f6ec; color:#1d7a43; }
.t-obs-err { background:#fde8e8; color:#b42318; }
.t-answer { background:#efe9fb; color:#5b3fb0; }
.t-warn { background:#fde8e8; color:#b42318; }
code.mono { background:#f4f6f9; padding:2px 6px; border-radius:6px; font-size:.82rem; word-break:break-word; }
.summary { margin-top:6px; padding:8px 12px; background:#f7f9fc; border-radius:10px; color:var(--muted); font-size:.82rem; }
"""

HERO = f"""<div id="hero"><img src="/gradio_api/file={LOGO}" alt="CedarSun logo">
<div><h1>CedarSun Energy: your solar advisor</h1>
<p>Pick who you would like to talk to, ask a question, and open the panel below to see how the agent thinks. (Fictional company, university project.)</p></div></div>"""


def persona_card(key):
    p = ra.PERSONAS[key]
    c = p.config
    mood, _ = PERSONA_BLURB.get(key, ("", ""))
    chips = "".join(f'<span class="chip">{html.escape(k)} {v}</span>' for k, v in
                    [("temperature", c.temperature), ("top_p", c.top_p), ("top_k", c.top_k),
                     ("freq. penalty", c.frequency_penalty), ("presence penalty", c.presence_penalty), ("max tokens", c.max_tokens)])
    return (f'<div class="pcard"><div class="ptitle">{PERSONA_ICON.get(key, "")} {html.escape(p.name)}</div>'
            f'<div class="pmood">{html.escape(mood)}</div><div class="chips">{chips}</div>'
            f'<div class="why">{html.escape(p.rationale)}</div></div>')


def _short(value, limit=600):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def trace_html(run):
    parts = []
    for e in run["trace"]:
        rows = [f'<h4>Step {e["step"]}</h4>']
        if e.get("thought"):
            rows.append(f'<div class="row"><span class="tag t-thought">💭 Thought</span>{html.escape(e["thought"])}</div>')
        if e.get("kind") == "action":
            call = f'{e["action"]} {json.dumps(e["action_input"], ensure_ascii=False)}'
            rows.append(f'<div class="row"><span class="tag t-action">🔧 Action</span><code class="mono">{html.escape(call)}</code></div>')
            obs = e.get("observation")
            bad = isinstance(obs, dict) and obs.get("status") == "error"
            cls, label = ("t-obs-err", "⚠️ Observation") if bad else ("t-obs", "👁 Observation")
            rows.append(f'<div class="row"><span class="tag {cls}">{label}</span><code class="mono">{html.escape(_short(obs))}</code></div>')
        elif e.get("kind") == "answer":
            rows.append('<div class="row"><span class="tag t-answer">💬 Answer</span>sent to the customer</div>')
        else:
            rows.append(f'<div class="row"><span class="tag t-warn">↻ Repair</span>the reply was malformed, so the graph asked the model '
                        f'to fix it: {html.escape(str(e.get("error")))}</div>')
        parts.append(f'<div class="step">{"".join(rows)}</div>')
    tokens = run["prompt_tokens"] + run["completion_tokens"]
    parts.append(f'<div class="summary">{run["steps"]} tool step(s) · {run["llm_calls"]} model call(s) · {tokens} tokens · '
                 f'{run["latency"]:.1f}s · model {html.escape(str(run["model"]))} · finished: {html.escape(str(run["finished"]))}</div>')
    return "".join(parts)


EMPTY_TRACE = '<div class="empty">Ask something and each Thought, Action and Observation will appear here.</div>'


def respond(message, history, persona_key):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return "", history, EMPTY_TRACE
    try:
        run = ra.run_agent(message, history, persona=persona_key, allow_fallback=True)
        reply, trace_out = run["answer"], trace_html(run)
    except Exception as exc:
        reply = "Sorry, I'm having a technical problem right now. Please try again in a moment."
        trace_out = f'<div class="empty">error: {html.escape(type(exc).__name__)}: {html.escape(str(exc)[:200])}</div>'
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history, trace_out


def compare(message):
    """Ask the same question to every persona, one after the other, and show the answers side by side."""
    message = (message or "").strip()
    if not message:
        return [EMPTY_COMPARE] * 3
    cards = []
    for key, p in ra.PERSONAS.items():
        head = f'<div class="ctitle">{PERSONA_ICON.get(key, "")} {html.escape(p.name)}</div>'
        try:
            run = ra.run_agent(message, [], persona=key, allow_fallback=True)
            body = (f'<div class="canswer">{html.escape(run["answer"])}</div>'
                    f'<details class="cdetails"><summary>Show the reasoning</summary>{trace_html(run)}</details>')
        except Exception as exc:
            body = f'<div class="empty">error: {html.escape(type(exc).__name__)}: {html.escape(str(exc)[:200])}</div>'
        cards.append(f'<div class="ccard">{head}{body}</div>')
    return cards


EMPTY_COMPARE = '<div class="ccard"><div class="empty">Ask a question above to see this persona answer.</div></div>'


def switch_persona(key):
    return persona_card(key), [], EMPTY_TRACE


def build_demo():
    theme = gr.themes.Soft(primary_hue="orange", secondary_hue="blue", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Nunito"), "Segoe UI", "sans-serif"]).set(
        body_background_fill="#fbfaf7", block_radius="16px")
    with gr.Blocks(title="CedarSun Energy: ReAct Agent (C4)", theme=theme, css=CSS) as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            with gr.Tab("💬 Chat with one persona"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        persona = gr.Radio(choices=[(f"{PERSONA_ICON[k]}  {p.name}", k) for k, p in ra.PERSONAS.items()],
                                           value="friendly", label="Who would you like to talk to?", elem_id="persona")
                        card = gr.HTML(persona_card("friendly"))
                    with gr.Column(scale=7, min_width=380):
                        chatbot = gr.Chatbot(type="messages", height=430, show_label=False, elem_id="chat",
                                             avatar_images=(None, LOGO), show_copy_button=True,
                                             placeholder="### Hello! 👋\nAsk me about solar panels, batteries, warranties or booking a free site survey.")
                        with gr.Row():
                            box = gr.Textbox(placeholder="Type your message and press Enter...", show_label=False,
                                             container=False, scale=6, elem_id="box", autofocus=True)
                            send = gr.Button("Send", scale=1, min_width=90, elem_id="send")
                        gr.Markdown("**Try one:**")
                        for group in (EXAMPLES[:3], EXAMPLES[3:]):
                            with gr.Row():
                                for text in group:
                                    gr.Button(text if len(text) < 48 else text[:45] + "...", size="sm",
                                              elem_classes="chipbtn").click(lambda t=text: t, None, box)
                with gr.Accordion("🧠 How the agent thought (Reasoning, Action, Observation)", open=True, elem_id="trace-wrap"):
                    trace = gr.HTML(EMPTY_TRACE)
            with gr.Tab("⚖️ Compare all personas"):
                gr.Markdown("Ask **one question** and see how each persona answers it, side by side. "
                            "It asks them one after the other, so allow a little time.")
                with gr.Row():
                    cbox = gr.Textbox(placeholder="Type one question for all three personas...", show_label=False,
                                      container=False, scale=6)
                    cgo = gr.Button("Ask all three", scale=1, min_width=130, elem_id="csend")
                gr.Examples(EXAMPLES, inputs=cbox, label="Examples")
                with gr.Row(equal_height=True):
                    outs = [gr.HTML(EMPTY_COMPARE) for _ in ra.PERSONAS]
        persona.change(switch_persona, persona, [card, chatbot, trace])
        box.submit(respond, [box, chatbot, persona], [box, chatbot, trace])
        send.click(respond, [box, chatbot, persona], [box, chatbot, trace])
        cbox.submit(compare, cbox, outs)
        cgo.click(compare, cbox, outs)
    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=int(os.getenv("PORT", "7861")), allowed_paths=[str(Path(__file__).parent)])
