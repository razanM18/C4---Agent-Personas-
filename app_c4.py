"""CedarSun ReAct agent (C4) with a persona switch and a live view of the reasoning.

    python app_c4.py        # then open the address it prints (usually http://127.0.0.1:7861)
"""
import json
import os

import gradio as gr

import react_agent as ra

NL = chr(10)
TICK = chr(96)

EXAMPLES = ["How many solar panels do I need? I have a fridge, 4 lights, a TV and 2 ACs in Beirut.",
            "What warranty do you give on the battery and the inverter?",
            "How much does a 5 kW system cost?",
            "I'd like a free site survey. My name is Rami and my email is rami@example.com",
            "My battery is making a burning smell and there is smoke.",
            "kifak, kam sene el garantie 3ala el battery?"]


def persona_card(key):
    p = ra.PERSONAS[key]
    c = p.config
    return (f"**{p.name}**" + NL + NL + f"`temperature {c.temperature}` `top_p {c.top_p}` `top_k {c.top_k}` "
            f"`frequency_penalty {c.frequency_penalty}` `presence_penalty {c.presence_penalty}` `max_tokens {c.max_tokens}`"
            + NL + NL + p.rationale)


def trace_markdown(run):
    lines = []
    for e in run["trace"]:
        lines.append(f"**Step {e['step']}**")
        if e.get("thought"):
            lines.append("- **Thought:** " + e["thought"])
        if e.get("kind") == "action":
            lines.append("- **Action:** " + TICK + e["action"] + " " + json.dumps(e["action_input"], ensure_ascii=False) + TICK)
            obs = json.dumps(e.get("observation"), ensure_ascii=False).replace(TICK, "'")
            lines.append("- **Observation:** " + TICK + (obs[:600] + ("..." if len(obs) > 600 else "")) + TICK)
        elif e.get("kind") == "answer":
            lines.append("- **Answer:** (sent to the customer)")
        else:
            lines.append("- *malformed turn, the graph asked the model to repair it:* " + str(e.get("error")))
    lines.append(NL + f"*{run['steps']} tool step(s), {run['llm_calls']} model call(s), {run['prompt_tokens'] + run['completion_tokens']} tokens, "
                 f"{run['latency']:.1f}s, model `{run['model']}`, finished: {run['finished']}*")
    return NL.join(lines)


def respond(message, history, persona_key):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return "", history, "*Ask something to see the agent's reasoning here.*"
    try:
        run = ra.run_agent(message, history, persona=persona_key, allow_fallback=True)
        reply, trace_md = run["answer"], trace_markdown(run)
    except Exception as exc:
        reply = "Sorry, I'm having a technical problem right now. Please try again in a moment."
        trace_md = f"*error: {type(exc).__name__}: {str(exc)[:200]}*"
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history, trace_md


def build_demo():
    with gr.Blocks(title="CedarSun ReAct Agent (C4)", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("## CedarSun Energy: ReAct agent with configurable personas" + NL +
                    "Pick a persona, chat, and open **Reasoning** to see each Thought, Action and Observation. "
                    "(Fictional company, university project.)")
        persona = gr.Radio(choices=[(p.name, k) for k, p in ra.PERSONAS.items()], value="friendly", label="Persona")
        card = gr.Markdown(persona_card("friendly"))
        chatbot = gr.Chatbot(type="messages", height=380, show_label=False,
                             placeholder="Say hello, or try one of the examples below.")
        box = gr.Textbox(placeholder="Type your message...", show_label=False)
        with gr.Accordion("Reasoning: Thought, Action, Observation", open=True):
            trace = gr.Markdown("*Ask something to see the agent's reasoning here.*")
        gr.Examples(EXAMPLES, inputs=box)
        persona.change(persona_card, persona, card)
        box.submit(respond, [box, chatbot, persona], [box, chatbot, trace])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=int(os.getenv("PORT", "7861")))
