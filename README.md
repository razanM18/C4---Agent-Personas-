---
title: CedarSun ReAct Agent
emoji: ☀️
sdk: gradio
app_file: app_c4.py
pinned: false
---

# C4: CedarSun ReAct agent with configurable personas

EECE 503P/798S Agentic Systems, Assignment C4. This extends the C3 business chatbot (a fictional Lebanese residential solar
company, **CedarSun Energy**) into a **ReAct agent built in LangGraph** with three personas, and compares how prompt style,
persona, instructions and LLM configuration change its behaviour.

## What is here

| File | Purpose |
|---|---|
| `c4_react_agent.ipynb` | **The main deliverable.** Use case, instructions, personas, the hand-written ReAct loop, all experiments, charts and the reflection answers (Section 13), with outputs saved. |
| `react_agent.py` | The same code as a module (the notebook is generated from it). |
| `app_c4.py` | Gradio demo: switch persona, chat, and watch each Thought / Action / Observation. |
| `run_experiments.py` | Runs or resumes every experiment from the command line. |
| `results/runs.jsonl` | **The experiment log**: every run with its trace, checks, score, tokens, latency and settings. |
| `tests/test_react_offline.py` | 21 offline tests (parser, graph, tools, prompt options, scoring) with a scripted fake model. |
| `me/` | The C3 knowledge base (`about_business.pdf`, `business_summary.txt`). |
| `build_notebook.py` | Rebuilds the notebook from `react_agent.py` (and, with `--run`, executes it and saves the outputs). |

## How the requirements are met

| Requirement | Where |
|---|---|
| One agent framework, ReAct written by hand, no prebuilt executor | **LangGraph** `StateGraph`: nodes `think`, `act`, `repair`, `finish`, `force_answer`, `fallback` and conditional edges (notebook Sections 5 to 7). The model writes `Thought / Action / Action Input` as plain text; no native function calling. |
| System prompt with behaviour and tool descriptions | `build_system_text` (persona + instructions + tool descriptions generated from the tool registry). |
| Logic to detect that the model suggests an action | `parse_react` (tolerates code fences, `Action: tool({...})`, invented observations, missing thoughts) and the router `route_after_think`. |
| Execute the tool, return the observation, continue | `act` node, then the edge back to `think`. |
| Parse the final answer | `parse_react` and the `finish` node, with `force_answer` / `fallback` guards. |
| Two or more personas (prompt traits **and** LLM configuration) | Friendly Advisor, Strict Expert, Cautious Helper (Section 4), each with a written rationale for its settings. |
| Chain-of-thought on/off, zero vs few-shot, system vs in-message | Experiment **E2** (Section 11). |
| Vary temperature, top-p, top-k, penalties, max tokens, model | Experiments **E3, E4, E5, E6** (Section 12). |
| Log experiments and compare | `results/runs.jsonl` and the tables and charts in the notebook. |
| Dedicated instructions block (single-purpose rules, I/O, worked example, no conflict with the persona) | `INSTRUCTIONS` and `EXAMPLES` (Section 3c). |
| Reflection questions | Notebook Section 13. |

## Setup

```
pip install -r requirements.txt
copy .env.example .env      # then add your keys (Groq is enough for everything except the model/top-k comparisons)
python -m pytest tests -q   # offline tests, no key needed
python app_c4.py            # the chat demo
python run_experiments.py   # run or resume the experiments (cached in results/runs.jsonl)
python build_notebook.py --run   # rebuild the notebook and save its outputs
```

**Running it without my keys.** API keys are never uploaded. Anyone can run the project with **one** free key (Groq, Gemini or
OpenRouter): copy `.env.example` to `.env`, paste the key, and the chat demo and the live demo cells use whichever provider has a key.
The experiment tables and charts in the notebook are built from the saved `results/runs.jsonl`, so they show **without any key**; only
re-running an experiment needs a key for its pinned model.

The experiments are pinned to Groq `qwen/qwen3.8-27b`. Free tiers have daily and per-minute limits, so the runner waits when a
provider asks it to, saves every finished run, and can be re-run later to fill any gaps.

## Notes

* All names, contacts and figures are fictional. The knowledge base is the same as in C3.
* Reflection answers in the notebook are based on the logged results in `results/runs.jsonl`.
* The Gradio demo can be deployed to a Hugging Face Space (`app_c4.py` is the app file); add your keys as Space secrets.
