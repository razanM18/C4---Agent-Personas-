"""Builds c4_react_agent.ipynb from react_agent.py (code and explanations) plus the demo, experiment and reflection cells below.

    python build_notebook.py            # writes the notebook without outputs
    python build_notebook.py --run      # also executes it and saves the outputs inside the notebook

react_agent.py is the single source of truth: edit it, then rebuild the notebook.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NL = chr(10)


def cells_from_source(text):
    """Split a '# %%' percent-format file into notebook cells."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = NL.join(buf).strip(NL)
        if not body:
            return
        if kind == "markdown":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.split(NL)]
            cells.append(new_markdown_cell(NL.join(lines)))
        else:
            cells.append(new_code_cell(body))

    for line in text.split(NL):
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def find(cells, heading):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(heading):
            return i
    raise KeyError(heading)


def md(text):
    return new_markdown_cell(text.strip(NL))


def code(text):
    return new_code_cell(text.strip(NL))


cells = cells_from_source((HERE / "react_agent.py").read_text(encoding="utf-8"))

# ---- after the graph definition: show the compiled graph
i = find(cells, "## 6. The ReAct loop as a LangGraph state machine")
cells.insert(i + 2, code("""
# The compiled graph, exactly as LangGraph sees it (paste this into https://mermaid.live to draw it).
print(build_graph(lambda messages, stop: None).get_graph().draw_mermaid())
"""))

# ---- after run_agent: live demo of the loop
i = find(cells, "## 7. Running the agent")
cells[i + 2:i + 2] = [
    md("""
### 7.1 Live demo: one full Thought, Action, Observation, Answer cycle
The warranty question needs one tool call, the sizing question needs the sizing tool. Watch the trace.
"""),
    code("""
DEMO_QUESTIONS = ["What warranty do you give on the battery and on the inverter?",
                  "I live in a 3 bedroom flat in Beirut with a fridge, 4 lights, a TV and 2 air conditioners. How many solar panels do I need?"]
for question in DEMO_QUESTIONS:
    print("=" * 100)
    print("Customer:", question)
    show_trace(run_agent(question, persona="friendly"))
"""),
]

# ---- the evaluation harness comes after section 8; experiments after section 9
i = find(cells, "## 9. Experiment plans")
j = i + 2                                           # the code cell with the plans
cells[j + 1:j + 1] = [
    code("""
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 110


def plot_scores(df, title, by="variant"):
    s = summarize(df, by)
    ax = s["score"].plot(kind="barh", figsize=(7.5, 0.5 * len(s) + 1.3), xlim=(0, 1.08), color="#2A6BB5")
    ax.set_xlabel("mean share of checks passed"); ax.set_title(title); ax.invert_yaxis(); ax.set_ylabel("")
    for k, v in enumerate(s["score"]):
        ax.text(v + 0.01, k, f"{v:.2f}", va="center")
    plt.tight_layout(); plt.show()


def show_answers(df, tasks=("sizing", "price"), by="variant"):
    for name in df[by].unique():
        for t in tasks:
            sub = df[(df[by] == name) & (df["task"] == t)]
            if len(sub):
                print(f"[{name} | {t}] score {sub.iloc[0]['score']:.2f}: " + sub.iloc[0]["answer"].replace(chr(10), " ")[:420])
        print()
"""),
    md("""
## 10. E1: which persona?
Three personas, each with its own prompt wording **and** its own LLM configuration, run on the same six tasks.
"""),
    code("""
E1 = run_all(("E1",))["E1"]
display(summarize(E1))
display(E1.pivot_table(index="variant", columns="task", values="score", sort=False).round(2))
plot_scores(E1, "E1: score by persona")
"""),
    code("""
show_answers(E1, tasks=("sizing", "price", "safety"))
"""),
    code("""
# Style of the replies (the part a score cannot capture): length, exclamation marks, questions, contractions.
style = E1.assign(**{k: E1["style"].apply(lambda s, k=k: s[k]) for k in ("words", "exclamations", "questions", "contractions", "avg_sentence_words")})
display(style.groupby("variant", sort=False)[["words", "avg_sentence_words", "exclamations", "questions", "contractions"]].mean().round(2))
"""),
    md("""
## 11. E2: prompt engineering
Baseline = the *friendly* persona with chain-of-thought, one worked example and the instructions in the system role.
Each variant changes **one** thing: no chain-of-thought, zero-shot (no example), few-shot (three examples), and instructions
placed inside the user message instead of the system role.
"""),
    code("""
E2 = run_all(("E2",))["E2"]
display(summarize(E2))
display(E2.pivot_table(index="variant", columns="task", values="score", sort=False).round(2))
plot_scores(E2, "E2: score by prompt technique")
"""),
    code("""
# Which individual checks fail most often, per technique?
rows = []
for _, r in E2.iterrows():
    for check, ok in r["checks"].items():
        rows.append({"variant": r["variant"], "check": check, "passed": bool(ok)})
display(pd.DataFrame(rows).pivot_table(index="variant", columns="check", values="passed", aggfunc="mean", sort=False).round(2))
"""),
    md("""
## 12. E3 to E6: LLM configuration
**E3** changes one sampling setting at a time against the friendly baseline (temperature, top-p, frequency penalty,
presence penalty, max tokens). **E4** changes the model. **E5** changes top-k on a provider that supports it
(top-k is not part of the OpenAI API, so Groq ignores it). **E6** repeats the same question to measure how much the
answer varies with temperature.
"""),
    code("""
E3 = run_all(("E3",))["E3"]
display(summarize(E3))
display(E3.pivot_table(index="variant", columns="task", values="score", sort=False).round(2))
plot_scores(E3, "E3: score by sampling configuration")
"""),
    code("""
E4 = run_all(("E4",))["E4"]
display(summarize(E4))
display(E4[["variant", "task", "score", "parse_errors", "finished", "dropped"]].assign(finished=E4["finished"].str[:70]))
plot_scores(E4, "E4: score by model")
"""),
    code("""
E5 = run_all(("E5",))["E5"]
display(summarize(E5))
display(E5.pivot_table(index="variant", columns="task", values="score", sort=False).round(2))
"""),
    code("""
E6 = run_all(("E6",))["E6"]
variance = E6.groupby("variant", sort=False).agg(runs=("answer", "size"), distinct_answers=("answer", "nunique"),
                                                  mean_words=("style", lambda s: sum(x["words"] for x in s) / len(s)),
                                                  mean_score=("score", "mean")).round(2)
display(variance)
for name in E6["variant"].unique():
    print(name)
    for a in E6[E6["variant"] == name]["answer"]:
        print("   -", a.replace(chr(10), " ")[:230])
"""),
    code("""
# One table for the whole project: every setup, ranked.
frames = [E1, E2, E3, E4, E5]
for code_name, frame in zip(("E1", "E2", "E3", "E4", "E5"), frames):
    frame["experiment"] = code_name
everything = pd.concat(frames, ignore_index=True)
overall = summarize(everything, by=["experiment", "variant"]).sort_values("score", ascending=False)
display(overall.head(12))
print("Total model runs logged:", len(pd.read_json(RUNS_FILE, lines=True)), "| file:", RUNS_FILE.name)
"""),
]

# ---- reflection
reflection = (HERE / "reflection.md").read_text(encoding="utf-8") if (HERE / "reflection.md").exists() else "## 13. Reflection" + NL + "(to be written)"
cells.append(md(reflection))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = HERE / "c4_react_agent.ipynb"

if "--run" in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    print("executed")
nbformat.write(nb, out)
print("wrote", out.name, "with", len(nb.cells), "cells")
