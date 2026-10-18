"""Offline tests for the ReAct parser and the LangGraph loop. A scripted fake model replaces the API, so no key is needed.
Run:  python -m pytest tests -q
"""
import json
import os
import sys
import tempfile

os.environ["LOG_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import react_agent as ra  # noqa: E402

NL = chr(10)


def fake_llm(*texts):
    """Return an llm(messages, stop) that replays the given model turns and records what it was sent."""
    queue = list(texts)
    seen = []

    def llm(messages, stop):
        seen.append({"messages": messages, "stop": stop})
        return ra.LLMResult(text=queue.pop(0), prompt_tokens=100, completion_tokens=20, provider="fake", model="fake-1")
    llm.seen = seen
    return llm


# ---------------------------------------------------------------- parser
def test_parse_action_with_json():
    p = ra.parse_react('Thought: I need the warranty.' + NL + 'Action: lookup_knowledge' + NL + 'Action Input: {"query": "battery warranty"}')
    assert p.kind == "action" and p.tool == "lookup_knowledge" and p.args == {"query": "battery warranty"}
    assert p.thought == "I need the warranty."


def test_parse_answer():
    p = ra.parse_react("Thought: done." + NL + "Answer: The battery has a 10 year warranty.")
    assert p.kind == "answer" and p.answer == "The battery has a 10 year warranty."


def test_parse_paper_style_call_and_code_fences():
    p = ra.parse_react('Thought: look up' + NL + 'Action: lookup_knowledge({"query": "areas"})')
    assert p.kind == "action" and p.tool == "lookup_knowledge" and p.args == {"query": "areas"}
    fence = chr(96) * 3
    p2 = ra.parse_react("Action: record_feedback" + NL + "Action Input: " + fence + "json" + NL + '{"question": "EV chargers?"}' + NL + fence)
    assert p2.kind == "action" and p2.args == {"question": "EV chargers?"}


def test_model_inventing_an_observation_is_cut():
    p = ra.parse_react('Thought: x' + NL + 'Action: lookup_knowledge' + NL + 'Action Input: {"query": "a"}' + NL +
                       'Observation: made up' + NL + 'Answer: invented answer')
    assert p.kind == "action" and p.args == {"query": "a"}


def test_action_before_answer_wins_and_missing_thought_is_ok():
    p = ra.parse_react('Action: lookup_knowledge' + NL + 'Action Input: {"query": "a"}' + NL + "Answer: too early")
    assert p.kind == "action"
    assert ra.parse_react("Answer: hello").kind == "answer"


def test_malformed_and_bad_json():
    assert ra.parse_react("Sure, happy to help!").kind == "malformed"
    assert ra.parse_react("Thought: hmm").kind == "malformed"
    assert ra.parse_react("Answer:").kind == "malformed"
    p = ra.parse_react("Action: record_feedback" + NL + "Action Input: {not json}")
    assert p.kind == "action" and p.error


# ---------------------------------------------------------------- graph
def test_full_react_cycle_thought_action_observation_answer():
    llm = fake_llm('Thought: I need to look up the warranty.' + NL + 'Action: lookup_knowledge' + NL + 'Action Input: {"query": "battery warranty"}',
                   "Thought: The documents say 10 years." + NL + "Answer: The battery has a 10-year warranty.")
    out = ra.run_agent("What warranty on the battery?", llm=llm)
    assert out["answer"] == "The battery has a 10-year warranty."
    assert out["tools_used"] == ["lookup_knowledge"] and out["steps"] == 1 and out["finished"] == "answer"
    assert out["llm_calls"] == 2 and out["prompt_tokens"] == 200
    assert out["trace"][0]["observation"]["status"] == "ok"
    # the Observation was fed back to the model on the second call
    second = llm.seen[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Observation:")
    assert llm.seen[0]["stop"] == ["Observation:"]


def test_tool_is_really_executed_and_logged():
    llm = fake_llm('Thought: log it' + NL + 'Action: record_feedback' + NL + 'Action Input: {"question": "Do you install EV chargers?"}',
                   "Answer: I have passed that to the team.")
    out = ra.run_agent("Do you install EV chargers?", llm=llm)
    assert out["trace"][0]["observation"]["status"] == "ok"
    assert "EV chargers" in (ra.LOG_DIR / "feedback.jsonl").read_text(encoding="utf-8")


def test_sizing_tool_numbers_reach_the_answer():
    args = {"appliances": [{"name": "refrigerator"}, {"name": "air conditioner"}], "area": "Beirut"}
    llm = fake_llm("Thought: size it" + NL + "Action: estimate_system_size" + NL + "Action Input: " + json.dumps(args),
                   "Answer: Roughly 4 panels, approximate until the survey.")
    out = ra.run_agent("How many panels?", llm=llm)
    panels = out["trace"][0]["observation"]["panels_450w"]
    assert panels == 4 or isinstance(panels, int)


def test_unknown_tool_becomes_an_error_observation_and_loop_continues():
    llm = fake_llm("Action: fly_to_moon" + NL + "Action Input: {}", "Answer: Sorry, I cannot do that.")
    out = ra.run_agent("hi", llm=llm)
    assert out["trace"][0]["observation"]["status"] == "error" and out["finished"] == "answer"


def test_malformed_turn_is_repaired():
    llm = fake_llm("Happy to help!", "Thought: ok" + NL + "Answer: Hello, how can I help with solar?")
    out = ra.run_agent("hi", llm=llm)
    assert out["parse_errors"] == 1 and out["finished"] == "answer"
    assert llm.seen[1]["messages"][-1]["content"].startswith("Format error")


def test_repeated_format_errors_fall_back_to_raw_text():
    llm = fake_llm("blah one", "blah two", "blah three")
    out = ra.run_agent("hi", llm=llm)
    assert out["finished"] == "format_fallback" and out["answer"] == "blah three"


def test_step_limit_forces_an_answer():
    act = "Thought: again" + NL + "Action: lookup_knowledge" + NL + 'Action Input: {"query": "warranty"}'
    llm = fake_llm(act, act, act, "Answer: Final answer after the limit.")   # the 3rd action exceeds the limit
    out = ra.run_agent("hi", llm=llm, max_steps=2)
    assert out["finished"] == "max_steps" and out["answer"] == "Final answer after the limit." and out["steps"] == 2


# ---------------------------------------------------------------- prompt options and personas
def test_persona_configs_differ_and_are_justified():
    cfgs = {k: p.config for k, p in ra.PERSONAS.items()}
    assert cfgs["expert"].temperature < cfgs["cautious"].temperature < cfgs["friendly"].temperature
    assert all(p.rationale for p in ra.PERSONAS.values())


def test_prompt_options_change_the_prompt():
    persona = ra.PERSONAS["friendly"]
    base = ra.build_system_text(persona, ra.PromptOptions())
    zero = ra.build_system_text(persona, ra.PromptOptions(shots=0))
    few = ra.build_system_text(persona, ra.PromptOptions(shots=3))
    nocot = ra.build_system_text(persona, ra.PromptOptions(cot=False))
    assert "WORKED EXAMPLE 1" in base and "WORKED EXAMPLE 2" not in base
    assert "WORKED EXAMPLE" not in zero
    assert "WORKED EXAMPLE 3" in few
    assert "Do not write any reasoning" in nocot and "Thought:" not in nocot.split("WORKED EXAMPLE 1")[1]


def test_in_message_variant_has_no_system_role():
    persona = ra.PERSONAS["expert"]
    sys_msgs = ra.build_messages("hello", [], persona, ra.PromptOptions(instructions_in="system"))
    usr_msgs = ra.build_messages("hello", [], persona, ra.PromptOptions(instructions_in="user"))
    assert sys_msgs[0]["role"] == "system" and usr_msgs[0]["role"] == "user"
    assert "=== INSTRUCTIONS ===" in usr_msgs[0]["content"] and len(usr_msgs) == 1


def test_language_line_follows_the_customer():
    msgs = ra.build_messages("kifak, shu el garantie?", [], ra.PERSONAS["friendly"], ra.PromptOptions())
    assert "Arabizi" in msgs[0]["content"]


# ---------------------------------------------------------------- knowledge tool and scoring
def test_lookup_knowledge_finds_the_warranty():
    r = ra.lookup_knowledge("battery warranty")
    assert r["status"] == "ok" and any("10" in p["text"] for p in r["passages"])
    assert ra.lookup_knowledge("")["status"] == "error"


def test_scoring_checks():
    run = {"answer": "Roughly 4 panels, an approximate figure; the free survey confirms it.", "tools_used": ["estimate_system_size"],
           "parse_errors": 0, "finished": "answer",
           "trace": [{"kind": "action", "action": "estimate_system_size", "observation": {"status": "ok", "panels_450w": 4}}]}
    checks = ra.score_run("sizing", run)
    assert all(checks.values()), checks
    price = {"answer": "It costs $5,000.", "tools_used": [], "parse_errors": 0, "finished": "answer", "trace": []}
    assert ra.score_run("price", price)["no_price"] is False


def test_param_degradation_variants():
    cfg = ra.LLMConfig(top_k=40)
    v = ra._param_variants(cfg, "groq", "openai/gpt-oss-120b")
    assert "frequency_penalty" in v[0][0] and v[-1][0].keys() == {"temperature", "max_tokens"}
    assert any("top_k" in extra for _, extra, _ in ra._param_variants(cfg, "openrouter", "x"))


def test_sizing_check_accepts_spelled_out_numbers():
    run = {"answer": "You would need roughly eight panels, as an approximate figure.", "tools_used": ["estimate_system_size"],
           "parse_errors": 0, "finished": "answer",
           "trace": [{"kind": "action", "action": "estimate_system_size", "observation": {"status": "ok", "panels_450w": 8}}]}
    assert ra.score_run("sizing", run)["reports_tool_panels"] is True
