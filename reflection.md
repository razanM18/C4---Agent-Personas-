
## 13. Reflection questions

*All numbers below come from the 69 logged runs in `results/runs.jsonl` (tables in Sections 10 to 12). Each cell is a small sample on free-tier models (one run per task, three for E6), so differences of a single check are treated as noise. The scores come from automatic checks, not from a human or model judge.*

### What the experiments showed, in one table

| Question | Result |
|---|---|
| Persona (E1) | All three personas scored 0.94 to 1.00. They differ clearly in **style and cost**: Layal 35.5 words per reply, Zeina 49.3, Dana 51.2; Dana used 28% more tokens (3,939 vs about 3,050) because she verified more. |
| Chain-of-thought (E2) | With CoT: 100% clean format, 2.0 model calls, 3,116 tokens. Without: 67% clean format, 2.67 calls, 4,139 tokens (+33%). |
| Examples (E2) | Few-shot (3 examples) scored the same as one example but cost +26% tokens. Zero-shot kept the format but tried to save a lead with the name "Unknown" and the email "pending". |
| System vs in-message (E2) | No measurable difference on this model (3,169 vs 3,116 tokens, same score). |
| Sampling settings (E3) | Every setting scored 1.00, so the checks cannot separate them. Temperature 0.0 gave **1 distinct answer in 3 runs**, temperature 0.7 and 1.3 gave **3 in 3** (E6). `max_tokens 120` shortened replies (34 vs 52 words) without cutting them. |
| Model type (E4) | Qwen 1.00, gpt-oss 1.00, Gemini 3.6 flash 0.94, Laguna 0.93. The last two answered a question directly instead of acting on their first turn (67% clean format). |
| Top-k (E5) | 0.88, 0.93, 0.87 for top-k 5, 40, 100: within noise, no reliable effect. |

### Which persona gave the most helpful or natural results?
**Zeina (Friendly Advisor) read as the most natural and the most helpful for the target audience**, a non-technical homeowner:
short, warm replies that react to the customer ("Your safety is the most important thing right now"), the only persona that used contractions
(0.83 per reply) and exclamation marks (0.67 per reply). **Eng. Layal (Strict Expert) was the most reliable and the most concise**:
no exclamation marks, no questions, the only persona to quote the array size in kWp ("8 solar panels (3.6 kWp)"), and the only one that
never slipped in Arabizi. **Dana (Cautious Helper)** behaved as designed, hedging ("approximate figures", "I have passed your question to our team") and
**verifying more**: she made a second lookup in 2 of 6 tasks (sizing and safety), at the cost of the longest sentences (18.8 words on average) and 28% more tokens.
The one persona-specific failure was the Arabizi task: Zeina's reply contained an Arabic-script fragment (failing the language check, score 0.94), and Dana's reply
repeated a phrase three times ("3awel b el free site survey"). None of the three sounded convincingly Lebanese, so the Arabizi result is a limitation of the model, not a win for any persona.

### Which prompt and configuration combination performed best for this use case?
Among the setups tested, the **friendly persona with chain-of-thought, one worked example and the instructions in the system prompt, on Qwen 3.8 (27B), temperature 0.7**,
scored 1.00 in both E2 and E3 at the lowest cost (2.0 model calls, about 3,100 tokens per answer, tied with gpt-oss). Because the sampling sweep could not separate
the settings on score, the choice between temperatures rests on the variance result: **temperature 0.0 is deterministic (the same answer three times), which suits the Strict Expert,
while 0.7 gives natural variety for the Friendly Advisor, and 1.3 starts adding unrequested details** (for example workmanship and manufacturer caveats on a plain warranty question).
Our Arabizi failure suggests a lower temperature (around 0.3) for non-English replies, but **that combination was not tested**, so it is a hypothesis, not a result.

### How well did the agent reason and use tools?
Across the 69 logged runs the agent averaged **1.04 tool steps, 2.13 model calls and 3,351 tokens per question**, and no run needed the step limit or the plain-text fallback.
It called the sizing tool in **18 of 18** sizing runs and reported the tool's panel count every time; it looked up the answer in **23 of 23** knowledge questions (all said 10 years);
it saved the lead with the right name and email in 3 of 3 runs; it gave **no price in 69 of 69** runs; and every one of the 7 safety replies said to keep away, switch off at the isolators and call support or emergency services.
Weaknesses: 6 of 69 runs had at least one malformed turn (the model answered in prose without an `Action:` or `Answer:` line), which the `repair` node fixed at the cost of one extra call;
and one price run did not log the question (17 of 18). The Thought step genuinely mattered: without it the model skipped the tool and answered the sizing question from its own head until the repair step forced the action.

### Did the instructions block change the agent's behaviour in a noticeable way?
Yes, in three measurable ways. **(1) Tool discipline:** rule I1 sent every CedarSun question through `lookup_knowledge`, and I4 sent every sizing question through the sizing tool, so numbers came from Python, not from the model's memory
(an early development run without the tool invented "15 to 18 panels of 300 W"). **(2) Guardrails:** I2 (no prices) held in all 69 runs, across every persona, prompt variant, sampling setting and model. **(3) The worked example:**
removing it (zero-shot) did not break the format on this model but made it **violate I6**, trying to save a lead with invented contact details ("Unknown", "pending"). Only the email validation inside the tool stopped it,
which is a reminder that instructions and tool-side validation work best together. Persona and instructions did not conflict because they split the work: the persona controls *how it sounds* (length, tone, hedging),
the instructions control *what it does* (tools, limits, output format).

### What were the biggest challenges in implementation?
1. **Models that fight the text format.** Groq's gpt-oss models are trained to call tools natively. In development they sometimes failed with *"Tool choice is none, but model called a tool"*, and a stop word on `Observation:` cut their hidden reasoning and returned empty replies. We stopped sending the stop word to that family (the parser already cuts at `Observation:`) and retry with simpler settings on a 400 error. In the logged runs gpt-oss worked 3 of 3 times, so this is an intermittent problem.
2. **Providers differ in what they accept.** Gemini ignores frequency and presence penalty and top-k, so the LLM layer records which settings were dropped; the penalty and top-k experiments therefore run on providers that accept them.
3. **Free-tier limits.** Groq allows 8,000 tokens per minute on this model (about 2 to 3 runs a minute), Gemini 20 requests per day per model. The runner waits as the provider asks, caches every finished run, and can be resumed. A request that hung for minutes taught us to add a time limit to every call.
4. **Scoring honestly.** Our first sizing check looked for the digit "8" and wrongly failed an answer that said "eight panels"; the cached runs are now re-scored with the corrected check. The sampling sweep scored 1.00 everywhere, a ceiling effect: the automatic checks measure correctness, not naturalness, so style metrics and our reading of the answers carry the persona conclusions.
5. **Latency is not reported as a result**, because the measured times include the provider's own rate-limit waits.
