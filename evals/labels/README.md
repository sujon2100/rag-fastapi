# Hand labels

Labels written by a person reading the agent's answers, used to measure how often the LLM judge agrees with a human. They are not used to score the agent.

`baseline_refusals.jsonl`: did the answer decline to answer (true/false), for the 38 non-authz cases of the baseline agent run `results/2026-10-29_131139_agent.json`. `answer_sha` is a hash of the answer the label refers to, so a label is never applied to a different answer. A request for clarification with no factual content (lg-02) counts as a refusal.

First pass done during the eval build on 2026-10-29, to be confirmed by Helal.

`fixB_grounding.jsonl`: groundedness (supported / partial / unsupported) for 17 answers from the fix B run `results/2026-10-29_143035_fixB-prompt-forced_all.json`, judged against the text the agent actually retrieved. 15 answerable cases were picked to cover all three docs, plus mc-04 and un-01, which were flagged during an earlier spot check.
