# Final Pitch Strategy and Script — AI Cup 2026 Congress

_Last updated: 2026-04-13_

## Purpose
This file is the final grounded brief for the 4-minute AI Cup congress pitch. It is based on:
- `docs/JURY_TARGETING_PLAYBOOK.md`
- `docs/CONGRESS_PROFILE.md`
- `docs/presentation/RALPLAN_REVISED_PLAN.md`
- `FINAL_ARCHITECTURE.md`
- `E176_FULL_REPORT.md`
- `TRACKER.md`
- `G:\Projects\research-copilot\README.md` and workflow docs (Q&A only)

The goal is not to sound like the team with the most tricks. The goal is to sound like the team with the best judgment.

---

## 1) Strongest differentiators

### 1. We found the real failure mode of the competition
The strongest evidence-backed differentiator is that we discovered the core generalization problem: **month shift / temporal overfitting**.

Grounding:
- `TRACKER.md` documents the root cause clearly:
  - train months = `[1,4,9,10]`
  - test months = `[2,5,9,10,12]`
  - **33% of test data comes from months never seen in training**
  - earlier high CV scores were inflated by temporal proxies
- The clearest proof object is the failure case:
  - **E15: 0.7535 CV -> 0.52 leaderboard**
  - temporal-feature config predicted **0 pigeons** on test

Why this differentiates us:
Most finalist teams will likely talk about modeling. Our stronger story is that we diagnosed what made modeling deceptively easy and deployment deceptively hard.

### 2. We used honest evaluation instead of flattering evaluation
`E176_FULL_REPORT.md` shows that **SKF and LOMO can tell opposite stories**. That is a rare and powerful team signal.

Grounding:
- Baseline E175 blend: **SKF 0.7043, LOMO 0.5461**
- Isotonic non-CV looked "overfit" by SKF but was best by LOMO
- Month-by-month breakdown exposed **Month 9 as catastrophic**

Why this differentiates us:
We can credibly say we optimized for what survives shift, not what wins a comfortable validation split.

### 3. We built a research-to-design pipeline, not just a stack of experiments
`FINAL_ARCHITECTURE.md` is unusually strong because it maps architectural choices back to evidence and rejected alternatives.

Grounding:
- explicit research-to-design mapping
- explicit "what not to do"
- cross-month stability selection logic
- ranking-aware design rationale

Why this differentiates us:
We do not just have results; we can explain why each major choice exists.

### 4. We learned from failures aggressively and visibly
The repo documents many discarded ideas: pseudo-labeling, isotonic CV, TTA for trees, MOMENT embeddings, path signatures, MultiRocket, etc.

Why this differentiates us:
Our process looks less like random searching and more like **structured elimination**.

### 5. We treated the task as ecology + operations + sensing, not just classification
This is the congress-facing differentiator.

Grounding:
- jury playbook and congress profile both point toward deployment, mitigation, trust, and ecology
- `RALPLAN_REVISED_PLAN.md` correctly positions the work as a **system-anchored workflow hybrid**

Why this differentiates us:
It makes us more than a competition team; it makes us look like a team that can build real applied AI.

---

## 2) Biggest defensible claims

These are the strongest claims we can make safely.

1. **We identified and corrected the core generalization failure mode: unseen-month shift.**
2. **We optimized for cross-month robustness, not just within-month CV.**
3. **We built a system, not a single black-box model.**
4. **We know where the model breaks and what classes remain hardest.**
5. **We rejected attractive methods that did not generalize.**
6. **We treated the output as decision support for monitoring and mitigation, not autonomous truth.**
7. **Our team’s experimentation discipline was a competitive asset** — use the most-submissions award as a secondary proof chip of learning velocity, not as headline proof.

Best phrasing version:
> We did not just optimize a leaderboard score. We built a research-driven, shift-aware, ecology-grounded decision-support system and a workflow that let us separate ideas that were merely promising from ideas that were actually robust.

---

## 3) Claims to avoid

Avoid these even if they sound impressive:

- **Do not claim the final architecture was fully validated end-to-end** if parts remain design synthesis rather than final leaderboard proof.
- **Do not claim path signatures / direct ranking architecture / Group DRO already delivered the final winning result** unless the exact experiment is in the repo as a validated success.
- **Do not overclaim species certainty.** Use: radar as ecological sensor, not species oracle.
- **Do not overclaim deployment readiness.** Use: deployable direction / decision-support pathway / realistic operational fit.
- **Do not imply other finalists only chased the leaderboard.** Contrast by describing our rigor, not by insulting them.
- **Do not overplay the most-submissions award.** It is a badge, not the proof.
- **Do not overstate Research Copilot.** If mentioned, it is an internal bounded research-ops tool and is safer in Q&A than in the main pitch.

---

## 4) Best proof-anchor options

### Best primary proof anchor
**Temporal overfitting discovery and correction**

Why it is best:
- strongest evidence of insight
- strongest evidence of rigor
- most memorable technical lesson
- strongest answer to Geert-Jan and Bart
- also persuasive to Joep because it ties to real seasonal ecology

Use this version:
> Our turning point was realizing that some of our best-looking models were wrong for the right-looking reasons. We found that 33% of the test set came from months never seen in training, so we rebuilt our validation and feature logic around cross-month robustness.

### Best secondary proof anchors
1. **E175 baseline vs LOMO split honesty**
2. **GMM archetype correction: 0.5461 -> 0.5633 LOMO, Month 9: 0.411 -> 0.433**
3. **Per-month breakdown showing Month 9 failure and hard classes**
4. **MOMENT failure** as anti-hype proof
5. **Most submissions award** as disciplined throughput badge

### Good proof-anchor combinations
- **Technical jury combo:** month shift + LOMO + Month 9 breakdown
- **Congress room combo:** month shift + system snapshot + deployment payoff
- **Team-quality combo:** month shift + rejected flashy methods + most-submissions badge

---

## 5) Best question-shaping strategy

### Questions we want
- Why is this more robust than a typical competition model?
- How would it actually be used in wind-farm monitoring or mitigation?
- What did you learn about ecology and seasonality from the data?
- What makes your team different from other strong teams?
- How could this become a deployable product or research platform?

### Questions we do not want
- Is this just a leaderboard hack?
- Are you claiming exact species identification from radar?
- Did you just brute-force submissions?
- Is this already a fully production-ready autonomous system?

### How to shape questions in the talk
- **Open with the shared-airspace problem** to trigger deployment and ecology questions.
- **Show the system early** so the room knows there is a concrete object.
- **Introduce workflow as the reason to trust the system**, not as self-congratulation.
- **Use month shift as the memorable technical lesson**; it naturally triggers rigor questions instead of score-only questions.
- **End on monitoring + mitigation + human oversight** to trigger operational and impact questions.

### Universal answer format
**Direct answer -> evidence -> limitation -> deployment implication**

---

## Pitch-building principles

1. **System first, workflow second, proof third, deployment last.**
2. **One strong idea per slide.**
3. **One visible proof object beats five small numbers.**
4. **Use the workflow to explain judgment, not effort.**
5. **Thread ecology through the technical middle.**
6. **Never sound like a Kaggle recap.**
7. **Sell the team as much as the model.**

A useful internal rule from the pitch strategy work:
> We are not just presenting a model; we are presenting a credible way of building deployable AI for a difficult real-world problem.

---

## 4-minute pitch script

Good afternoon. We’re Team Epoch, and our project sits in a difficult shared airspace: wind energy on one side, biodiversity on the other.

Our challenge was to classify bird-related radar tracks in a way that is not only accurate, but actually useful for nature-inclusive wind-energy decisions. And very early on, we realized that this was not just a classification task. It was an ecology problem, an operations problem, and a generalization problem.

So instead of treating the competition like a normal leaderboard sprint, we built a system around one question: **what will still work when conditions change?**

Our solution combines three layers.
First, we model the **trajectory itself**: how something moves through the airspace. That gives us behavioral signal.
Second, we add **contextual and environmental information**, because radar signal alone is not enough.
Third, we use **ranking-aware modeling and post-processing**, because the evaluation metric rewards correct ranking, not just raw classification.

But the biggest turning point was not one model. It was a failure.
At one point, we had a configuration with a very strong cross-validation score — but it collapsed on the leaderboard. When we investigated, we found the real issue: the train and test sets had different month distributions, and a third of the test set came from months never seen in training. In other words, some of our best-looking models were learning the calendar instead of the birds.

That changed our entire workflow.
We moved from flattering validation to honest validation, using month-aware analysis and per-month breakdowns. We rejected ideas that looked promising locally but failed under shift. We documented failures, not just successes. That included generic foundation-model approaches, pseudo-labeling schemes, and calibration methods that sounded strong but did not generalize.

What came out of that process was not just a better score. It was a more defensible system.
We can explain what the model uses, where it is strong, and where it still struggles — especially on difficult classes like cormorants and waders, and in the hardest seasonal regime.

That is also why we see this as **decision support**, not blind automation.
In practice, a system like this could help operators and ecologists prioritize monitoring, support better targeted mitigation, and make shutdown decisions more evidence-based and more defensible — especially in settings where manual observation is difficult.

And I think that is what differentiates our team.
We did not just build a model that worked once. We built a research workflow that let us find what was real, reject what was brittle, and turn a hard sensing problem into something more robust and more deployable.

So our final message is simple:
The strength of our work is not only that it performs competitively, but that it is rigorous, ecologically grounded, and built with real-world deployment in mind.

---

## Optional Q&A note on Research Copilot
Only mention if asked about team workflow, AI tooling, or how you managed experimentation.

Safe line:
> Alongside the competition work, we built a bounded internal research-ops copilot that lets a human and coding agents work against the same local experiment state. It helped us structure triage, experiments, review, and next-step decisions, but it was support infrastructure, not the core competition artifact.

Do not make it a main-slide headline.

---

## Final recommendation
If we need one dominant identity for the room, it is this:

> We are the team that understood the real problem best.

Not because we had the flashiest model, but because we showed the judgment to find the real failure mode, build around it, and explain why the final system deserves trust.
