# Critic Verdict — RALPLAN Revised Plan

_Last updated: 2026-04-11_

Grounded in:
- `docs/presentation/RALPLAN_REVISED_PLAN.md`
- `docs/presentation/RALPLAN_ARCHITECT_REVIEW.md`
- `docs/JURY_TARGETING_PLAYBOOK.md`
- `docs/CONGRESS_PROFILE.md`
- `docs/presentation/STORYTELLING_MAP.md`
- `docs/presentation/FINAL_SLIDE_PLAN.md`

## Verdict
# ITERATE

The revised plan is a strong improvement over the initial workflow-first version and it correctly incorporates the architect's main thesis: **the deck now leads with a concrete system object and uses workflow as a credibility engine rather than as the protagonist**. However, it is **not yet ready for direct slide-copy/deck production** because several consensus-planning quality criteria are only partially satisfied, and two grounding artifacts still materially conflict with the revised plan.

---

## 1) Principle-option consistency

## Assessment: **Mostly passes, with one important inconsistency**

The revised plan's chosen option (**system-anchored workflow hybrid**) is broadly consistent with its stated principles:

- **Principle 1: Lead with the object, not the meta-story**  
  Consistent with Option C and the revised slide order.
- **Principle 2: Workflow as credibility infrastructure**  
  Consistent with the new role of the workflow slide.
- **Principle 3: Ecology inside the technical core**  
  Present in the system / insights / deployment language.
- **Principle 4: Proof must be visible, selective, and fast to read**  
  Supported by the proof-anchor policy.
- **Principle 5: Restrained research-notebook design**  
  Supported by explicit visual restraint rules.

### But the inconsistency is material:
Two grounding docs still describe the **old workflow-first narrative**:
- `docs/presentation/STORYTELLING_MAP.md` still has **Slide 3 = Team workflow** and **Slide 4 = Architecture**.
- `docs/presentation/FINAL_SLIDE_PLAN.md` still has **Slide 3 = Our workflow** and **Slide 4 = The system**.

That means the revised plan is internally coherent, but the **planning bundle is not yet coherent as a whole**. For a consensus plan to be execution-ready, the core supporting artifacts must agree on the same slide spine.

### Required fix
Update `STORYTELLING_MAP.md` and `FINAL_SLIDE_PLAN.md` so they match the revised hybrid order:
1. Opening
2. Problem
3. System snapshot
4. Workflow as credibility engine
5. Key insights
6. Proof/results
7. Deployment

---

## 2) Fairness of alternatives

## Assessment: **Passes**

The revised plan presents alternatives fairly and with bounded pros/cons:
- **Option A — Model-first competition deck** is presented as a real viable choice with real advantages.
- **Option B — Workflow-first congress deck** is represented fairly rather than caricatured.
- **Option C — Hybrid** is justified by tradeoffs rather than by hand-waving.
- ADR alternatives are also reasonable and explain why proposal-section and deployment-first approaches were not chosen.

This meets the critic criterion that the chosen direction should emerge from a plausible option set, not a strawman field.

---

## 3) Risk mitigation clarity

## Assessment: **Partially passes**

The revised plan meaningfully improves risk handling versus the initial draft. It clearly addresses the main architect risks:
- **System arrives too late** -> fixed by moving system snapshot to Slide 3.
- **Workflow over-weighted** -> fixed by reframing workflow as credibility engine.
- **Ecology under-threaded** -> explicitly called out in system, insight, and deployment sections.
- **Most-submissions award overweighted** -> demoted to secondary proof chip.
- **Notebook style could become decorative clutter** -> explicit restraint rules added.

### Remaining mitigation gaps
The plan still lacks enough specificity on several production risks:

1. **Proof-anchor selection risk**  
   The plan says “choose the primary result/proof anchor” later, but that choice is central to Slide 6 and affects the whole credibility balance. Until it is chosen, the plan leaves a core narrative risk unresolved.

2. **Evidence availability risk**  
   The plan says to prefer real data-derived visuals and one direct result anchor, but it does not list which actual assets exist today and which would need to be created. That is a production risk.

3. **Narrative compression risk**  
   The plan compresses problem, system, workflow, insights, proof, and deployment into 7 slides / ~4 minutes, but does not yet specify which claims are mandatory versus optional if timing slips.

4. **Visual execution risk**  
   The restraint rules are good, but there is still no explicit anti-failure rule for common slide mistakes such as too many labels in the system figure, too much prose in the insight cards, or overcrowding the proof slide.

### Required fix
Add a short “production risks and mitigations” section covering:
- chosen proof anchor and fallback anchor
- which figures are available now vs need design
- what gets cut first if timing exceeds 4 minutes
- max word count / object count by slide type

---

## 4) Testable acceptance criteria

## Assessment: **Fails in current form**

The plan contains good intentions, but not enough **testable pass/fail criteria**.

Examples of currently useful but not yet testable statements:
- “show one concrete result/proof anchor”
- “readable in under 3 seconds”
- “no slide should feel empty”
- “the room has a concrete object to hold onto”
- “workflow should avoid slipping back into look-how-much-we-did”

These are directionally right, but not yet operationalized enough for execution.

### What is missing
A strong consensus plan should define criteria such as:
- **Slide-order lock:** slide 3 must be the system snapshot; slide 4 must be workflow.
- **Per-slide word budget:** e.g. no more than 25–35 visible words on system / insight / deployment slides, excluding micro-annotations.
- **Proof-slide requirement:** exactly 1 primary proof anchor + up to 2 supporting evidence blocks + 1 optional throughput badge.
- **Ecology-thread requirement:** system slide, insight slide, and deployment slide must each contain one explicit ecology/mitigation statement.
- **Timing requirement:** full spoken script must land between 3:45 and 3:55 in rehearsal.
- **Readability test:** each slide must pass a 3-second glance test and a room-distance readability test.
- **Award-weighting requirement:** most-submissions award appears once as a secondary chip only.

### Required fix
Add a short acceptance-criteria table with explicit pass/fail checks for:
- narrative order
- timing
- proof anchor inclusion
- ecology thread inclusion
- award weighting
- slide density / readability
- deployment specificity

---

## 5) Concrete verification steps

## Assessment: **Partially passes**

The plan has a verification phase:
- timing test
- readability test from distance
- one-glance test
- Q&A stress test by juror group
- clutter trim pass

This is good, but it is still too high-level for a critic approval because it does not specify:
- who runs each check
- against what rubric
- what constitutes pass/fail
- what happens if a check fails

### Missing verification detail
A concrete verification section should include:
1. **Narrative verification** — compare final slide order and copy against the revised hybrid plan.
2. **Audience verification** — confirm each juror's top interest is explicitly served by at least one slide or prepared answer.
3. **Evidence verification** — verify that the chosen proof anchor is factually supported by actual project evidence.
4. **Design verification** — check each slide against restraint rules and density limits.
5. **Delivery verification** — timed rehearsal under 4 minutes, plus a second rehearsal with interruption margin.

### Required fix
Add a checklist with owner + pass condition, for example:
- “Proof anchor selected and source-verified”
- “Slide 3 system figure understandable without narration in 3 seconds”
- “Slide 6 does not rely on the submissions award as primary proof”
- “Slide 7 states a concrete operational change”

---

## 6) Does the revised hybrid really solve the architect's critique?

## Assessment: **Mostly yes**

The revised plan does solve the architect's core criticism **at the strategy level**:

### Architect critique vs revised response
- **System arrives too late** -> **Solved strategically** by moving system snapshot to Slide 3.
- **Workflow over-centered** -> **Solved strategically** by redefining workflow as trust infrastructure.
- **Ecology under-threaded** -> **Largely solved** by putting ecology into system/insight/deployment language.
- **Most-submissions over-weighted** -> **Solved strategically** by making it secondary.
- **Need visual restraint rules** -> **Solved strategically** with max-figure/annotation/motif rules.
- **Deployment slide too generic** -> **Improved**, though it still needs one or two exact operational sentences in final copy.

### Why this is not a full pass yet
The plan solves the critique in prose, but **not yet across the whole planning stack** because the supporting storytelling docs still encode the old order. So the revised hybrid is correct in principle, but not yet consistently propagated.

---

## 7) Is the plan ready to drive actual slide copy / deck production?

## Assessment: **Not yet**

It is ready to drive the **next iteration of planning**, but not yet safe to use as the sole execution blueprint.

### Why not yet
1. **Core support docs are out of sync** (`STORYTELLING_MAP.md`, `FINAL_SLIDE_PLAN.md`).
2. **Primary proof anchor is still undecided**, which blocks Slide 6 copy and affects the overall trust arc.
3. **Acceptance criteria are not explicit enough** to prevent drift during copy/design execution.
4. **Verification steps are not operationalized enough** to catch failure before PPT assembly.

### What would make it execution-ready
Before moving to `FINAL_SLIDE_COPY.md`, the plan package needs:
- synchronized slide order across all planning docs
- one locked primary proof anchor
- explicit acceptance criteria
- explicit verification checklist

Once those four items exist, the plan should be ready for copy and deck production.

---

## 8) Summary judgment

### What is strong
- Corrected the main architectural mistake.
- Better aligned to jury and congress needs.
- Threads ecology and deployment more intelligently.
- Contains much better visual discipline.
- Preserves the true reason the team became a finalist: disciplined research workflow + honest validation + deployable interpretation.

### What still blocks approval
- Planning artifacts are not yet mutually consistent.
- Acceptance criteria are not sufficiently testable.
- Verification is present but not concrete enough.
- Proof anchor is still unresolved.

---

## Required revision list before approval
1. **Synchronize** `STORYTELLING_MAP.md` and `FINAL_SLIDE_PLAN.md` to the revised hybrid order.
2. **Lock the primary proof anchor** and add a fallback.
3. **Add explicit acceptance criteria** with pass/fail thresholds.
4. **Add concrete verification steps** with owner + pass condition.
5. **Add a compact production-risk section** covering timing, evidence availability, and clutter drift.

Once those are done, this plan should be approvable and ready to drive:
- `FINAL_SLIDE_COPY.md`
- `PITCH_SCRIPT_4MIN.md`
- `FIGURE_SPECS.md`
- final deck production
