# RALPLAN Revised Plan — AI Cup 2026 Congress Presentation

_Last updated: 2026-04-11_

## Scope
Revised planning artifact for the 4-minute AI Cup 2026 congress pitch, updated from the initial plan using the architect review and grounded in:
- `.omx/context/congress-presentation-plan-20260411T155346Z.md`
- `docs/JURY_TARGETING_PLAYBOOK.md`
- `docs/CONGRESS_PROFILE.md`
- `docs/presentation/STORYTELLING_MAP.md`
- `docs/presentation/FINAL_SLIDE_PLAN.md`
- `docs/presentation/DESIGN_DIRECTION_FOR_CONGRESS.md`
- `docs/presentation/VISUAL_STORYBOARD.md`
- `docs/presentation/RALPLAN_ARCHITECT_REVIEW.md`

This revision keeps the deck presentation-first, but changes the narrative spine from workflow-first to a **system-anchored workflow hybrid**. The deck must now prove the system early, use workflow as the credibility engine, thread ecology through the technical middle, and show one concrete result/proof anchor without turning into a benchmark dump.

---

# 1) RALPLAN-DR Summary

## Principles
1. **Lead with the object, not the meta-story.** The room must understand what we built before it is asked to care about how we built it.
2. **Workflow is still core, but as credibility infrastructure.** The hypothesis -> experiment -> validation -> keep/reject loop should explain why the final system is trustworthy, not replace the system as the protagonist.
3. **Ecology must stay inside the technical core.** Radar is an ecological sensor, month shift reflects migration behavior, and the output is for mitigation support rather than abstract classification.
4. **Proof must be visible, selective, and fast to read.** Include one concrete result anchor plus robust-validation evidence; omit dense tables and anything that cannot be understood in seconds.
5. **Design should feel like a restrained research notebook.** Use the Math Notebook Airspace direction with hard visual restraint rules so the deck feels full, technical, and memorable without becoming cluttered.

## Decision Drivers (Top 3)
1. **4-minute hard constraint:** the deck needs an early concrete system anchor, not a slow build-up.
2. **Audience mix:** the jury and congress room both reward clarity, deployability, and trustworthy AI, but Bart and Geert-Jan especially need system/ecology/validation logic earlier.
3. **Team differentiation:** what got the team here was not just a final score but disciplined research workflow, honest validation under shift, and system-building judgment.

## Viable Options

### Option A — Model-first competition deck
**Shape:** problem -> system details -> result -> deployment note.

**Pros**
- Fast to understand technically.
- Gives an early concrete object.
- Easy to defend in a benchmark-style room.

**Cons**
- Undersells workflow and team capability.
- Risks sounding like a leaderboard recap.
- Weak fit for the broader congress audience.

### Option B — Workflow-first congress deck
**Shape:** problem -> workflow -> system -> insights -> trust -> deployment.

**Pros**
- Makes the team memorable.
- Highlights experimentation discipline.
- Fits the broader congress room.

**Cons**
- System arrives too late for a hard 4-minute room.
- Risks sounding self-referential before proving the object.
- Under-serves early ecology/system credibility.

### Option C — System-anchored workflow hybrid
**Shape:** opening -> problem -> system snapshot -> workflow as credibility engine -> insights -> proof/results -> deployment.

**Pros**
- Gives the room an early concrete object.
- Preserves workflow as a differentiator without over-centering it.
- Better fit for jury + congress + visual direction.
- Makes ecology and validation easier to thread through the middle.

**Cons**
- Requires tighter slide discipline.
- Needs very selective proof choices.
- Workflow slide must avoid slipping back into “look how much we did.”

## Recommended Option
**Option C — System-anchored workflow hybrid.**

It best incorporates the architect critique while preserving the initial strategy’s strengths. It shows what the system is early, explains why the workflow made it robust, keeps ecology inside the technical story, and ends with operational payoff.

---

# 2) Recommended Presentation Strategy for a 4-Minute Congress Pitch

## Strategic thesis
The pitch should present the team as:

> **a research-driven team that built a trustworthy, ecologically grounded, and realistically deployable AI system for biodiversity-aware wind-energy decisions.**

## Talk strategy
- **Open with the shared-airspace problem** so relevance is immediate.
- **Show the system in one glance early** so the room has a concrete object to hold onto.
- **Use workflow after the system** to explain why the final design deserves trust.
- **Thread ecology through the technical middle** so features, shift, and output stay tied to bird movement and mitigation.
- **Show one visible proof anchor** plus robust-validation evidence.
- **Close on operational change** so the final memory is practical and consequential.

## Desired audience reaction
- Jury: “They showed the system early, they understand ecology and shift, and their process makes the result believable.”
- Congress audience: “These are serious builders/researchers with a real system and a credible way of working.”

## Recommended pacing
- Slide 1: 20–25s
- Slide 2: 25–30s
- Slide 3: 35–40s
- Slide 4: 35–40s
- Slide 5: 35–40s
- Slide 6: 35–40s
- Slide 7: 20–25s

Total target: ~3:45–3:55.

---

# 3) Explicit Slide Narrative and Message Hierarchy

## Master hierarchy
### Level 1 — Main message of the deck
We are not just a model team; we are a research-driven team that built a trustworthy, deployable AI system for biodiversity-aware wind-energy decisions.

### Level 2 — Supporting messages
1. The problem is real and multi-stakeholder.
2. The system is coherent, metric-aware, and shift-aware.
3. The workflow made the system robust.
4. Ecology shaped the technical choices.
5. Trust comes from selective proof and honest validation.
6. The output has a believable operational role.

### Level 3 — Proof points
- wind energy and biodiversity share the same airspace
- radar is an ecological sensor, not a species oracle
- month shift reflects migration behavior and operating variation
- motion + context + validated signals outperformed brittle shortcuts
- macro-mAP/ranking alignment changed the modeling strategy
- finalist status and one direct result anchor prove outcome
- most-submissions award is a secondary signal of disciplined throughput, not a headline proof
- final output supports mitigation, not blind automation

## Slide-by-slide narrative

### Slide 1 — Opening / identity
**Message:** Bird-safe wind energy needs trustworthy AI.
**Job:** Make the audience care and position the team as serious, domain-aware builders.
**Hierarchy:** problem relevance > team identity > technical seriousness.

### Slide 2 — Problem framing
**Message:** This is an ecology + operations + sensing challenge, not a simple classification exercise.
**Job:** Establish why pure accuracy is insufficient and why seasonality/uncertainty matter.
**Hierarchy:** shared-airspace conflict > migration/seasonality > monitoring and mitigation constraints.

### Slide 3 — System snapshot
**Message:** We built a system that combines trajectory behavior, contextual signals, and ranking-aware modeling into decision support.
**Job:** Give the room an early concrete object.
**Hierarchy:** inputs > ecological/behavioral features > ranking-aware output > uncertainty-aware support.
**Ecology thread:** explicitly note radar as ecological sensor; month shift is migration behavior, not just dataset drift.

### Slide 4 — Why our workflow made it robust
**Message:** The final system came from a disciplined engine that filtered noise into robust choices.
**Job:** Show hypothesis -> experiment -> validation -> keep/reject as the reason the system deserves trust.
**Hierarchy:** honest validation > rejected brittle ideas > retained robust components > experimentation discipline.
**Most-submissions award treatment:** a small secondary badge/callout only.

### Slide 5 — What we learned
**Message:** The project produced real understanding: ranking mattered, shift mattered, and ecology had to stay inside the model logic.
**Job:** Elevate the team from model-tuners to thoughtful researchers/builders.
**Hierarchy:** ranking alignment > migration/seasonal shift > ecological/contextual meaning > judgment over tricks.

### Slide 6 — Why trust it / proof
**Message:** Trust comes from a visible result anchor, robust validation, and breadth of disciplined experimentation.
**Job:** Prove outcome without drowning the slide in numbers.
**Hierarchy:** one direct result/proof anchor > validation under shift > breadth/rejected methods > small submissions-award chip.

### Slide 7 — Deployment / closing
**Message:** This system matters because it can change real monitoring and mitigation decisions.
**Job:** End on practical value, responsibility, and deployability.
**Hierarchy:** scalable monitoring > ranked uncertainty-aware support > expert review > better targeted mitigation / more defensible shutdown decisions.

---

# 4) Visual-System Decisions Tied to Storytelling

## Chosen visual direction
**Math Notebook Airspace**
- headline font: Archivo
- body font: IBM Plex Sans
- annotations/technical notes: IBM Plex Mono
- palette: warm paper neutral, dark ink, signal orange, validation teal

## Why this visual system is the right fit
- It supports “simple but effective, not empty.”
- It signals a serious, nerdy, research-driven identity without feeling juvenile.
- It matches the DeFabrique / AIC4NL fair-tech tone better than a glossy startup deck or sterile academic slides.
- It works best when attached to clear system visuals and real proof, not decorative noise.

## Storytelling-to-visual mapping
- **Opening/problem slides:** shared-airspace composition with turbine silhouette, bird/radar traces, and one ecological/operational tension callout.
- **System snapshot:** one dominant flow diagram showing inputs -> feature stack -> ranking-aware output, with ecological notes on movement and seasonality.
- **Workflow slide:** clean decision engine diagram with keep/reject logic and validation gates; small throughput badge only.
- **Insights slide:** three strong insight panels/cards with sparse equation-like notes and real terms like ranking, migration, and contextual signal.
- **Proof slide:** one result anchor block plus two supporting evidence blocks, styled like a lab note rather than a dashboard.
- **Deployment slide:** operational flow with explicit human review and mitigation decision nodes.

## Visual restraint rules
1. **Max 1 dominant figure per slide.**
2. **Max 2 annotation clusters per slide.**
3. **Max 1 hashed/sketch motif family per slide.**
4. **One accent color should dominate per slide; secondary accent only if functional.**
5. **Prefer real data-derived plots/trajectories where possible.** Invented doodles should frame content, not replace evidence.
6. **Notebook styling should support the story, not become the story.** Clean first, notebook second.
7. **No slide should feel empty, but no slide should need deciphering.**

## Animation / reveal policy
- Use progressive reveal only when it improves comprehension.
- Prefer 1–2 reveals on the system, workflow, and deployment slides.
- Avoid decorative transitions.
- Each reveal must correspond to a spoken step.

## “Not empty” enforcement
Each slide must contain:
- one strong title
- one anchor sentence/subtitle
- one dominant visual structure
- one supporting annotation layer
- one motif layer (grid, trace, hashed edge, silhouette, or figure callout)

---

# 5) Proof / Evidence Policy

## Include
- **one concrete result/proof anchor** that is instantly legible and defensible
- evidence of validation under month/season shift
- evidence that the workflow rejected brittle methods and kept robust ones
- finalist status as competitive credibility
- most-submissions award only as a secondary throughput signal
- operational evidence that supports mitigation/decision-support framing

## Omit
- dense benchmark tables
- exhaustive model/feature inventories
- anything that needs appendix-level explanation to be believed
- unsupported performance claims
- repeated mention of the submissions award
- decorative visuals masquerading as proof

## Proof hierarchy
1. **Result / competitive evidence**
   - finalist status and one direct result anchor (e.g. placement, clear relative improvement, or defensible robustness outcome)
2. **Robustness / honest validation under shift**
   - cross-month or season-aware validation signal
3. **Experiment breadth / rejected brittle methods**
   - breadth matters as evidence of judgment, not volume for its own sake
4. **Most-submissions award as throughput signal**
   - secondary chip only

## Result/proof anchor policy
The slide must contain **one primary proof object** that answers: “what concrete outcome proves this worked?”

Acceptable proof anchors:
- finalist/top placement framing
- one simple performance delta vs baseline
- one strong robustness callout across held-out months
- one compact “X experiments -> Y robust system choice -> finalist outcome” object

Selection rules:
- readable in under 3 seconds
- defensible in Q&A
- supports trust, not just hype
- tied to the system and validation story

## Rule of thumb
If a piece of evidence does not improve **trust**, **understanding**, or **memorability**, move it to speaker notes or backup slides.

---

# 6) Q&A Priorities by Juror Group

## Marelle van Beerschoten — adoption / clarity / practical AI
**Priority answers to prepare:**
- how this fits an operational workflow
- how human oversight is preserved
- why the solution is usable, not just accurate
- what trade-offs remain

## Joep Breuer — wind-energy ecology / mitigation / operations
**Priority answers to prepare:**
- how the output supports monitoring and mitigation decisions
- how errors matter operationally
- how the system behaves under seasonal / field variation
- why this is useful where manual monitoring is difficult

## Andy Lürling — productization / scale / impact
**Priority answers to prepare:**
- who the user/buyer is
- how this becomes a product or deployable capability
- why the workflow is a repeatable team asset
- how trust and adoption are handled

## Geert-Jan Houben — rigor / systems / generalization
**Priority answers to prepare:**
- why the validation is believable under shift
- how metric alignment influenced modeling
- which methods were rejected and why
- what makes the workflow reproducible and context-aware

## Bart Kranstauber — ecological validity / movement logic / uncertainty
**Priority answers to prepare:**
- why the features are biologically meaningful
- what radar can and cannot tell us
- why seasonality was treated centrally
- how uncertainty is handled in ecological decision support

## Cross-juror answer rule
Every answer should ideally follow:
**direct answer -> evidence -> limitation -> deployment implication**

---

# 7) Production Sequence: Plan -> Copy -> Visuals -> Final Deck

## Phase 1 — Planning lock
Create and approve:
- final slide narrative using the system-anchored hybrid
- key claims per slide
- primary proof anchor choice
- evidence inclusion list
- speaker ownership / timing map

## Phase 2 — Copy drafting
Produce:
- exact slide titles
- anchor sentence per slide
- on-slide text blocks/cards
- full 4-minute script
- backup shorter script version (~3:30)
- ecology-thread wording for system/insight/deployment slides

## Phase 3 — Visual translation
Produce:
- figure specs per slide
- which figures are real-data-derived vs illustrative
- reveal order and animation notes
- proof-card structure centered on one visible anchor
- consistent annotation vocabulary with restraint rules

## Phase 4 — Deck assembly
Build:
- final PPTX with the Math Notebook Airspace system
- speaker notes
- backup/static PDF
- optional backup Q&A slides or appendix visuals

## Phase 5 — Rehearsal + refinement
Run:
- timing test
- readability test from distance
- “one glance” test for each slide
- Q&A stress test by juror group
- final trim pass for clutter, repeated claims, or weak proof

---

# 8) ADR

## Decision
Use a **system-anchored workflow hybrid** in the **Math Notebook Airspace** visual system, built around a 7-slide story: opening -> problem -> system snapshot -> workflow as credibility engine -> insights -> proof/results -> deployment.

## Drivers
- 4-minute time limit forces an early concrete object and ruthless narrative discipline.
- Jury + congress audience mix requires both technical legitimacy and memorability.
- Team differentiation is strongest when workflow, ecology, shift-awareness, and deployability reinforce the system rather than compete with it.

## Alternatives considered
### Alternative 1 — Model-first deck
Rejected because it overweights technical specifics and underweights workflow, ecology framing, and team identity.

### Alternative 2 — Workflow-first deck
Rejected because it brings process too early and delays the system/object the room needs to trust.

### Alternative 3 — Deployment-first product deck
Rejected because it risks flattening the team’s methodological depth and ecological rigor.

### Alternative 4 — Proposal-section deck (Digital System / Data / Responsible Use / Mitigation)
Rejected because it mirrors the written proposal too literally and weakens 4-minute live flow.

## Why chosen
The chosen structure best reconciles:
- the architect critique
- jury priorities
- congress room vibe
- selected visual direction
- the team’s true story of disciplined research producing a robust system

## Consequences
- Workflow stays important but is no longer the deck’s main spine.
- The deck must show one concrete result/proof anchor.
- Ecology must stay explicit inside system and insights slides.
- The most-submissions award must stay small and secondary.
- Visual texture must be controlled by hard restraint rules.
- Q&A and backup material remain important because detail is intentionally compressed.

## Follow-ups
- choose the primary result/proof anchor
- lock final slide copy
- lock visual restraint/style guide
- build juror-specific Q&A sheet
- build final PPTX and rehearse against timing

---

# 9) Concrete Next-Step Artifact List

## Priority 1 — Content lock
1. `docs/presentation/FINAL_SLIDE_COPY.md`
   - exact on-slide copy for all 7 slides using the revised order
2. `docs/presentation/PITCH_SCRIPT_4MIN.md`
   - full speaker script aligned to reveals and proof anchor
3. `docs/presentation/QA_BATTLE_SHEET.md`
   - juror-group answer bank

## Priority 2 — Visual production
4. `docs/presentation/FIGURE_SPECS.md`
   - figure-by-figure build instructions for each slide, including ecology-thread visuals
5. `docs/presentation/DECK_STYLE_GUIDE.md`
   - final typography, spacing, annotation vocabulary, proof-card structure, and visual restraint rules
6. final PPTX in `docs/presentation/`
   - built in the Math Notebook Airspace style with the revised narrative spine

## Priority 3 — Rehearsal / supporting materials
7. `docs/presentation/SPEAKER_HANDOFFS.md`
   - who says what, timing, and transition lines
8. `docs/presentation/QA_BACKUP_SLIDES.md` or appendix PPTX notes
   - backup proof visuals and deeper method/evidence support
9. `docs/presentation/FINAL_REHEARSAL_CHECKLIST.md`
   - timing, readability, confidence, and export checks

---

# Planning Note
This revision is intentionally optimized around **why this team became a finalist**:
- not one clever trick
- not just raw submission volume
- but disciplined experimentation
- honest validation under shift
- ecological and system thinking
- and a deployment-aware interpretation of the challenge


---

# 10) Acceptance Criteria (Pass / Fail)

| Area | Pass condition |
|---|---|
| Narrative order | Final deck and all planning docs use: opening -> problem -> system snapshot -> workflow -> insights -> proof -> deployment |
| Timing | Full spoken script rehearses between **3:45 and 3:55** |
| Proof anchor | Slide 6 contains exactly **1 primary proof anchor** and at most **2 support blocks** + **1 optional award chip** |
| Ecology thread | Slides 2, 3, 5, and 7 each contain one explicit ecology / migration / mitigation statement |
| Award weighting | Most-submissions award appears **once only** and never as primary proof |
| Slide density | No main slide exceeds ~35 visible content words excluding micro-annotations; max 1 dominant figure and max 2 annotation clusters per slide |
| Readability | Each slide passes a 3-second glance test and is readable from projected distance |
| Deployment specificity | Slide 7 includes at least **2 concrete operational changes** (e.g. targeted mitigation, defensible shutdown decisions, scalable monitoring) |
| Evidence discipline | Every proof item is sourceable to repo evidence or public competition facts |

# 11) Verification Checklist

| Check | Owner | Pass condition |
|---|---|---|
| Doc sync | presentation lead | STORYTELLING_MAP.md, FINAL_SLIDE_PLAN.md, and revised plan all match the same 7-slide order |
| Proof anchor verification | evidence owner | Primary proof anchor chosen and backed by a sourceable repo/public fact |
| System glance test | design owner | Slide 3 is understandable in ~3 seconds without narration |
| Ecology thread check | content owner | Slides 2/3/5/7 each contain explicit ecology/mitigation phrasing |
| Award de-emphasis | content owner | Most-submissions award appears once as a small secondary chip only |
| Visual restraint check | design owner | No slide breaks max-figure / max-annotation / max-motif limits |
| Timing rehearsal | speaker lead | Full talk lands between 3:45 and 3:55 |
| Distance readability | rehearsal reviewer | Titles, anchor lines, and proof objects remain legible from room distance |
| Q&A readiness | team | At least one prepared answer exists for each juror priority cluster |

# 12) Production Risks and Mitigations

## Risk: proof anchor stays vague
- **Primary choice:** finalist outcome / finalist placement framing
- **Fallback:** compact robustness callout tied to shift-aware validation
- **Mitigation:** lock one source-backed proof object before writing final slide copy

## Risk: timing overruns
- **Cut first:** extra workflow commentary
- **Cut second:** secondary proof detail
- **Cut third:** explanatory sub-lines on problem slide

## Risk: visuals become decorative clutter
- **Mitigation:** max 1 dominant figure, max 2 annotation clusters, max 1 sketch/hash motif family per slide

## Risk: ecology disappears from middle slides
- **Mitigation:** require explicit ecology/migration/mitigation wording on slides 3 and 5 during copy review

## Risk: evidence object unavailable or weak
- **Mitigation:** choose only repo/publicly supportable evidence; if unavailable, use finalist outcome + experiment breadth combination as fallback
