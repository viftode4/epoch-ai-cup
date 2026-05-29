# RALPLAN Initial Plan — AI Cup 2026 Congress Presentation

_Last updated: 2026-04-11_

## Scope
Initial planning artifact for the 4-minute AI Cup 2026 congress pitch, grounded in:
- `.omx/context/congress-presentation-plan-20260411T155346Z.md`
- `docs/JURY_TARGETING_PLAYBOOK.md`
- `docs/CONGRESS_PROFILE.md`
- `docs/presentation/STORYTELLING_MAP.md`
- `docs/presentation/FINAL_SLIDE_PLAN.md`
- `docs/presentation/DESIGN_DIRECTION_FOR_CONGRESS.md`
- `docs/presentation/VISUAL_STORYBOARD.md`

This plan is presentation-first. It assumes the deck must win both the jury room and the broader Dutch AI Congress audience while staying truthful to the team’s actual research workflow, system architecture, and experimentation history.

---

# 1) RALPLAN-DR Summary

## Principles
1. **Research workflow is part of the product.** The pitch must present not only the final model/system, but the disciplined hypothesis -> experiment -> validation -> keep/reject loop that got the team here.
2. **Win the room with clarity, not compression.** In 4 minutes, reduce complexity into one coherent system story instead of trying to cover every section of the written proposal.
3. **Show rigor through selective evidence.** Include only proof that supports trust, generalization, and deployment readiness; omit detail that behaves like appendix material.
4. **Design should feel like a congress-ready research notebook.** Use the Math Notebook Airspace visual system to signal technical seriousness, fair-tech alignment, and visual memorability without feeling empty or overdesigned.
5. **Every slide must do double duty.** Each slide must help both the jury (rigor, ecology, deployment, productization) and the wider congress audience (clarity, importance, team identity, memorability).

## Decision Drivers (Top 3)
1. **4-minute hard constraint:** the story must be extremely compressive while still showing seriousness, originality, and deployment value.
2. **Audience mix:** the room includes jurors plus a broader AIC4NL congress audience that rewards responsible AI, public-private value, and deployable systems.
3. **Team differentiation:** the strongest distinctive asset is not just model performance but the team’s research workflow, iteration discipline, and evidence-based system design.

## Viable Options

### Option A — Model-first competition deck
**Shape:** problem -> features/models -> result -> deployment note.

**Pros**
- Easy to build quickly.
- Feels conventionally technical.
- Makes model/system details explicit.

**Cons**
- Undersells the workflow and team capability that actually differentiate this team.
- Risks sounding like a leaderboard recap.
- Weak fit for Marelle, Andy, and the broader congress room.
- Leaves little room for the “why trust us” / “why this matters” story.

### Option B — Workflow-first congress deck
**Shape:** real-world problem -> research workflow -> system -> insights -> trust/evidence -> deployment.

**Pros**
- Best fit for jury + congress dual audience.
- Makes the team memorable beyond the benchmark.
- Naturally supports the most-submissions award as disciplined experimentation.
- Creates a believable deployment story.

**Cons**
- Requires disciplined slide design to avoid feeling too abstract.
- Must carefully balance workflow with enough technical specificity.
- Needs strong evidence selection to avoid becoming “process-heavy.”

### Option C — Deployment-first product deck
**Shape:** operational problem -> user workflow -> system role -> value -> proof.

**Pros**
- Strong for industry / venture / congress audience.
- Good fit for Joep, Marelle, Andy.
- Makes real-world relevance immediate.

**Cons**
- Risks under-serving Geert-Jan and Bart on rigor/ecology.
- Can flatten the actual research contribution.
- May sound too commercial if not handled carefully.

## Recommended Option
**Option B — Workflow-first congress deck with strong deployment payoff.**

It best matches the repo’s existing docs, the selected design direction, the jury targeting guidance, and the congress profile. It also best captures why this team became a finalist: not one lucky model, but disciplined experimentation and system-building under real constraints.

---

# 2) Recommended Presentation Strategy for a 4-Minute Congress Pitch

## Strategic thesis
The pitch should present the team as:

> **a research-driven team that turned a hard real-world sensing problem into a rigorous, ecologically grounded, and realistically deployable AI system.**

## Talk strategy
- **Open with the shared-airspace problem** so everyone immediately understands relevance.
- **Move quickly to the team workflow** to establish that the team’s process is part of the reason the result is credible.
- **Show one clean architecture** to prove technical coherence.
- **Distill the big lessons learned** to elevate the team from model-builders to serious thinkers.
- **Use an evidence slide to justify trust** in both the system and the team.
- **Close on deployment/impact** so the final memory is “this can matter beyond the contest.”

## Desired audience reaction
- Jury: “They understand the domain, the metric, the shift problem, and the deployment story.”
- Congress audience: “These people are serious builders/researchers; I’d talk to them after.”

## Recommended pacing
- Slide 1: 20–25s
- Slide 2: 30–35s
- Slide 3: 35–40s
- Slide 4: 40–45s
- Slide 5: 35–40s
- Slide 6: 30–35s
- Slide 7: 20–25s

Total target: ~3:45–3:55, leaving buffer for stage pacing.

---

# 3) Explicit Slide Narrative and Message Hierarchy

## Master hierarchy
### Level 1 — Main message of the deck
We are not just a model team; we are a research-driven team that built a trustworthy, deployable AI system for biodiversity-aware wind-energy decisions.

### Level 2 — Supporting messages
1. The problem is real and multi-stakeholder.
2. Our workflow created the result.
3. Our system architecture is coherent and metric-aligned.
4. Our strongest contribution is robust judgment under shift.
5. Our evidence shows trustworthiness and seriousness.
6. The output has a believable operational use case.

### Level 3 — Proof points
- seasonal/month shift was a first-class challenge
- macro-mAP/ranking alignment mattered
- feature stack combined motion + context + validated signals
- brittle methods were rejected
- the team won the most submissions award via structured experimentation
- final output is decision support, not blind automation

## Slide-by-slide narrative

### Slide 1 — Opening / identity
**Message:** Bird-safe wind energy needs trustworthy AI.
**Job:** Make the audience care and frame the team as serious.
**Hierarchy:** problem relevance > team identity > technical seriousness.

### Slide 2 — Problem framing
**Message:** This is not just a classification task; it is an ecology + operations + sensing challenge.
**Job:** Establish why accuracy alone is insufficient.
**Hierarchy:** shared-airspace conflict > operational constraints > radar classification difficulty.

### Slide 3 — Team workflow
**Message:** What got us here was disciplined experimentation, not a one-off modeling trick.
**Job:** Sell the research loop and most-submissions award as evidence of throughput and judgment.
**Hierarchy:** method > iteration discipline > team credibility.

### Slide 4 — System architecture
**Message:** We built a coherent, metric-aware, shift-aware system.
**Job:** Make the technical approach understandable in one glance.
**Hierarchy:** inputs > features > models > decision-support output, with shift as a highlighted cross-cutting constraint.

### Slide 5 — What we learned
**Message:** The project produced understanding, not just a score.
**Job:** Show the team’s strongest conceptual insights.
**Hierarchy:** ranking alignment > shift awareness > judgment over tricks.

### Slide 6 — Why trust the result
**Message:** Trust comes from breadth of experimentation, honest validation, and defensible choices.
**Job:** Validate the team and the system without resorting to dense benchmark tables.
**Hierarchy:** finalist status / experiment scale / submissions award / honest validation.

### Slide 7 — Deployment / closing
**Message:** This work matters because it fits a real decision-support workflow.
**Job:** End on impact, responsibility, and deployability.
**Hierarchy:** radar monitoring > AI ranking + uncertainty > expert review > targeted mitigation.

---

# 4) Visual-System Decisions Tied to Storytelling

## Chosen visual direction
**Math Notebook Airspace**
- headline font: Archivo
- body font: IBM Plex Sans
- annotations/technical notes: IBM Plex Mono
- palette: warm paper neutral, dark ink, signal orange, validation teal

## Why this visual system is the right fit
- It supports the repo’s desired “simple but effective, not empty” rule.
- It visually encodes “research-driven” without becoming messy.
- It matches the DeFabrique / AIC4NL fair-tech tone better than a slick startup deck or a sterile academic deck.
- It helps the team look like a serious applied-AI lab rather than a student competition team.

## Storytelling-to-visual mapping
- **Opening/problem slides:** overlap diagrams, trajectory traces, turbine silhouettes, shared-airspace cues.
- **Workflow slide:** pipeline with notebook annotations and keep/reject logic.
- **Architecture slide:** one dominant flow diagram with a highlighted seasonal-shift constraint box.
- **Insight slide:** three substantial cards with equation-like or notebook-style marginalia.
- **Proof slide:** lab-report style evidence blocks instead of tables.
- **Deployment slide:** operational flow with expert review as an explicit visual node.

## Animation / reveal policy
- Use **progressive reveal** only where it improves comprehension.
- Prefer 1–2 reveals on workflow, architecture, and deployment slides.
- Avoid decorative transitions.
- Each reveal should map to a verbal step in the script.

## “Not empty” enforcement
Each slide must contain:
- one strong title
- one anchor sentence or subtitle
- one dominant visual structure
- one technical annotation layer
- one motif layer (grid, trajectory trace, hashed figure edge, silhouette)

---

# 5) Proof / Evidence Policy

## Include
- evidence that proves seriousness, rigor, or operational plausibility
- evidence that demonstrates the team’s workflow and learning process
- evidence that supports the chosen key insights
- finalist status and most-submissions award, framed as outcomes of disciplined work
- selective numeric or comparative evidence only if it is instantly legible and defensible

## Omit
- dense benchmark tables
- exhaustive method lists
- every feature family, ablation, or experiment detail
- details that require appendix-level explanation
- unsupported performance claims
- visuals that look like Kaggle screenshots unless they directly reinforce trust

## Evidence hierarchy
1. **Trust evidence** — finalist status, breadth of experiments, honest validation under shift
2. **Conceptual evidence** — what changed because of month shift and metric alignment
3. **Operational evidence** — why decision-support framing is justified
4. **Selective quantitative evidence** — only if it strengthens the above and can be read in under 3 seconds

## Rule of thumb
If a piece of evidence does not improve either **trust**, **understanding**, or **memorability**, it should be omitted from the main deck and moved to speaker notes or Q&A backup.

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
- why the team’s experimentation workflow is an asset
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
- final slide narrative
- key claims per slide
- evidence inclusion list
- speaker ownership / timing map

## Phase 2 — Copy drafting
Produce:
- exact slide titles
- anchor sentence per slide
- on-slide text blocks/cards
- full 4-minute script
- backup shorter script version (~3:30)

## Phase 3 — Visual translation
Produce:
- figure specs per slide
- icon/silhouette/trace needs
- reveal order and animation notes
- proof-card structure
- consistent annotation vocabulary

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
- final trim pass for any visual clutter or verbal over-explanation

---

# 8) ADR

## Decision
Use a **workflow-first congress deck** in the **Math Notebook Airspace** visual system, built around a 7-slide story: problem -> workflow -> system -> insights -> trust -> deployment.

## Drivers
- 4-minute time limit forces compression and narrative discipline.
- Jury + congress audience mix requires both rigor and memorability.
- Team differentiation is strongest in research workflow, shift-aware reasoning, and deployability.

## Alternatives considered
### Alternative 1 — Model-first deck
Rejected because it overweights technical specifics and underweights the workflow and team identity that make the pitch more persuasive for this event.

### Alternative 2 — Deployment-first product deck
Rejected because it risks underrepresenting the team’s methodological depth and ecological rigor, which are crucial for Bart and Geert-Jan and for overall trust.

### Alternative 3 — Proposal-section deck (Digital System / Data / Responsible Use / Mitigation)
Rejected because it mirrors the written implementation proposal too literally and likely hurts flow in a 4-minute live presentation.

## Why chosen
The chosen structure best reconciles:
- jury priorities
- congress room vibe
- selected visual direction
- the team’s true story
- the need to look serious, memorable, and deployable

## Consequences
- The deck must be highly selective and cannot include everything.
- Numeric results must be carefully curated rather than dumped.
- Design and copy need to be tightly integrated; placeholders will weaken the whole effect.
- Q&A and backup material become more important because some detail is intentionally omitted from the main deck.

## Follow-ups
- lock final slide copy
- lock evidence choices
- build the juror-specific Q&A sheet
- build the final deck in PPTX
- rehearse timing and refine density

---

# 9) Concrete Next-Step Artifact List

## Priority 1 — Content lock
1. `docs/presentation/FINAL_SLIDE_COPY.md`
   - exact on-slide copy for all 7 slides
2. `docs/presentation/PITCH_SCRIPT_4MIN.md`
   - full speaker script aligned to reveals
3. `docs/presentation/QA_BATTLE_SHEET.md`
   - juror-group answer bank

## Priority 2 — Visual production
4. `docs/presentation/FIGURE_SPECS.md`
   - figure-by-figure build instructions for each slide
5. `docs/presentation/DECK_STYLE_GUIDE.md`
   - final typography, spacing, card styles, annotation vocabulary, animation rules
6. final PPTX in `docs/presentation/`
   - built in the Math Notebook Airspace style

## Priority 3 — Rehearsal / supporting materials
7. `docs/presentation/SPEAKER_HANDOFFS.md`
   - who says what, timing, and transition lines
8. `docs/presentation/QA_BACKUP_SLIDES.md` or appendix PPTX notes
   - optional backup proof visuals
9. `docs/presentation/FINAL_REHEARSAL_CHECKLIST.md`
   - timing, readability, confidence, and final export checks

---

# Planning Note
This plan is intentionally optimized around **why this team became a finalist**:
- not a single clever trick
- but disciplined experimentation
- honest validation under shift
- system thinking
- and a deployment-aware interpretation of the challenge
