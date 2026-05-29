# AI Cup Congress Pitch — Final Slide Plan

_Last updated: 2026-04-11_

## Design rule for this deck
The slides should feel:
- **simple but not empty**
- **clean but not sterile**
- **designed but not decorative**
- **technical but still understandable in one glance**

## Density rule
Each slide should contain:
- **1 main message**
- **1 main visual structure**
- **2–4 supporting points max**
- **1 visual accent / technical figure / annotation layer**

## Recommended total
**7 slides**
This is enough for 4 minutes if each slide gets ~25–40 seconds.

## Locked order
1. Opening / identity
2. Problem
3. System snapshot
4. Workflow as credibility engine
5. What we learned
6. Proof / why trust it
7. Deployment / closing

---

# Slide 1 — Opening / identity
## Goal
Win attention immediately and frame the team correctly.

## Title
**Bird-safe wind energy needs trustworthy AI**

## Supporting subtitle
We are not just presenting a model.
We are presenting a rigorous way of building deployable AI for a difficult real-world problem.

## What to show
### Left side
- strong title
- one short subtitle
- 3 small badges:
  - biodiversity
  - wind energy
  - deployable AI

### Right side
A large airspace-style technical figure:
- trajectory traces
- one or two turbine silhouettes
- notebook annotations like:
  - `airspace overlap`
  - `real-world sensing problem`

## Fullness strategy
Use:
- visible figure work
- subtle grid / paper texture
- one technical annotation cluster

---

# Slide 2 — The problem
## Goal
Make the audience care about the problem in operational terms.

## Title
**Why this problem matters**

## Main statement
Wind energy and biodiversity share the same airspace.

## Content structure
Use **3 strong cards**:

### Card 1 — Ecology
Bird strikes matter most when migration dynamics and turbine activity overlap.

### Card 2 — Operations
Shutdowns are costly, so mitigation has to be targeted and defensible.

### Card 3 — Monitoring challenge
Radar is scalable, but bird-group classification is noisy, imbalanced, and seasonally shifting.

## Visual addition
Behind or beside the cards:
- soft trajectory traces
- one diagrammatic bracket tying the 3 cards together
- a tiny note: `not just an accuracy problem`

## Fullness strategy
Cards should be visually rich enough:
- border
- slight texture / hash corner
- one accent line or small icon/figure per card

---

# Slide 3 — System snapshot
## Goal
Explain the architecture clearly and early.

## Title
**System in one glance**

## Main structure
A **4-part system flow**:
- inputs
- feature stack
- models
- decision support output

## Suggested wording
### Inputs
radar trajectories + environmental context

### Feature stack
kinematics + signatures + catch22 + validated context

### Models
ranking-aware ensemble + multiclass probability model

### Output
probabilistic bird-group ranking for operator / ecologist review

## Critical side annotation
A highlighted constraint box:
**The hardest part was not only class imbalance.
It was seasonal / month shift.**

## Ecology thread to show explicitly
- radar is an ecological sensor, not a species oracle
- seasonality reflects migration behavior
- output is decision support, not autopilot

## Fullness strategy
Use:
- one dominant central figure
- one constraint panel
- subtle overlay traces
- 2–3 mono notes max

---

# Slide 4 — Workflow as credibility engine
## Goal
Show that the workflow is why the system deserves trust.

## Title
**Why our workflow made it robust**

## Core statement
We did not keep everything we tried.
We kept what survived honest validation under shift.

## Main visual
A **research engine / filter pipeline**:
1. hypothesis
2. experiment
3. honest validation
4. keep / reject
5. robust system

## Supporting text
Under or beside the pipeline:
- fast iteration
- strong judgment
- failed ideas were tracked, not hidden
- small secondary badge: **most submissions award**

## Visual additions
- arrows with mono labels
- notebook notes like:
  - `generalizes?`
  - `reject if brittle`
  - `keep only what survives shift`

## Fullness strategy
Use:
- step blocks
- arrows
- hash fills
- side notes
- small throughput badge only

---

# Slide 5 — Key insight / what we learned
## Goal
Show that the project produced understanding, not just a result.

## Title
**What actually moved the needle**

## Main structure
Use **3 large insight cards**:

### Insight 1
**Ranking, not just classification**
Macro-mAP rewards ordering, so metric alignment mattered.

### Insight 2
**Shift was a first-class failure mode**
Ideas that looked good locally often failed across months.

### Insight 3
**Ecology had to stay inside the model**
Movement, radar, and context had to stay connected to migration and mitigation logic.

## Visual additions
- one notebook annotation in margin:
  `workflow -> insight -> system`
- one tiny label:
  `generalization > local gain`

## Fullness strategy
Use:
- large headings
- short body copy
- hashed corners / technical framing
- one connective annotation line or bracket

---

# Slide 6 — Why trust us / why trust the result
## Goal
Build credibility for both the jury and the wider room.

## Title
**Why we trust this result**

## Main structure
A **proof grid**:
- **Primary proof anchor:** AI Cup 2026 finalist outcome
- **Support 1:** robust validation under shift
- **Support 2:** breadth of experiments / rejected brittle methods
- **Secondary chip only:** most submissions award

## Supporting statement
The point is not that we solved everything.
The point is that we built a stronger, more defensible system through disciplined experimentation and deployment thinking.

## Fullness strategy
This slide should feel evidence-rich, not text-heavy.
Use compact proof blocks with strong labels.

---

# Slide 7 — Deployment / closing
## Goal
Show why the work matters beyond the competition and leave the room with one strong final impression.

## Title
**How this gets used in the real world**

## Main structure
Operational flow:
1. radar monitoring
2. AI ranking + uncertainty
3. operator / ecologist review
4. targeted mitigation action

## Closing statement
The strength of our work is not only that it performs well,
but that it is rigorous, ecologically grounded, and realistically deployable.

## Operational phrases to preserve
- better targeted mitigation
- more defensible shutdown decisions
- scalable monitoring where manual observation is hard

## Fullness strategy
Make the operational flow large and strong.
The close should feel like a payoff, not an afterthought.

---

# Overall visual system
## Repeating motifs
Use these across slides so the deck feels coherent:
- trajectory traces
- notebook / math grid
- hash fills / sketched corners
- mono annotations
- turbine silhouettes
- structured cards

## Color logic
- **Orange** = urgency / energy / migration pressure / action
- **Teal** = trust / validation / control / ecological support
- **Warm paper** = fair-tech / civic / real-world tone
- **Dark ink** = rigor / technical clarity

## Typography logic
- **Archivo** = strong headlines
- **IBM Plex Sans** = readable body copy
- **IBM Plex Mono** = annotations, notebook feel, system notes

---

# Slide-by-slide “not empty” checklist
Each slide should have:
- headline
- subheadline or anchor sentence
- one major central figure or card structure
- one secondary annotation layer
- one texture / trace / motif layer

If a slide has only title + 2 bullets, it will feel empty.
If a slide has structure + figure + annotation, it will feel designed.

---

# Final deck principle
This deck should feel like:
> a well-designed research notebook made for a congress stage

not:
> a student competition slideshow
