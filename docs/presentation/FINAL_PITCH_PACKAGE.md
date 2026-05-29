# Final Pitch Package — AI Cup 2026 Congress

_Last updated: 2026-04-13_

## What this pitch is trying to win
This presentation has to do three things at once:
1. **convince the jury** that our work is strong, rigorous, and deployable;
2. **make the congress room remember us** as one of the strongest teams there;
3. **position the team** as more than a competition team — as a group that knows how to build serious applied AI.

The pitch therefore should not sound like:
- a Kaggle recap,
- a model zoo,
- or a benchmark diary.

It should sound like:

> **a serious team explaining how it turned a hard ecological sensing problem into a robust and deployable AI system.**

---

# Research-backed pitch principles used here
This package integrates the earlier jury/venue/design work with presentation guidance from reputable sources.

## The principles
1. **Lead with what it is, fast.**  
   YC guidance stresses that the audience must understand what you do quickly and in simple language. The audience does not reward sounding impressive; it rewards clarity.

2. **Short talks need a clear story spine.**  
   Stanford pitch guidance frames good pitches as a story with an expected arc: problem -> solution -> evidence. Do not move on before the previous step is clear.

3. **Each slide needs a take-home message.**  
   Nature presentation guidance emphasizes that each slide should communicate one main point and be recoverable even if the audience was briefly distracted.

4. **Evidence must be selective and legible.**  
   Aaron Harris / YC-style advice strongly warns against confusing charts and overloaded slides. One visible proof point beats many unreadable ones.

5. **The room remembers your strongest distinction.**  
   For us, that distinction is not just the model. It is the combination of research workflow, shift-aware judgment, ecological grounding, and deployment thinking.

### Sources
- Y Combinator — How to Pitch Your Company: https://www.ycombinator.com/blog/how-to-pitch-your-company/
- Stanford — Giving a Great Pitch: https://web.stanford.edu/class/cs224g/lectures/Giving_a_Great_Pitch.pdf
- Nature Communications — each slide should have a single-sentence take-home message: https://www.nature.com/articles/s41467-020-18656-6
- Nature Scitable — slide titles should state the message, not just the topic: https://www.nature.com/scitable/topicpage/presentation-slides-13905480/
- Aaron Harris — tactical short-pitch advice: https://blog.aaronkharris.com/advice-on-pitching

---

# Our strongest differentiators versus other teams
We do not need to say what other teams did wrong.
We need to make it obvious what makes **our team** stronger.

## Differentiator 1 — We understood the real problem better
We did not treat this as a generic classification task.
We treated it as an **ecology + operations + sensing** problem.

## Differentiator 2 — We understood the real technical challenge better
We realized that the hard part was not just class imbalance.
It was **generalization under seasonal / month shift**.

## Differentiator 3 — We built a system, not just a submission
Our story is not one lucky model.
It is a coherent pipeline from sensing to decision support.

## Differentiator 4 — Our workflow itself is a proof of strength
We used a disciplined loop of:
- hypothesis,
- experiment,
- honest validation,
- keep / reject.

That process was strong enough that it even earned us the **most submissions award**, but the real point is what it produced: judgment.

## Differentiator 5 — We are stronger on deployment
We can describe how the output fits into:
- monitoring,
- review,
- mitigation,
- and more defensible shutdown decisions.

## Differentiator 6 — We can explain why our choices make sense
We can explain:
- why ranking mattered,
- why shift mattered,
- why movement/context features mattered,
- and why some ideas were rejected.

That is stronger than just saying “this model performed best.”

---

# Biggest defensible claims we can make
These are the strongest claims we can make **without overclaiming**.

## Claim 1
> **We built a trustworthy AI system for bird-aware wind-energy decision support, not just a competition classifier.**

Why defensible:
- matches the system framing,
- matches the ecology/operations/problem framing,
- does not claim full autonomy.

## Claim 2
> **Our strongest contribution is the combination of disciplined experimentation, honest validation under shift, and deployment-aware system design.**

Why defensible:
- consistent with repo artifacts and workflow docs,
- consistent with your most-submissions signal,
- consistent with the final architecture story.

## Claim 3
> **The hardest challenge was not only imbalance, but generalization under seasonal shift.**

Why defensible:
- strongly supported by your own architecture/reporting docs.

## Claim 4
> **We treated radar as an ecological sensor, not as a species oracle.**

Why defensible:
- strong framing for Bart/Joep,
- humble enough to remain believable,
- helps avoid overclaiming.

## Claim 5
> **We aligned our modeling to the actual evaluation logic by treating this as a ranking problem.**

Why defensible:
- directly connected to macro-mAP reasoning.

## Claim 6
> **The final result is competitive, but more importantly, it is a system we can explain and defend.**

Why defensible:
- keeps the pitch from sounding leaderboard-first.

---

# Claims to avoid
These would weaken trust.

Do **not** say:
- “We solved bird detection.”
- “Our model identifies species exactly.”
- “This system can automate mitigation decisions.”
- “We proved it generalizes everywhere.”
- “We won because we worked harder than everyone else.”
- “We made the most submissions, therefore we are best.”

Instead say:
- “decision support”
- “probabilistic output”
- “expert review”
- “robust under the shift we tested”
- “disciplined experimentation”

---

# What questions we WANT to trigger
The pitch should pull the room toward questions we answer well.

## Desired questions
- How would this be used in real wind-farm monitoring?
- Why do you believe it generalizes better?
- What made your workflow different?
- How do you handle uncertainty?
- Why are your features ecologically meaningful?
- How could this become something deployable or scalable?

## Questions we want to avoid inviting
- What exact model architecture trick gave the best score?
- Did you just brute-force the leaderboard?
- Are you overclaiming what radar can do?
- Is this just a competition artifact?

---

# How to shape the room strategically
## Opening should signal to everyone
- **Marelle:** practical and usable
- **Joep:** mitigation relevance
- **Andy:** team capability and product potential
- **Geert-Jan:** rigor and system coherence
- **Bart:** ecological seriousness

## Middle should satisfy the technical jurors
- system early
- shift early
- ecology inside the system
- workflow as credibility, not self-congratulation

## End should satisfy the whole congress room
- deployment
- human oversight
- practical usefulness
- strong final identity line

---

# Research Copilot — how it should affect this pitch
I checked `G:/Projects/research-copilot`.

## What it is, in one paragraph
Research Copilot appears to be a **terminal-first research operations system** with:
- a human TUI,
- agent-safe JSON commands,
- explicit onboarding / triage / experiment / review workflows,
- workspace-local state,
- and ultrawork profiles for structured research execution.

It looks like a real system for making ML research more structured, inspectable, and operable.

## Should it be in the main AI Cup pitch?
### Current recommendation
**Not as a central object of the AI Cup pitch** unless it was directly part of the AI Cup solution workflow and you want the team/tooling to be part of the judged story.

### Best default position
Use it as a **Q&A / team-capability / follow-up** point:
- “One reason our workflow was so disciplined is that we care a lot about research operations and reproducibility.”
- then mention Research Copilot if asked how you structured research.

### Risk of putting it in the core pitch
It could distract from the AI Cup system itself.
The central pitch already has enough weight:
- problem,
- system,
- workflow,
- insights,
- proof,
- deployment.

If you insert another product, the story may split.

## Best way to mention it if useful
> “One thing that also shaped our process as a team is that we’ve been building structured research tooling, so we think a lot about how experimentation becomes reproducible and reviewable.”

That keeps it in the background unless the audience asks more.

---

# Final 4-minute pitch

## 0:00–0:25 — Opening
Bird-safe wind energy needs trustworthy AI.
That is the challenge we worked on in AI Cup 2026.
And the reason it matters is simple: wind energy and biodiversity share the same airspace. If we want cleaner energy without avoidable ecological damage, then monitoring and mitigation have to become more precise.

## 0:25–0:55 — Problem framing
Radar gives us a scalable way to observe what moves through that airspace, but classification is hard. The data are noisy, imbalanced, and strongly affected by seasonal change. So this was never just a machine-learning benchmark. It was an ecology, operations, and sensing problem at the same time.

## 0:55–1:35 — System in one glance
Our system combines radar trajectories with environmental context and turns them into uncertainty-aware decision support.
We start from movement and context, build a feature stack that captures behavior and validated external signal, and then use ranking-aware modeling to produce probabilistic bird-group outputs for expert review.
One of the most important design choices was to treat radar as an ecological sensor, not as a species oracle. And one of the hardest constraints was not only class imbalance, but seasonal or month shift, which reflects real migration behavior and changing operating conditions.

## 1:35–2:10 — Workflow / why trust us
What got us to a strong final system was our workflow.
We did not keep everything we tried. We worked in a disciplined loop of hypothesis, experiment, honest validation, and keep-or-reject decisions. That process filtered attractive but brittle ideas out of the system and preserved the components that actually generalized under shift.
That experimentation discipline even earned us the award for most submissions — but the more important result was judgment, not volume.

## 2:10–2:50 — What we learned
Three insights changed the project.
First, this was a ranking problem, not just a classification problem, because macro-mAP rewards ordering.
Second, shift had to be treated as a first-class failure mode: some ideas looked promising locally but collapsed across months.
And third, ecology had to stay inside the model logic. Movement, radar, and context only became useful when we kept them tied to migration and mitigation meaning.

## 2:50–3:20 — Proof / why credible
Why do we trust the result?
Because this approach brought us to the AI Cup 2026 finals, and because the path there was built on honest validation rather than one lucky run. We explored broadly, rejected weak shortcuts, and converged on a system we can explain and defend.

## 3:20–4:00 — Deployment / close
In practice, this system is most valuable inside a monitoring and mitigation workflow. Radar monitoring feeds the model, the model produces a ranked and uncertainty-aware output, and operators or ecologists can use that to support better targeted mitigation, more defensible shutdown decisions, and scalable monitoring where manual observation is difficult.

So the strength of our work is not only that it performs well, but that it is rigorous, ecologically grounded, and realistically deployable.
And that is what we hope you remember about our team.

---

# Why this pitch should work
## It gives the room what it wants early
- real problem
- concrete system
- clear team identity

## It gives the jury what they need
- shift-aware rigor
- ecology-aware framing
- deployment logic
- disciplined workflow

## It gives the congress room something memorable
- this team is serious
- this team can build
- this team is not just chasing benchmarks
- this could become something real

---

# Best answer structure for Q&A
For almost every hard question, answer in this order:
1. **direct answer**
2. **evidence**
3. **limitation**
4. **deployment implication**

Example:
> We treat this as decision support, not autonomous control. We validated under seasonal shift because that was the main generalization risk we observed. The limitation is that radar cannot remove all ecological uncertainty. But that is exactly why expert review remains in the loop in the deployment workflow.

---

# Final note for delivery
The pitch should not sound like you are asking to be believed.
It should sound like you know exactly why your work matters.

The strongest tone is:
- clear
- calm
- specific
- non-defensive
- serious

That will make the room feel that your team belongs there.


---

# How to carry this pitch on stage
This pitch works best when it is delivered with the feeling that the team already understands why the work matters.

## The strongest delivery mindset
Do not present it like:
- a school assignment,
- a competition summary,
- or a performance that has to be perfect.

Present it like:
- a team explaining something real that it has genuinely understood,
- a group of builders showing a system they can defend,
- people who know why their work deserves to be in the finals.

## The mental frame that fits this room best
The room does not need perfection.
It needs:
- clarity,
- conviction,
- structure,
- and visible belief in the work.

The strongest internal framing is:

> **We know this problem. We know why our choices make sense. We know what makes our system credible.**

That is enough.

## The most useful speaking posture
- Speak **slightly slower** than feels natural.
- Finish the sentence before trying to sound impressive.
- Land the key nouns clearly: **wind energy**, **biodiversity**, **shift**, **workflow**, **deployment**.
- Treat short pauses as a strength, not a mistake.
- On important lines, sound declarative rather than apologetic.

## What to remember if nerves spike
If delivery starts to feel shaky, reduce the whole talk to five anchors:
1. real problem
2. clear system
3. disciplined workflow
4. honest validation
5. real-world use

If those five land, the pitch still works.

## Best way to sound confident without overacting
Confidence here is not loudness.
It is:
- sounding like you know why each sentence is there,
- not rushing,
- not backing away from your strongest claims,
- and not overclaiming what the system cannot do.

That balance reads as maturity.

---

# Vision layer — what this pitch is really saying
Underneath the technical content, this pitch should leave the room with a bigger idea:

> **This team can take a messy real-world problem, think rigorously, iterate hard, and turn it into something useful.**

That matters because the room is not only judging the artifact.
It is also judging:
- judgment,
- seriousness,
- potential,
- and whether the team looks capable of doing this again on bigger problems.

## The deeper version of the story
This is a story about:
- sensing a difficult real-world phenomenon,
- refusing shallow optimization,
- learning what actually generalizes,
- and building AI that is useful under real constraints.

That is the version of the story people remember afterwards.

## What the room should feel at the end
- this team understands the problem deeply
- this team built something real
- this team knows how to think
- this team could build serious things after this competition too

---

# Lines worth leaning into on stage
These lines have both persuasive and emotional value because they are strong without sounding inflated.

- **We did not just optimize a score; we built a system we could defend.**
- **We treated this as an ecology, operations, and sensing problem at the same time.**
- **The hardest challenge was not only imbalance, but generalization under seasonal shift.**
- **Our workflow filtered attractive ideas into robust ones.**
- **The output is decision support, not blind automation.**
- **What matters most is not just that the model worked, but that the reasoning behind it is credible.**

---

# If the energy in the room drops
Use one of these turns:

- **Why this matters is simple: wind energy and biodiversity share the same airspace.**
- **The key point is that this was never just an accuracy problem.**
- **What made our team strong was not one trick, but judgment under pressure.**
- **The reason we trust the system is not one result, but the way we got there.**

These lines reset attention without sounding theatrical.

---

# Final delivery note
The best version of this pitch is not the most polished-sounding version.
It is the version where the audience feels:

> **they believe what they are saying.**

That is what makes a technical pitch persuasive.
