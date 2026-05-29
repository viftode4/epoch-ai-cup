# 4-Minute Pitch — Strong Draft

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
