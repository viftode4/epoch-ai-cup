# AI Cup Congress Pitch — Speaker Notes (v1)

## Slide 1 — Opening (~30s)
Bird-safe wind energy needs trustworthy AI.
We are presenting more than a model: we are presenting a rigorous way of building deployable AI for a difficult real-world problem.

## Slide 2 — Problem (~30s)
Wind energy and biodiversity share the same airspace.
Radar gives scalable monitoring, but bird-group classification is difficult because the data are noisy, imbalanced, and seasonally shifting.
So the challenge is ecological, operational, and technical at the same time.

## Slide 3 — Team workflow (~40s)
Our team advantage was our workflow.
We built a disciplined research loop: hypothesis, experiment, honest validation, keep or reject.
That let us iterate quickly without fooling ourselves, and it even earned us the award for most submissions.

## Slide 4 — Architecture (~40s)
Our system combines radar trajectories and environmental context.
We engineered features that capture movement, trajectory geometry, and validated external signal, then combined ranking-aware models with multiclass probabilities to produce probabilistic decision support.
The key design pressure throughout was month / seasonal shift.

## Slide 5 — Main insights (~40s)
Three insights changed the project.
First, this is a ranking problem, not just a classification problem.
Second, temporal shift had to be treated as a first-class failure mode.
Third, judgment mattered more than flashy tricks: we rejected ideas that looked good locally but failed to generalize.

## Slide 6 — Credibility (~35s)
What gives us confidence is not just the final score.
It is that we explored broadly, tracked many iterations, reached the finals, won the most submissions award, and converged on a system we can explain and defend.
The workflow itself increased the quality of the final system.

## Slide 7 — Deployment + close (~35s)
In practice, this is most useful as decision support inside a monitoring and mitigation workflow.
Radar monitoring feeds the model, the model ranks bird-group likelihoods with uncertainty, and operators or ecologists can use that to guide more targeted mitigation.
The strength of our work is not only that it performs well, but that it is rigorous, ecologically grounded, and realistically deployable.
