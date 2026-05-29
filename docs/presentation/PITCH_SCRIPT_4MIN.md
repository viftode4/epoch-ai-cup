# PITCH SCRIPT — 4 MINUTES

_Last updated: 2026-04-11_

Bird-safe wind energy needs trustworthy AI.
That is the problem we worked on in AI Cup 2026.
And what we want to show today is not just a model, but a rigorous way of building deployable AI for a difficult real-world problem.

Wind energy and biodiversity share the same airspace.
That creates a real tension.
From an ecological perspective, migration can create moments of high bird risk.
From an operational perspective, shutdowns are costly, so mitigation has to be targeted and defensible.
And from a monitoring perspective, radar is scalable, but bird-group classification is difficult because the data are noisy, imbalanced, and seasonally shifting.
So this was never just an accuracy problem.

Our system combines radar trajectories and environmental context into a shift-aware decision-support pipeline.
We start from trajectory behavior and contextual information, build a feature stack with kinematics, signatures, catch22, and validated context, and then combine ranking-aware models with a multiclass probability model.
The output is not blind automation. It is a probabilistic bird-group ranking for expert review.
One of the most important lessons for us was that radar should be treated as an ecological sensor, not as a species oracle.
And the hardest part was not only class imbalance. It was seasonal or month shift, which reflects real migration behavior and changing operating conditions.

That is where our workflow mattered.
We did not keep everything we tried.
We used a disciplined loop of hypothesis, experiment, honest validation, and keep-or-reject decisions.
That workflow acted like a filter: it removed brittle ideas and kept the components that generalized under shift.
That intense experimentation process even earned us the competition award for the most submissions, but for us the more important point is what that process produced: judgment.

The biggest insights were threefold.
First, this was a ranking problem, not just a classification problem, because macro-mAP rewards ordering.
Second, shift was a first-class failure mode: some ideas looked good locally but failed across months.
And third, ecology had to stay inside the model logic. Movement, radar, and context only became useful when we kept them tied to migration and mitigation meaning.

Why do we trust the result?
Because this approach brought us to the AI Cup 2026 top-5 finals, and because the path there was built on honest validation rather than one lucky run.
We explored broadly, rejected weak shortcuts, and converged on a system we can explain and defend.

In practice, this system is most useful inside a monitoring and mitigation workflow.
Radar monitoring feeds the model, the model produces a ranked and uncertainty-aware output, and operators or ecologists can use that to support better targeted mitigation, more defensible shutdown decisions, and scalable monitoring where manual observation is difficult.

So the strength of our work is not only that it performs well, but that it is rigorous, ecologically grounded, and realistically deployable.
And that is what we hope you remember about our team.
