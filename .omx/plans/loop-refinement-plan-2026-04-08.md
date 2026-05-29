# Plan: Tighten the epoch-ai-cup pilot loop before widening scope

## Summary
Refine only the current pilot loop:
1. `baseline-run`
2. `validate-compare-runs`
3. `report-and-memory-update`

Do **not** broaden the framework yet. The next iteration should make the loop more deterministic, more opinionated, and easier for a research operator or agent to run without reading legacy code.

## Why this is the right next step
- The current CLI surface exists and is usable (`research_os/cli.py:15-168`), but each workflow still feels like a thin wrapper over legacy behavior.
- `baseline-run` currently relies on a raw artifact path and an output name, with no shared workflow spec or artifact validation (`research_os/cli.py:71-93`, `research_os/workflows.py:68-108`).
- `validate-compare-runs` works, but the result contract is just the raw `eval_pp` payload plus wrapper metadata (`research_os/workflows.py:111-137`, `src/validate.py:355-556`).
- Memory writes are functional, but still too free-form and not explicitly governed by per-file policy (`research_os/memory.py:16-36`, `research_os/workflows.py:140-187`).
- The README already pitches the correct wedge — canonical pilot surface vs legacy bridge — so now the workflow should match that clarity (`README.md:20-60`).

## Decision
Keep the current 3-workflow pilot surface and refine it into a stricter operator loop instead of adding new workflows or domains.

### Decision drivers
1. The pilot surface is already good enough to refine, not rebuild.
2. The main remaining risk is workflow ambiguity, not missing infrastructure.
3. A sharper loop is the best predictor of product value for the target use case.

### Alternatives considered
- **Add more workflows now** — rejected because it increases surface area before the core loop is disciplined.
- **Pivot to a new repo now** — rejected because the current pilot surface is proving useful and the main pain is contract sharpness, not location.

## Implementation plan

### 1. Introduce a shared workflow contract
Add a small schema module for all 3 workflows, for example `research_os/contracts.py`.

Define at least:
- `WorkflowSpec`
  - workflow name
  - input artifact paths
  - output root
  - memory targets
  - run label / experiment id
- `WorkflowResult`
  - workflow name
  - resolved inputs
  - produced outputs
  - metrics
  - recommendation
  - memory updates
  - timestamps
  - warnings / caveats

Why:
- Today each command assembles its own ad hoc payload (`research_os/workflows.py:98-107`, `126-136`, `178-187`).
- A shared result shape will make downstream reporting, docs, and future automation much easier.

Acceptance criteria:
- All 3 workflows return the same top-level result structure.
- CLI prints the same envelope shape for all workflows.
- Report files under `.pilot/reports` share the same metadata fields.

### 2. Make `baseline-run` deterministic and artifact-aware
Refine `baseline-run` first because it is the narrowest loop entry.

Changes:
- Add explicit artifact validation before loading predictions.
- Detect and clearly fail on Git LFS pointer files like the current `e50` defaults instead of surfacing raw NumPy/pickle errors.
- Add a small artifact inspection step that records:
  - path used
  - shape
  - dtype
  - whether the file is a real array vs a pointer / invalid artifact
- Move output naming rules into one helper rather than passing loose `output_name` strings around.

Grounding:
- Current load path is a direct `np.load(..., allow_pickle=True)` wrapper (`research_os/workflows.py:22-23`).
- Current defaults in the workspace still point to `e50` artifacts (`research_os/workspace.py:8-14`) even though the real proof path used `e175` artifacts.

Acceptance criteria:
- `baseline-run` returns a structured artifact-inspection section.
- If a user points to an LFS pointer, the command fails with a readable pilot-specific message.
- Workspace defaults can be updated to a valid local pair without touching code.

### 3. Turn `validate-compare-runs` into the real decision engine
This should become the heart of the loop.

Changes:
- Wrap the raw `eval_pp` output in a stricter comparison result contract.
- Separate these concepts explicitly in output:
  - artifact inputs
  - validation metrics
  - safety checks
  - recommendation
  - recommendation reason
- Make recommendation policy explicit and documented:
  - `submit`
  - `safe-trial`
  - `review`
  - `reject`
- Add a short human-readable summary field designed to feed directly into `report-and-memory-update`.

Grounding:
- The current `eval_pp` payload contains the right raw ingredients but no stable product-facing recommendation contract (`src/validate.py:355-556`).
- The current wrapper just forwards that payload into JSON (`research_os/workflows.py:111-137`).

Acceptance criteria:
- The command emits both raw metrics and a normalized decision block.
- The normalized decision block has stable fields and no NumPy-typed keys.
- Recommendation thresholds are documented in README and/or a pilot guide.

### 4. Enforce per-file memory write policy
Keep the memory contract strict instead of generic.

Changes:
- Add explicit writer methods by memory type, not just generic `append_section_note`.
- Enforce policy:
  - `EXPERIMENTS.md` gets structured run rows only
  - `TRACKER.md` gets next-step / status updates only
  - `RESEARCH.md` gets findings / synthesis only
  - `FINAL_ARCHITECTURE.md` only gets durable design decisions
  - `CLAUDE.md` stays read-only in v1
- Make `report-and-memory-update` choose targets from a structured input instead of relying mostly on loose note flags.

Grounding:
- Current memory service is minimal and generic (`research_os/memory.py:16-36`).
- Current report workflow updates 3 files by default with largely free-form note content (`research_os/workflows.py:155-187`).

Acceptance criteria:
- Each memory target has its own explicit update function or policy gate.
- The report workflow records exactly which files were changed and why.
- Architecture updates require explicit opt-in, not optional string presence alone.

### 5. Tighten `init`, `doctor`, and workspace defaults around the loop
The operator loop should start cleanly.

Changes:
- Make `init` optionally scaffold missing memory files for a new pilot root.
- Make `doctor` validate workflow prerequisites, not just path existence.
- Store workflow defaults in `.pilot/config.json`, including:
  - default baseline OOF/test artifacts
  - default memory targets
  - default report output location
  - recommendation policy version

Grounding:
- `init` currently only writes config/layout (`research_os/cli.py:15-22`).
- `doctor` currently checks only file presence (`research_os/cli.py:42-59`).

Acceptance criteria:
- A fresh pilot root can be initialized and pass `doctor` without manual file creation.
- `doctor` can explain why a loop is not runnable, not just which file is missing.

### 6. Add one golden end-to-end proof for the loop
Add a single high-signal regression test/proof path for the operator loop.

Changes:
- Add one end-to-end smoke test or script that proves:
  1. `init`
  2. `status`
  3. `baseline-run`
  4. `validate-compare-runs`
  5. `report-and-memory-update`
- Use known-good local artifacts (for example the `e175` pair) or committed tiny fixtures, not the current LFS-pointer defaults.

Grounding:
- Current tests cover pieces, not the whole loop (`tests/test_cli.py:25-39`, `tests/test_validate.py:11-29`, `tests/test_memory.py:22-33`, `tests/test_submission.py:11-33`).

Acceptance criteria:
- One command or one test path proves the loop end to end.
- The proof confirms no default repo-root output leakage.

## Verification plan
Required proof for the refinement pass:
- `python -m pytest tests`
- `research-os --root . init --force`
- `research-os --root . status`
- `research-os --root . doctor`
- `research-os --root . run baseline-run ...`
- `research-os --root . validate-compare-runs ...`
- `research-os --root <temp-root> report-and-memory-update ...`
- explicit no-root-leak check on `submission.csv`

## Risks and mitigations
- **Risk:** result schema drifts across workflows
  - **Mitigation:** central contract module and golden smoke test
- **Risk:** validation command remains too tied to legacy `eval_pp`
  - **Mitigation:** wrap raw metrics in a normalized decision block instead of exposing raw payload as the product contract
- **Risk:** memory updates stay too informal
  - **Mitigation:** per-file writer policy and explicit target selection
- **Risk:** defaults keep pointing to unusable artifacts
  - **Mitigation:** artifact inspection + `doctor` validation + configurable defaults

## Follow-up after this loop pass
Only after this is stable should we evaluate:
- whether the pilot should move to a clean-shell repo
- whether to broaden beyond the current applied-ML workflow
- whether to expose a higher-level UX or product pitch around it
