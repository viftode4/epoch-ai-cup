# epoch-ai-cup Research OS Pilot

This repository now includes a **CLI/OMX-first applied-ML research OS pilot** for technical
research labs and solo power users.

It is **not**:
- a generic agent platform
- a UI-first product
- a full rewrite of the legacy research repo

## Install

Use a clean Python 3.11+ environment and install the pilot surface in editable mode:

```bash
python -m pip install -e .
research-os --root . status
```

## Workspace model

The repo is split conceptually into:

- **canonical pilot surface** — `research-os` commands and `.pilot/`
- **legacy research surface** — the historical `src/` + `experiments/` workflow
- **quarantine/generated outputs** — `.pilot/outputs`, `.pilot/reports`, `.pilot/quarantine`

If a command writes to the repo root, that is a **legacy escape hatch**, not the pilot default.

## Canonical commands

```bash
research-os --root . init
research-os --root . status
research-os --root . doctor
research-os --root . run baseline-run --base-test test_e175_best.npy --output-label pilot-baseline
research-os --root . validate-compare-runs --base-oof oof_e175_best.npy --base-test test_e175_best.npy
research-os --root . report-and-memory-update --experiment-id E900 --name pilot --cv-map 0.7000 --note "pilot note"
research-os --root . memory inspect --kind all
```

## Workflow contract

All 3 canonical workflows emit the same top-level result shape:
- `workflow`
- `spec`
- `artifacts`
- `outputs`
- `metrics`
- `decision`
- `memory_updates`
- `summary`
- `warnings`
- `generated_at`

## Artifact validation

`baseline-run` and `validate-compare-runs` validate prediction artifacts before loading them.

The pilot surface now:
- detects Git LFS pointer files explicitly
- records artifact metadata such as path, shape, dtype, and validity
- fails with a readable pilot-specific error instead of exposing raw NumPy loader noise

## Recommendation policy

`validate-compare-runs` now normalizes the raw validation output into one of four states:
- `submit`
- `safe-trial`
- `review`
- `reject`

These states are driven by a documented threshold policy (`loop-v1`) using:
- estimated delta
- shared-month safety
- prediction-shift magnitude

## Memory contract

Each memory file has a distinct role:
- `EXPERIMENTS.md` — append-only run ledger
- `TRACKER.md` — mutable status / priority board
- `RESEARCH.md` — curated findings and synthesis
- `FINAL_ARCHITECTURE.md` — durable architecture snapshot
- `CLAUDE.md` — operator guide

External `~/.claude/.../memory/*` files are **out of scope for v1** unless explicitly bridged later.

## Legacy caveats

Legacy scripts may still:
- use repo-local paths
- inject `sys.path`
- write `submission.csv` at repo root
- depend on cached `.npy/.pkl` artifacts

Those behaviors remain available only through the **legacy bridge** and are not the canonical pilot path.
