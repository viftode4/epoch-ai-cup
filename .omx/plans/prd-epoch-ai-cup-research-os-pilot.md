# PRD: epoch-ai-cup → CLI/OMX-first Applied-ML Research OS Pilot

- Status: revised draft for private-pilot execution
- Horizon: 4-6 weeks
- Audience: technical research labs and solo power users
- Positioning: applied-ML research OS, not a generic agent framework
- Repo grounding (inspected 2026-04-07, epoch-ai-cup only): no repo `AGENTS.md`; shared library code under `src/`; large experiment sprawl under `experiments/`; strong research-memory docs in `EXPERIMENTS.md`, `TRACKER.md`, `RESEARCH.md`, `FINAL_ARCHITECTURE.md`, `CLAUDE.md`; no `pyproject.toml`/`setup.py`/`setup.cfg`; dirty git state with many modified/untracked artifacts; `src/data.py` reads repo-local `data/`; `src/submission.py` writes repo-root `submission.csv`; representative script `experiments/e175_validated_architecture.py` uses `sys.path.insert(...)` and repo-local cached artifacts under `data/`.

## RALPLAN-DR (short mode)

### Principles
1. Build the framework from proven epoch-ai-cup research patterns, not from generic agent abstractions.
2. Insert an anti-corruption layer before CLI canonicalization so legacy repo-local assumptions do not leak into pilot surfaces.
3. Promote research memory to a first-class product contract, not side documentation.
4. Enforce hygiene/quarantine so pilot users operate on a clean surface even if the legacy repo remains messy.

### Decision Drivers
1. **Pilot usability in 4-6 weeks:** must create a usable private pilot without rewriting the full experiment corpus.
2. **Boundary repair need:** current code assumes repo-root paths, root-side effects, and cached local artifacts.
3. **Research leverage:** the strongest reusable asset is the combination of shared `src/` logic plus mature research-memory docs.

### Viable Options

#### Option A — Thin CLI over current repo conventions
- Pros: fastest initial wrapper.
- Cons: preserves root writes, local-path coupling, dirty-workspace fragility, and script-level import hacks.
- Rejection rationale: unsafe for pilot users because it canonizes legacy leakage instead of containing it.

#### Option B — In-repo extraction with anti-corruption layer first (chosen)
- Pros: preserves proven epoch-ai-cup logic and docs while isolating repo-local assumptions behind adapters; fits 4-6 week pilot.
- Cons: requires disciplined boundary definition and temporary dual-path support.
- Why chosen: best speed/safety tradeoff for a private pilot.

#### Option C — Immediate clean-shell/new-repo pivot
- Pros: strongest cleanliness and packaging boundary.
- Cons: duplicates too much too early and risks losing validated workflow knowledge.
- Rejection rationale: not the default path; keep as an explicit pivot if boundary repair balloons.

## ADR
- **Decision:** Extract an installable CLI/OMX-first applied-ML research OS from `epoch-ai-cup` in place, but require an anti-corruption layer that mediates repo-local data access, cache access, path mutation, and output writes before any workflow is treated as canonical CLI behavior.
- **Drivers:** short pilot window; path/output coupling in `src/` and `experiments/`; high value of existing research docs.
- **Alternatives considered:** thin wrapper over current repo conventions; immediate clean-shell/new-repo pivot.
- **Why chosen:** it keeps validated research logic while creating a safer pilot surface and a reversible decision point.
- **Consequences:** temporary dual world (legacy scripts + stabilized pilot surface), mandatory hygiene policy, and a formal pivot gate.
- **Follow-ups:** anti-corruption layer, quarantine/ignore policy, memory contract, command-level verification, pivot decision checkpoint.

## Concrete research-memory contract
The pilot surface must treat these files as first-class research memory:
- `EXPERIMENTS.md` — experiment catalog and outcome ledger
- `TRACKER.md` — active work/progress ledger
- `RESEARCH.md` — research synthesis / findings memory
- `FINAL_ARCHITECTURE.md` — validated architecture rationale
- `CLAUDE.md` — operator/research workflow guidance artifact

Required contract:
1. Read access through one memory service/interface, not ad-hoc file reads spread across commands.
2. Stable memory verbs at the CLI/OMX layer: inspect, append/update, reference in run/report flows.
3. Each canonical workflow must record evidence into the appropriate memory target(s).
4. File ownership rules must distinguish canonical memory from generated artifacts/reports.

## Final execution plan

### Phase 0 — Hygiene, quarantine, and pilot surface creation (2-3 days)
- Create a clean pilot branch/worktree or equivalent clean pilot surface.
- Define quarantine/ignore policy for generated caches, submissions, reports, `.omx/`, `.serena/`, and other non-product artifacts.
- Separate three zones: `legacy-research surface`, `pilot framework surface`, `quarantine/generated outputs`.
- Capture a boundary inventory of repo-local assumptions: `src/data.py` data-root binding, `src/submission.py` root write behavior, `sys.path.insert(...)` imports, cached artifact dependencies in representative experiments.
- **Acceptance:** clean pilot surface exists; quarantine/ignore policy is written; no canonical pilot command writes to repo root by default; legacy/generated artifacts are explicitly non-canonical.

### Phase 1 — Anti-corruption layer before CLI canonicalization (4-5 days)
- Build adapters for data access, cache/materialized artifact lookup, output writes, and legacy experiment invocation.
- Redirect root-coupled behaviors behind explicit interfaces: workspace paths, output directories, memory service, and legacy-run bridge.
- Keep legacy scripts functional, but forbid direct elevation of repo-local conventions into the new CLI.
- **Acceptance:** pilot workflows can consume data/cache/output services without direct repo-root assumptions; root `submission.csv` write becomes adapter-mediated behavior, not default canonical behavior.

### Phase 2 — Package + framework boundary (3-4 days)
- Add installable package metadata and CLI entrypoint.
- Define stable modules for: `core/services`, `workflows`, `memory`, `adapters`, and `legacy_bridge`.
- Document what remains legacy under `experiments/` versus what graduates into supported workflow modules.
- **Acceptance:** editable install works; CLI resolves; framework boundary doc maps `src/` + `experiments/` usage into canonical vs legacy surfaces.

### Phase 3 — Canonicalize exactly 3 pilot workflows (1-1.5 weeks)
Canonical workflows:
1. **baseline-run** — execute a selected validated experiment path through the new surface using the anti-corruption layer.
2. **compare-runs** — compare outputs/metrics/artifacts from two research runs and produce a structured comparison result.
3. **report-and-memory-update** — generate a report summary and append/update the correct research-memory artifacts.

For each workflow:
- define inputs, outputs, memory writes, and failure modes
- route through CLI/OMX first; legacy scripts only behind the bridge
- collect explicit command-level evidence
- **Acceptance:** all 3 workflows run end-to-end from the CLI/OMX surface with consistent outputs and memory capture.

### Phase 4 — Pilot hardening + pivot gate (1-1.5 weeks)
- Add smoke/integration tests for install, CLI entrypoints, workflow execution, memory updates, and output isolation.
- Produce operator docs for labs and solo users: workspace expectations, commands, recovery, and legacy escape hatches.
- Run a pivot gate review:
  - If anti-corruption + boundary repair remains contained, continue in-repo.
  - If boundary repair balloons (for example: too many hidden root-side effects, too many workflow-critical sys.path/cache assumptions, or repeated hygiene regressions), pivot to a clean pilot shell/repo that vendors only the stabilized modules and memory contract.
- **Acceptance:** pivot decision recorded explicitly with rationale; either in-repo pilot surface is stable enough, or clean-shell pivot plan is approved.

### Phase 5 — Private pilot readiness (3-4 days)
- Run pilot scenarios for lab-user and solo-power-user paths.
- Lock command set, docs, known gaps, and issue intake loop.
- **Acceptance:** private pilot checklist passes with command-level evidence and an explicit out-of-scope list.

## Command-level verification / evidence targets
Execution should refine exact commands, but the plan requires evidence at this level:
- **Install/package smoke:** editable install command succeeds; CLI `--help` and `status` succeed.
- **baseline-run evidence:** one canonical baseline command completes and writes outputs only to approved pilot/quarantine locations.
- **compare-runs evidence:** one comparison command produces a structured result/artifact from two prior runs.
- **report-and-memory-update evidence:** one report command updates the intended research-memory files and emits a report artifact.
- **hygiene evidence:** no canonical command writes repo-root `submission.csv` or other root artifacts by default; any legacy write path is explicit, adapter-mediated, and documented.

## Available agent types roster
- `planner` — PRD/test-spec upkeep, sequencing, pivot-gate maintenance
- `architect` — anti-corruption layer, framework boundary, pivot review
- `executor` — package/CLI/adapters/workflow implementation
- `debugger` — root-side effects, path coupling, migration failures
- `test-engineer` — command-level verification, smoke/integration harness
- `verifier` — release-readiness and evidence audit
- `writer` — operator docs, memory contract docs, migration notes
- `explore` — narrow repo mapping/lookups
- `critic` — challenge scope creep and invalid canonicalization

## Staffing guidance

### For `$ralph`
- Single-owner completion loop with three support lanes:
  - `executor` (high) — primary extraction/implementation owner
  - `test-engineer` or `verifier` (medium/high) — command-level evidence owner
  - `architect` (standard minimum) — anti-corruption/boundary sign-off
- Best launch hint: `$ralph --prd "Complete the approved epoch-ai-cup CLI/OMX-first applied-ML research OS pilot with anti-corruption layer and pivot gate"`

### For `$team`
- Recommended staffing: 4 workers
  1. `executor` (high) — anti-corruption layer + package/CLI lane
  2. `executor` (high) — canonical workflow extraction + legacy bridge lane
  3. `test-engineer` (medium) — command/evidence + hygiene lane
  4. `writer` or `architect` (medium/high) — memory contract + docs + pivot review lane
- Add `debugger` temporarily if repo-local side effects block stabilization.
- Best launch hint: `$team 4:executor "Execute approved epoch-ai-cup pilot extraction plan with anti-corruption layer, hygiene gate, workflow extraction, and verification lanes"`

## Team verification path
1. `planner` confirms PRD + test spec remain source of truth.
2. Delivery lanes implement anti-corruption and canonical workflow slices only.
3. `test-engineer` runs command-level install/workflow/hygiene evidence.
4. `architect` validates boundary integrity, anti-corruption containment, and pivot-gate recommendation.
5. `writer` checks docs/memory contract against shipped commands.
6. `verifier` performs final pilot-readiness claim check.
7. Use `$ralph` only for final polish/fix loops after team evidence is mostly green.

## Launch hints
- Planning refresh: `$ralplan "epoch-ai-cup pilot extraction plan review"`
- Coordinated execution: `$team 4:executor "Execute approved epoch-ai-cup CLI/OMX-first pilot extraction plan"`
- Persistent owner loop: `$ralph --prd "Complete epoch-ai-cup private pilot extraction per approved PRD/test spec"`

## Out of scope for the 4-6 week pilot
- Generic multi-domain agent framework
- Full rewrite of all experiment scripts
- Whole-repo cleanup/history rewrite
- Broad public-release packaging/distribution work
