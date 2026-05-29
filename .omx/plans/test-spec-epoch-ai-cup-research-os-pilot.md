# Test Spec: epoch-ai-cup CLI/OMX-first Applied-ML Research OS Pilot

- Scope: validates the approved private-pilot extraction plan for `G:\Projects\epoch-ai-cup`
- Source of truth: `prd-epoch-ai-cup-research-os-pilot.md`
- Grounding: no repo `AGENTS.md`; no package manifest today; legacy workflows depend on repo-local paths, caches, and root writes.

## Exit criteria
1. Editable/package install works from a clean pilot surface.
2. Anti-corruption layer mediates data, cache/materialized artifacts, output writes, and legacy experiment invocation before CLI workflows are considered canonical.
3. Exactly 3 canonical workflows exist and run through CLI/OMX first:
   - `baseline-run`
   - `compare-runs`
   - `report-and-memory-update`
4. Research-memory contract covers `EXPERIMENTS.md`, `TRACKER.md`, `RESEARCH.md`, `FINAL_ARCHITECTURE.md`, and `CLAUDE.md` through a stable service/interface.
5. Canonical commands avoid repo-root side effects by default, especially root `submission.csv` writes.
6. Pivot gate decision is explicitly recorded.

## Verification matrix
- **Hygiene/quarantine:** clean pilot surface exists; quarantine/ignore policy covers caches, submissions, reports, `.omx/`, `.serena/`, and other generated artifacts.
- **Boundary repair:** `src/data.py`-style repo-local data binding and `src/submission.py` root-write behavior are mediated behind adapters/interfaces.
- **Legacy containment:** representative legacy experiment pattern (`sys.path.insert(...)`, repo-local cached artifact reads) runs only through documented bridge behavior, not direct canonical CLI assumptions.
- **Packaging:** editable install succeeds; CLI help/status resolve.
- **Workflow verification:** all 3 canonical workflows produce expected outputs/evidence.
- **Memory verification:** workflow-driven memory reads/writes land in the intended research-memory files.
- **Pivot gate:** decision logged with evidence on whether in-repo extraction remains viable.

## Required command-level evidence
Execution must capture and report the exact commands used, with pass/fail outputs, for at least:
1. package install smoke
2. CLI help/status smoke
3. `baseline-run` command
4. `compare-runs` command
5. `report-and-memory-update` command
6. hygiene check proving no default repo-root output leakage

## Suggested evidence expectations
- install command exits 0
- CLI `--help` and `status` exit 0
- baseline-run writes only to approved pilot/quarantine output locations
- compare-runs emits a structured comparison result/artifact
- report-and-memory-update updates the intended memory files and emits a report artifact
- explicit check shows repo-root `submission.csv` is not the default canonical output target

## Team verification ownership
- `test-engineer`: smoke/integration harness + command-level evidence capture
- `architect`: anti-corruption/boundary/pivot review
- `verifier`: final release-readiness check
- `writer`: docs + memory-contract consistency pass

## Failure gates
- No installable package/CLI entrypoint
- Anti-corruption layer missing or bypassed
- Canonical workflows still require direct script-level `sys.path`/repo-root assumptions
- Memory files are not accessible via the new contract
- Canonical commands leak outputs to repo root by default
- Pivot gate not evaluated when boundary repair expands materially
