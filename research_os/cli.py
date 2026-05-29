from __future__ import annotations

import argparse
import json

from .artifacts import inspect_prediction_artifact
from .memory import MEMORY_FILES, MemoryService
from .workspace import DEFAULT_CONFIG, PilotWorkspace


def _workspace_from_args(args: argparse.Namespace) -> PilotWorkspace:
    return PilotWorkspace.from_root(args.root)


def cmd_init(args: argparse.Namespace) -> int:
    workspace = _workspace_from_args(args)
    if workspace.config_path.exists() and not args.force:
        raise SystemExit(f"Config already exists: {workspace.config_path} (use --force to overwrite)")
    workspace.config = dict(DEFAULT_CONFIG)
    workspace.save()
    created = []
    if not args.no_scaffold_memory:
        created = MemoryService(workspace.root).scaffold_missing()
    print(json.dumps({"config_path": str(workspace.config_path), "scaffolded_files": created}, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    workspace = _workspace_from_args(args)
    workspace.ensure_layout()
    payload = {
        "root": str(workspace.root),
        "pilot_dir": str(workspace.pilot_dir),
        "outputs_dir": str(workspace.outputs_dir),
        "reports_dir": str(workspace.reports_dir),
        "quarantine_dir": str(workspace.quarantine_dir),
        "config_path": str(workspace.config_path),
        "config_exists": workspace.config_path.exists(),
        "defaults": {
            "base_oof": workspace.config.get("default_base_oof_predictions"),
            "base_test": workspace.config.get("default_base_test_predictions"),
            "policy": workspace.config.get("recommendation_policy_version"),
        },
        "memory_files": {name: str(workspace.root / filename) for name, filename in MEMORY_FILES.items()},
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    workspace = _workspace_from_args(args)
    issues: list[str] = []
    warnings: list[str] = []

    for _, filename in MEMORY_FILES.items():
        if not (workspace.root / filename).exists():
            issues.append(f"Missing {filename}")

    default_artifacts = {
        "base_oof": workspace.resolve_repo_path(None, default_key="default_base_oof_predictions"),
        "base_test": workspace.resolve_repo_path(None, default_key="default_base_test_predictions"),
    }
    artifact_checks = {}
    for name, path in default_artifacts.items():
        if path is None or not path.exists():
            warnings.append(f"Configured default artifact missing: {name}")
            continue
        inspection = inspect_prediction_artifact(path)
        artifact_checks[name] = inspection.to_dict()
        if not inspection.valid:
            warnings.append(f"Configured default artifact invalid: {name} -> {inspection.message}")
        elif inspection.shape and len(inspection.shape) == 2 and inspection.shape[1] != 9:
            warnings.append(f"{name} has unexpected class dimension {inspection.shape[1]} (expected 9)")

    payload = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "artifact_checks": artifact_checks,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 1


def cmd_memory_inspect(args: argparse.Namespace) -> int:
    workspace = _workspace_from_args(args)
    memory = MemoryService(workspace.root)
    kinds = MEMORY_FILES.keys() if args.kind == "all" else [args.kind]
    payload = {kind: memory.inspect(kind)[: args.preview] for kind in kinds}
    print(json.dumps(payload, indent=2))
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    from .workflows import run_baseline

    workspace = _workspace_from_args(args)
    base_test_path = workspace.resolve_repo_path(args.base_test, default_key="default_base_test_predictions")
    if base_test_path is None:
        raise SystemExit("baseline-run requires --base-test or a configured default_base_test_predictions")
    result = run_baseline(workspace, base_test_path=base_test_path, output_label=args.output_label)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_validate_compare_runs(args: argparse.Namespace) -> int:
    from .workflows import validate_compare_runs

    workspace = _workspace_from_args(args)
    base_oof_path = workspace.resolve_repo_path(args.base_oof, default_key="default_base_oof_predictions")
    base_test_path = workspace.resolve_repo_path(args.base_test, default_key="default_base_test_predictions")
    if base_oof_path is None or base_test_path is None:
        raise SystemExit("validate-compare-runs requires --base-oof/--base-test or configured defaults")
    result = validate_compare_runs(workspace, base_oof_path=base_oof_path, base_test_path=base_test_path)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_report_memory(args: argparse.Namespace) -> int:
    from .workflows import report_and_memory_update

    workspace = _workspace_from_args(args)
    result = report_and_memory_update(
        workspace,
        experiment_id=args.experiment_id,
        name=args.name,
        cv_map=args.cv_map,
        note=args.note,
        tracker_summary=args.tracker_summary,
        research_summary=args.research_summary,
        architecture_decision=args.architecture_decision,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-os")
    parser.add_argument("--root", default=".", help="Repository root")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Initialize the pilot workspace")
    init_parser.add_argument("--force", action="store_true")
    init_parser.add_argument("--no-scaffold-memory", action="store_true")
    init_parser.set_defaults(func=cmd_init)

    status_parser = sub.add_parser("status", help="Show pilot workspace status")
    status_parser.set_defaults(func=cmd_status)

    doctor_parser = sub.add_parser("doctor", help="Run loop readiness checks")
    doctor_parser.set_defaults(func=cmd_doctor)

    memory_parser = sub.add_parser("memory", help="Inspect memory files")
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
    inspect_parser = memory_sub.add_parser("inspect", help="Inspect configured memory files")
    inspect_parser.add_argument("--kind", choices=["all", *MEMORY_FILES.keys()], default="all")
    inspect_parser.add_argument("--preview", type=int, default=400)
    inspect_parser.set_defaults(func=cmd_memory_inspect)

    run_parser = sub.add_parser("run", help="Run canonical pilot workflows")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)
    baseline_parser = run_sub.add_parser("baseline-run", help="Run the v1 baseline workflow")
    baseline_parser.add_argument("--base-test", default=None, help="Path to base test prediction .npy")
    baseline_parser.add_argument("--output-label", "--output-name", dest="output_label", default="pilot_baseline")
    baseline_parser.set_defaults(func=cmd_run_baseline)

    validate_parser = sub.add_parser("validate-compare-runs", help="Validate and compare a baseline flow")
    validate_parser.add_argument("--base-oof", default=None, help="Path to base OOF prediction .npy")
    validate_parser.add_argument("--base-test", default=None, help="Path to base test prediction .npy")
    validate_parser.set_defaults(func=cmd_validate_compare_runs)

    report_parser = sub.add_parser("report-and-memory-update", help="Write a pilot report and update memory")
    report_parser.add_argument("--experiment-id", required=True)
    report_parser.add_argument("--name", required=True)
    report_parser.add_argument("--cv-map", required=True)
    report_parser.add_argument("--note", required=True)
    report_parser.add_argument("--tracker-summary", "--tracker-note", dest="tracker_summary")
    report_parser.add_argument("--research-summary", "--research-note", dest="research_summary")
    report_parser.add_argument("--architecture-decision", "--architecture-note", dest="architecture_decision")
    report_parser.set_defaults(func=cmd_report_memory)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
