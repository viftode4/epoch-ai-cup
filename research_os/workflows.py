from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_prediction_matrix
from .contracts import DecisionBlock, MemoryUpdate, WorkflowResult, WorkflowSpec
from .memory import MemoryService
from .utils import json_ready, slugify
from .workspace import PilotWorkspace


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@contextmanager
def legacy_bridge(workspace_root: Path):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    data = importlib.import_module("src.data")
    submission = importlib.import_module("src.submission")
    postprocessing = importlib.import_module("src.postprocessing")
    validate = importlib.import_module("src.validate")

    originals = {
        "data_root": data.ROOT,
        "data_dir": data.DATA_DIR,
        "submission_root": submission.ROOT,
        "postprocessing_root": postprocessing.ROOT,
        "validate_root": validate.ROOT,
    }

    data.ROOT = workspace_root
    data.DATA_DIR = workspace_root / "data"
    submission.ROOT = workspace_root
    postprocessing.ROOT = workspace_root
    validate.ROOT = workspace_root
    validate._cache.clear()

    try:
        yield {
            "data": data,
            "submission": submission,
            "postprocessing": postprocessing,
            "validate": validate,
        }
    finally:
        data.ROOT = originals["data_root"]
        data.DATA_DIR = originals["data_dir"]
        submission.ROOT = originals["submission_root"]
        postprocessing.ROOT = originals["postprocessing_root"]
        validate.ROOT = originals["validate_root"]
        validate._cache.clear()


def _write_result(workspace: PilotWorkspace, report_name: str, result: WorkflowResult) -> str:
    workspace.ensure_layout()
    report_path = workspace.reports_dir / report_name
    report_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return str(report_path)


def _validation_decision(raw: dict, policy_version: str) -> tuple[DecisionBlock, str]:
    estimated_delta = raw["estimated_delta"]
    shared_delta = raw["shared_delta"]
    pct_changed = raw.get("pct_changed")

    thresholds = {
        "submit_delta_gt": 0.005,
        "safe_trial_delta_gte": -0.010,
        "shared_delta_gte": -0.002,
        "review_pct_changed_gt": 30.0,
    }

    if not raw["shared_pass"]:
        status = "reject"
        reason = "shared months degraded beyond the safety threshold"
    elif pct_changed is not None and pct_changed > thresholds["review_pct_changed_gt"]:
        status = "review"
        reason = "prediction shift is larger than the safe automatic-review band"
    elif estimated_delta > thresholds["submit_delta_gt"]:
        status = "submit"
        reason = "estimated improvement exceeds the noise band"
    elif estimated_delta >= thresholds["safe_trial_delta_gte"]:
        status = "safe-trial"
        reason = "result is within the neutral noise band and shared months remain safe"
    else:
        status = "reject"
        reason = "estimated regression is outside the acceptable noise band"

    summary = (
        f"{status.upper()}: est Δ {estimated_delta:+.4f}, "
        f"shared Δ {shared_delta:+.4f}, "
        f"shift>5%={pct_changed if pct_changed is not None else 'n/a'}"
    )
    return (
        DecisionBlock(
            status=status,
            reason=reason,
            policy_version=policy_version,
            raw_recommendation=raw.get("recommendation"),
            thresholds=thresholds,
        ),
        summary,
    )


def run_baseline(
    workspace: PilotWorkspace,
    *,
    base_test_path: Path,
    output_label: str,
) -> WorkflowResult:
    base_test, base_test_inspection = load_prediction_matrix(base_test_path)
    output_slug = slugify(output_label)

    with legacy_bridge(workspace.root) as legacy:
        data = legacy["data"]
        submission = legacy["submission"]
        validate = legacy["validate"]

        test_df = data.load_test()
        train_df = data.load_train()
        sample_submission = data.load_sample_submission()
        y = np.asarray(pd.Categorical(train_df["bird_group"], categories=data.CLASSES).codes, dtype=int)
        test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
        output = validate.default_nb_pp(base_test, test_df, test_months, train_df, y)
        workspace.ensure_layout()
        submission_path = submission.save_submission(
            output,
            f"baseline-{output_slug}",
            cv_map=None,
            submissions_dir=workspace.submissions_dir,
            write_latest=False,
            test_df=test_df,
            sample_submission=sample_submission,
        )

    spec = WorkflowSpec(
        workflow="baseline-run",
        inputs={"base_test": str(base_test_path)},
        output_root=str(workspace.outputs_dir),
        memory_targets=[],
        label=output_slug,
    )
    decision = DecisionBlock(
        status="completed",
        reason="baseline run completed successfully",
        policy_version=workspace.config["recommendation_policy_version"],
    )
    result = WorkflowResult(
        workflow="baseline-run",
        spec=spec.to_dict(),
        artifacts={"base_test": base_test_inspection.to_dict()},
        outputs={"submission_path": str(submission_path)},
        metrics={"rows": int(output.shape[0]), "classes": int(output.shape[1])},
        decision=decision.to_dict(),
        memory_updates=[],
        summary=f"Baseline run completed with {output.shape[0]} rows.",
        warnings=[],
        generated_at=_utc_stamp(),
    )
    result.outputs["report_path"] = _write_result(workspace, f"baseline-{output_slug}.json", result)
    return result


def validate_compare_runs(
    workspace: PilotWorkspace,
    *,
    base_oof_path: Path,
    base_test_path: Path,
) -> WorkflowResult:
    _, oof_inspection = load_prediction_matrix(base_oof_path)
    _, test_inspection = load_prediction_matrix(base_test_path)

    with legacy_bridge(workspace.root) as legacy:
        validate = legacy["validate"]
        raw = validate.eval_pp(
            validate.default_nb_pp,
            verbose=False,
            oof_candidates=[base_oof_path],
            test_candidates=[base_test_path],
        )

    decision, summary = _validation_decision(raw, workspace.config["recommendation_policy_version"])
    spec = WorkflowSpec(
        workflow="validate-compare-runs",
        inputs={"base_oof": str(base_oof_path), "base_test": str(base_test_path)},
        output_root=str(workspace.reports_dir),
        memory_targets=[],
    )
    result = WorkflowResult(
        workflow="validate-compare-runs",
        spec=spec.to_dict(),
        artifacts={
            "base_oof": oof_inspection.to_dict(),
            "base_test": test_inspection.to_dict(),
        },
        outputs={},
        metrics={
            "raw_validation": json_ready(raw),
            "decision_inputs": {
                "estimated_delta": raw["estimated_delta"],
                "calibrated_delta": raw.get("calibrated_delta"),
                "shared_delta": raw["shared_delta"],
                "shared_pass": raw["shared_pass"],
                "pct_changed": raw.get("pct_changed"),
            },
        },
        decision=decision.to_dict(),
        memory_updates=[],
        summary=summary,
        warnings=[],
        generated_at=_utc_stamp(),
    )
    result.outputs["report_path"] = _write_result(workspace, f"validate-compare-runs-{_utc_stamp()}.json", result)
    return result


def report_and_memory_update(
    workspace: PilotWorkspace,
    *,
    experiment_id: str,
    name: str,
    cv_map: str,
    note: str,
    tracker_summary: str | None = None,
    research_summary: str | None = None,
    architecture_decision: str | None = None,
) -> WorkflowResult:
    workspace.ensure_layout()
    memory = MemoryService(workspace.root)
    today = datetime.now(timezone.utc).date().isoformat()

    updates: list[MemoryUpdate] = []
    experiment_row = (
        f"| {experiment_id} | {today} | {name} | {cv_map} | -- | -- | -- | -- | -- | -- | -- | -- | -- | {note} |"
    )
    experiments_path = memory.append_experiment_entry(experiment_row)
    updates.append(MemoryUpdate(target="EXPERIMENTS.md", action="append-row", summary=f"logged run {experiment_id}"))

    tracker_body = tracker_summary or note
    tracker_path = memory.append_tracker_update(f"Pilot update {experiment_id}", tracker_body)
    updates.append(MemoryUpdate(target="TRACKER.md", action="append-status", summary=tracker_body))

    research_body = research_summary or note
    research_path = memory.append_research_finding(f"Pilot finding {experiment_id}", research_body)
    updates.append(MemoryUpdate(target="RESEARCH.md", action="append-finding", summary=research_body))

    architecture_path = None
    if architecture_decision:
        architecture_path = memory.append_architecture_decision(
            f"Pilot architecture note {experiment_id}",
            architecture_decision,
        )
        updates.append(MemoryUpdate(target="FINAL_ARCHITECTURE.md", action="append-decision", summary=architecture_decision))

    report_path = workspace.reports_dir / f"{experiment_id}-memory-update.md"
    report_body = "\n".join(
        [
            f"# Pilot report: {experiment_id}",
            "",
            f"- Name: {name}",
            f"- CV mAP: {cv_map}",
            f"- Note: {note}",
            f"- Tracker summary: {tracker_body}",
            f"- Research summary: {research_body}",
            f"- Architecture decision: {architecture_decision or '(none)'}",
        ]
    )
    report_path.write_text(report_body + "\n", encoding="utf-8")

    spec = WorkflowSpec(
        workflow="report-and-memory-update",
        inputs={"experiment_id": experiment_id, "name": name, "cv_map": cv_map},
        output_root=str(workspace.reports_dir),
        memory_targets=[update.target for update in updates],
        label=experiment_id,
    )
    decision = DecisionBlock(
        status="completed",
        reason="report and memory updates were written successfully",
        policy_version=workspace.config["recommendation_policy_version"],
    )
    return WorkflowResult(
        workflow="report-and-memory-update",
        spec=spec.to_dict(),
        artifacts={},
        outputs={
            "report_path": str(report_path),
            "experiments_path": experiments_path,
            "tracker_path": tracker_path,
            "research_path": research_path,
            **({"architecture_path": architecture_path} if architecture_path else {}),
        },
        metrics={"cv_map": cv_map},
        decision=decision.to_dict(),
        memory_updates=[update.to_dict() for update in updates],
        summary=f"Recorded report and memory updates for {experiment_id}.",
        warnings=[],
        generated_at=_utc_stamp(),
    )
