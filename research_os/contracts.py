from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import json_ready


@dataclass(slots=True)
class WorkflowSpec:
    workflow: str
    inputs: dict[str, str]
    output_root: str
    memory_targets: list[str] = field(default_factory=list)
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


@dataclass(slots=True)
class ArtifactInspection:
    path: str
    exists: bool
    kind: str
    valid: bool
    shape: list[int] | None = None
    dtype: str | None = None
    message: str | None = None

    @property
    def artifact_type(self) -> str:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


@dataclass(slots=True)
class DecisionBlock:
    status: str
    reason: str
    policy_version: str
    raw_recommendation: str | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


@dataclass(slots=True)
class MemoryUpdate:
    target: str
    action: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


@dataclass(slots=True)
class WorkflowResult:
    workflow: str
    spec: dict[str, Any]
    artifacts: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    decision: dict[str, Any]
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))
