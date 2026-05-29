from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CONFIG = {
    "default_base_test_predictions": "test_e175_best.npy",
    "default_base_oof_predictions": "oof_e175_best.npy",
    "output_subdir": ".pilot/outputs",
    "reports_subdir": ".pilot/reports",
    "quarantine_subdir": ".pilot/quarantine",
    "recommendation_policy_version": "loop-v1",
    "default_memory_targets": ["experiments", "tracker", "research"],
}


@dataclass(slots=True)
class PilotWorkspace:
    root: Path
    config: dict = field(default_factory=dict)

    @classmethod
    def from_root(cls, root: str | Path = ".") -> "PilotWorkspace":
        workspace_root = Path(root).resolve()
        pilot_dir = workspace_root / ".pilot"
        config_path = pilot_dir / "config.json"
        config = dict(DEFAULT_CONFIG)
        if config_path.exists():
            config.update(json.loads(config_path.read_text(encoding="utf-8")))
        return cls(root=workspace_root, config=config)

    @property
    def pilot_dir(self) -> Path:
        return self.root / ".pilot"

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.config["output_subdir"]

    @property
    def submissions_dir(self) -> Path:
        return self.outputs_dir / "submissions"

    @property
    def reports_dir(self) -> Path:
        return self.root / self.config["reports_subdir"]

    @property
    def quarantine_dir(self) -> Path:
        return self.root / self.config["quarantine_subdir"]

    @property
    def config_path(self) -> Path:
        return self.pilot_dir / "config.json"

    def ensure_layout(self) -> None:
        for path in (self.pilot_dir, self.outputs_dir, self.submissions_dir, self.reports_dir, self.quarantine_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save(self) -> Path:
        self.ensure_layout()
        self.config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.config_path

    def resolve_repo_path(self, value: str | None, *, default_key: str | None = None) -> Path | None:
        if value:
            return (self.root / value).resolve() if not Path(value).is_absolute() else Path(value)
        if default_key:
            default_value = self.config.get(default_key)
            if default_value:
                return (self.root / default_value).resolve()
        return None
