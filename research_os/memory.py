from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MEMORY_FILES = {
    "experiments": "EXPERIMENTS.md",
    "tracker": "TRACKER.md",
    "research": "RESEARCH.md",
    "architecture": "FINAL_ARCHITECTURE.md",
    "guide": "CLAUDE.md",
}

MEMORY_TEMPLATES = {
    "experiments": "# Experiment Log\n\nTrack every experiment run. Add a row when you run something, even if it fails.\n\n## Results Table\n\n| ID | Date | Name | CV mAP | BoP | Clutter | Cormorants | Ducks | Geese | Gulls | Pigeons | Songbirds | Waders | Notes |\n|----|------|------|--------|-----|---------|------------|-------|-------|-------|---------|-----------|--------|-------|\n",
    "tracker": "# Technique Tracker\n\nStatus legend: `pending` | `discussing` | `testing` | `kept` | `discarded` | `skipped`\n",
    "research": "# Research Notes\n\n## Findings\n",
    "architecture": "# Final Architecture\n\n## Decisions\n",
    "guide": "# CLAUDE.md\n\nOperator guide for the pilot surface.\n",
}


@dataclass(slots=True)
class MemoryService:
    root: Path

    def path_for(self, kind: str) -> Path:
        return self.root / MEMORY_FILES[kind]

    def scaffold_missing(self) -> list[str]:
        created: list[str] = []
        for kind, filename in MEMORY_FILES.items():
            path = self.root / filename
            if not path.exists():
                path.write_text(MEMORY_TEMPLATES[kind], encoding="utf-8")
                created.append(filename)
        return created

    def inspect(self, kind: str) -> str:
        path = self.path_for(kind)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def append_experiment_entry(self, row: str) -> str:
        path = self.path_for("experiments")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{row.rstrip()}\n")
        return str(path)

    def append_tracker_update(self, heading: str, body: str) -> str:
        return self._append_section("tracker", heading, body)

    def append_research_finding(self, heading: str, body: str) -> str:
        return self._append_section("research", heading, body)

    def append_architecture_decision(self, heading: str, body: str) -> str:
        return self._append_section("architecture", heading, body)

    def record_experiment(self, row: str) -> dict:
        path = self.append_experiment_entry(row)
        return {"file": path, "policy": "append-only-run-ledger", "entry": row}

    def record_tracker_update(self, heading: str, body: str) -> dict:
        path = self.append_tracker_update(heading, body)
        return {"file": path, "policy": "mutable-status-board", "heading": heading}

    def record_research_finding(self, heading: str, body: str) -> dict:
        path = self.append_research_finding(heading, body)
        return {"file": path, "policy": "curated-findings-store", "heading": heading}

    def record_architecture_decision(self, heading: str, body: str) -> dict:
        path = self.append_architecture_decision(heading, body)
        return {"file": path, "policy": "durable-architecture-snapshot", "heading": heading}

    def _append_section(self, kind: str, heading: str, body: str) -> str:
        path = self.path_for(kind)
        note = f"\n\n## {heading}\n\n{body.rstrip()}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(note)
        return str(path)
