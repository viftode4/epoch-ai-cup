from __future__ import annotations

from pathlib import Path

from research_os.memory import MemoryService


def test_memory_service_updates_expected_files(tmp_path: Path):
    memory = MemoryService(tmp_path)
    memory.scaffold_missing()
    memory.append_experiment_entry(
        "| E999 | 2026-04-07 | pilot | 0.7000 | -- | -- | -- | -- | -- | -- | -- | -- | -- | test |"
    )
    memory.append_tracker_update("Pilot update", "tracker note")
    memory.append_research_finding("Pilot finding", "research note")
    memory.append_architecture_decision("Pilot decision", "architecture note")

    assert "E999" in (tmp_path / "EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "tracker note" in (tmp_path / "TRACKER.md").read_text(encoding="utf-8")
    assert "research note" in (tmp_path / "RESEARCH.md").read_text(encoding="utf-8")
    assert "architecture note" in (tmp_path / "FINAL_ARCHITECTURE.md").read_text(encoding="utf-8")
