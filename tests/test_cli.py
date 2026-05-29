from __future__ import annotations

from research_os.cli import build_parser


def test_cli_status_and_init(tmp_path, capsys):
    parser = build_parser()

    init_args = parser.parse_args(["--root", str(tmp_path), "init", "--force"])
    assert init_args.func(init_args) == 0

    status_args = parser.parse_args(["--root", str(tmp_path), "status"])
    assert status_args.func(status_args) == 0

    doctor_args = parser.parse_args(["--root", str(tmp_path), "doctor"])
    assert doctor_args.func(doctor_args) == 0

    captured = capsys.readouterr()
    assert ".pilot" in captured.out
    assert (tmp_path / "EXPERIMENTS.md").exists()
    assert "warnings" in captured.out
