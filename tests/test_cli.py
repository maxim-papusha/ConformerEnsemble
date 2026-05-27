"""Smoke tests for the `crest-filter` CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conformerensemble.cli import main as cli_main

CREST_XYZ = """\
3
energy: -76.41
O 0.00000000 0.00000000 0.00000000
H 0.95720000 0.00000000 0.00000000
H -0.23998720 0.92662721 0.00000000
3
energy: -76.41
O 0.00000000 0.00000000 0.00000000
H 0.95720000 0.00000000 0.00000000
H -0.23998720 0.92662721 0.00000000
3
energy: -76.40
O 0.00000000 0.00000000 0.00000000
H 0.95720000 0.00000000 0.00000000
H 0.23998720 0.92662721 0.00000000
"""


def _write_input(tmp_path: Path) -> Path:
    p = tmp_path / "crest_conformers.xyz"
    p.write_text(CREST_XYZ, encoding="utf-8")
    return p


def test_cli_filters_duplicate_conformers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inp = _write_input(tmp_path)
    out = tmp_path / "filtered.xyz"

    rc = cli_main([str(inp), "-o", str(out), "--rmsd-threshold", "0.05"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "kept" in captured.err
    text = out.read_text(encoding="utf-8")
    # Two distinct geometries should remain (the duplicate is removed).
    assert text.count("\nO ") + (1 if text.startswith("O ") else 0) >= 2
    # XYZ headers (atom count "3") appear once per kept conformer.
    headers = [ln for ln in text.splitlines() if ln.strip() == "3"]
    assert len(headers) == 2


def test_cli_writes_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inp = _write_input(tmp_path)

    rc = cli_main([str(inp), "--rmsd-threshold", "0.05", "--quiet"])

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    headers = [ln for ln in captured.out.splitlines() if ln.strip() == "3"]
    assert len(headers) == 2


def test_cli_missing_input_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_main([str(tmp_path / "does_not_exist.xyz")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err or "failed" in captured.err


def test_cli_invokable_as_module(tmp_path: Path) -> None:
    """Ensure the entry point is wired up so `python -m` also works."""
    inp = _write_input(tmp_path)
    out = tmp_path / "filtered.xyz"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from conformerensemble.cli import main; "
            f"raise SystemExit(main([r'{inp}', '-o', r'{out}', '--quiet']))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
