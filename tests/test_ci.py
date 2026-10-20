"""Exercise the CI failure gate and independence from historical outputs."""

import json
import os
import subprocess
import sys

import pytest

from scripts import ci


def test_workspace_contains_no_historical_models_processed_data_or_tracking(tmp_path):
    project = tmp_path / "checkout"
    for directory in ["src", "docs", "data/raw", "data/processed", "models", "tracking", "reports"]:
        (project / directory).mkdir(parents=True)
        (project / directory / "sentinel").write_text(directory)
    for name in ["pyproject.toml", "README.md", "requirements.txt"]:
        (project / name).write_text(name)
    output = tmp_path / "evidence"
    workspace = ci.prepare_workspace(project, output)
    assert (workspace / "data/raw/sentinel").is_file()
    for name in ["data/processed", "models", "tracking", "reports"]:
        assert not (workspace / name).exists()
    with pytest.raises(FileExistsError):
        ci.prepare_workspace(project, output)


def test_failed_check_retains_output_and_blocks_training(tmp_path, monkeypatch):
    def fail_check(name, command, **kwargs):
        # Exercise real subprocess handling, but make the first check fail cheaply.
        assert name == "dependencies"
        ci.run_step(
            name,
            [sys.executable, "-c", "print('deliberate failure'); raise SystemExit(7)"],
            cwd=ci.PROJECT,
            env=os.environ.copy(),
            output=kwargs["output"],
            results=kwargs["results"],
        )

    original = ci.run_step

    def first_check(name, command, **kwargs):
        monkeypatch.setattr(ci, "run_step", original)
        fail_check(name, command, **kwargs)

    monkeypatch.setattr(ci, "run_step", first_check)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    output = tmp_path / "evidence"
    with pytest.raises(subprocess.CalledProcessError) as error:
        ci.run_pipeline(output)
    assert error.value.returncode == 7
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert [step["name"] for step in summary["steps"]] == ["dependencies"]
    assert "deliberate failure" in (output / "dependencies.log").read_text()
    assert not (output / "workspace/models").exists()
