"""Container checks must fail clearly without Docker and clean up on HTTP failure."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import docker_smoke


def test_build_timeout_keeps_partial_output_and_does_not_run_container(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_smoke, "prepare_context", lambda bundle: tmp_path / "context")
    monkeypatch.setattr(docker_smoke.shutil, "which", lambda name: "docker")
    commands = []

    def run(args, **kwargs):
        commands.append(args[1])
        if args[1] == "build":
            raise subprocess.TimeoutExpired(
                args, 1200, output=b"download partial output", stderr=b"network stalled"
            )
        kwargs["stdout"].write("linux" if args[1] == "info" else "fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker_smoke.subprocess, "run", run)
    output = tmp_path / "evidence"
    with pytest.raises(TimeoutError, match="build timed out"):
        docker_smoke.check_docker(tmp_path / "bundle.zip", output)
    log = (output / "build.log").read_text()
    assert "download partial output" in log and "network stalled" in log
    assert "timed out" in log
    assert "run" not in commands
    assert json.loads((output / "validation.json").read_text())["status"] == "failed"


def test_missing_docker_is_blocked_and_never_reported_as_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_smoke, "prepare_context", lambda bundle: tmp_path / "context")
    monkeypatch.setattr(docker_smoke.shutil, "which", lambda name: None)
    output = tmp_path / "evidence"
    with pytest.raises(FileNotFoundError, match="Docker CLI"):
        docker_smoke.check_docker(tmp_path / "bundle.zip", output)
    result = json.loads((output / "validation.json").read_text())
    assert result["status"] == "blocked"
    assert result["build_executed"] is False and result["container_started"] is False


def test_http_failure_still_collects_logs_and_removes_container(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_smoke, "prepare_context", lambda bundle: tmp_path / "context")
    monkeypatch.setattr(docker_smoke.shutil, "which", lambda name: "docker")
    commands = []

    def run(args, **kwargs):
        commands.append(args[1:])
        if args[1] == "info":
            stdout = "linux"
        elif args[1:3] == ["image", "inspect"]:
            stdout = json.dumps([{"Id": "fixture-image", "Config": {"User": "10001:10001"}}])
        elif args[1] == "inspect":
            stdout = json.dumps(
                [{"NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "8000"}]}}}]
            )
        else:
            stdout = "fixture"
        kwargs["stdout"].write(stdout)
        return SimpleNamespace(returncode=0)

    def timeout(*args):
        raise TimeoutError("deliberate readiness failure")

    monkeypatch.setattr(docker_smoke.subprocess, "run", run)
    monkeypatch.setattr(docker_smoke, "wait_ready", timeout)
    output = tmp_path / "evidence"
    with pytest.raises(TimeoutError, match="deliberate"):
        docker_smoke.check_docker(tmp_path / "bundle.zip", output)
    result = json.loads((output / "validation.json").read_text())
    assert result["status"] == "failed" and result["container_started"]
    assert any(command[0] == "logs" for command in commands)
    assert commands[-1][:2] == ["rm", "--force"]


def test_build_log_is_readable_before_the_command_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_smoke, "prepare_context", lambda bundle: tmp_path / "context")
    monkeypatch.setattr(docker_smoke.shutil, "which", lambda name: "docker")
    output = tmp_path / "evidence"

    def run(args, **kwargs):
        log = kwargs["stdout"]
        if args[1] == "build":
            assert "--progress=plain" in args
            assert kwargs["stderr"] == subprocess.STDOUT
            log.write("still downloading a layer\n")
            log.flush()
            assert "still downloading" in (output / "build.log").read_text()
            raise RuntimeError("simulated interrupted build")
        log.write("linux" if args[1] == "info" else "fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker_smoke.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="interrupted build"):
        docker_smoke.check_docker(tmp_path / "bundle.zip", output)
    assert "still downloading" in (output / "build.log").read_text()
    assert json.loads((output / "validation.json").read_text())["status"] == "failed"
