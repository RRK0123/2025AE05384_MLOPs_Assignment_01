"""Check the serving-only transfer inventory and per-file checksums."""

import hashlib
import json
import zipfile

import pytest

from scripts import create_deployment_handoff as handoff


def test_handoff_contains_approved_release_and_code_without_machine_state(tmp_path, monkeypatch):
    project = tmp_path / "source"
    project.mkdir()
    required = [
        "RUN_DOCKER_KUBERNETES.bat",
        "COLLECT_DEPLOYMENT_LOGS.bat",
        "README.md",
        "Dockerfile",
        ".dockerignore",
        ".gitattributes",
        ".gitignore",
        "pyproject.toml",
    ]
    for name in required:
        (project / name).write_text(name)
    files = [
        "src/heart_disease/api.py",
        "scripts/run_deployment.ps1",
        "deployment/windows/START_HERE.md",
        "deployment/windows/CAPTURE_BEFORE_LEAVING.md",
        "deployment/kubernetes/deployment.yaml",
        "docs/sample_request.json",
        "requirements-runtime.txt",
        "models/releases/approved/model.joblib",
        "models/bundles/approved.zip",
        "tracking/private.db",
        "data/raw/source.csv",
        ".venv/Scripts/python.exe",
        "reports/old/log.txt",
        "src/heart_disease/__pycache__/api.pyc",
    ]
    for name in files:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    pointer = project / "reports/modeling/latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps({"run_id": "approved-run", "model_directory": "models/releases/approved"})
    )
    monkeypatch.setattr(handoff, "latest_bundle", lambda root: root / "models/bundles/approved.zip")
    monkeypatch.setattr(handoff, "prepare_context", lambda *args, **kwargs: None)
    archive = handoff.create_handoff(tmp_path / "handoff.zip", project=project)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        assert set(required) <= names
        for guide in ["START_HERE.md", "CAPTURE_BEFORE_LEAVING.md"]:
            assert zipped.read(guide) == zipped.read("deployment/windows/" + guide)
        assert {
            "models/releases/approved/model.joblib",
            "models/bundles/approved.zip",
            "reports/modeling/latest.json",
        } <= names
        assert not any(
            name.startswith(("tracking/", "data/", ".venv/", "reports/old/")) for name in names
        )
        assert not any("__pycache__" in name for name in names)
        manifest = json.loads(zipped.read("HANDOFF_FILES.json"))
        assert manifest["model_run_id"] == "approved-run"
        assert set(manifest["files_sha256"]) == names - {"HANDOFF_FILES.json"}
        for name, digest in manifest["files_sha256"].items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        handoff.create_handoff(archive, project=project)
