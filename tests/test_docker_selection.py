"""Docker defaults must retain the approved bundle when a later publication fails."""

import json
import shutil
import sys
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from heart_disease import releases
from scripts import docker_smoke, prepare_docker


def write_bundle(path, files, run_id):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps({"model_run_id": run_id}))
        archive.writestr("requirements-runtime.txt", "requirements-runtime.txt")
        for name, content in files.items():
            archive.writestr(f"model/{name}", content)


@pytest.fixture
def approved(tmp_path):
    root = tmp_path / "project"
    candidate = root / "models/runs/approved/serving"
    candidate.mkdir(parents=True)
    metadata = {
        "run_id": "approved",
        "report_directory": "reports/modeling/approved",
        "serving_directory": "models/runs/approved/serving",
        "experiment_tracking": {"parent_run_id": "parent"},
    }
    files = {
        "model.joblib": b"approved-model",
        "metadata.json": json.dumps(metadata).encode(),
        "feature_schema.json": b'{"fields": ["age"]}',
    }
    for name, content in files.items():
        (candidate / name).write_bytes(content)
    (root / "dist").mkdir()
    bundle = root / "dist/approved.zip"
    write_bundle(bundle, files, "approved")
    model_check = {
        "run_id": "approved",
        "model_test_predictions_verified": True,
        "tuning_tables_and_selected_parameters_reconciled": True,
        "fitted_classifier_parameters_match_tuning": True,
        "model_files_sha256": {name: releases.checksum(candidate / name) for name in files},
    }
    tracking_check = {
        "training_session": "approved",
        "parent_run_id": "parent",
        "all_finished": True,
        "tuning_tables_parameters_histories_and_artifacts_reconciled": True,
    }
    release = releases.promote_release(candidate, model_check, tracking_check, bundle, root)
    retained = root / json.loads((release / "bundle.json").read_text())["path"]
    for name in prepare_docker.CONTEXT_FILES:
        (root / name).write_text(name)
    return SimpleNamespace(
        root=root,
        bundle=retained,
        release=release,
        files=files,
        pointer=root / "reports/modeling/latest.json",
    )


def reseal_reference(case, reference):
    """Keep the release chain valid while testing its bundle identity contract."""
    path = case.release / "bundle.json"
    path.write_text(json.dumps(reference))
    manifest_path = case.release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["bundle.json"] = releases.checksum(path)
    manifest_path.write_text(json.dumps(manifest))
    selection = json.loads(case.pointer.read_text())
    selection["release_manifest_sha256"] = releases.checksum(manifest_path)
    case.pointer.write_text(json.dumps(selection))


def test_default_requires_no_last_build_report(approved):
    assert not (approved.root / "reports/packaging/build_validation.json").exists()
    assert prepare_docker.latest_bundle(approved.root) == approved.bundle


@pytest.mark.parametrize("failure", ["logging", "promotion"])
def test_post_build_failure_preserves_default_bundle_and_context(approved, monkeypatch, failure):
    from heart_disease import package_model, train, validate_model, validate_tracking

    case = approved
    original_pointer = case.pointer.read_bytes()
    rejected = case.root / "dist/rejected.zip"
    metadata = {
        "serving_directory": "models/runs/rejected/serving",
        "report_directory": "reports/modeling/rejected",
    }
    statuses = []

    class Client:
        def log_artifact(self, parent, path, group):
            if failure == "logging" and group == "release":
                raise OSError("Injected post-build logging failure")

        def set_terminated(self, parent, status):
            statuses.append(status)

    class Tracker:
        parent_id = "rejected-parent"
        client = Client()

        @contextmanager
        def session_run(self):
            yield self

    def build(**kwargs):
        write_bundle(rejected, {**case.files, "model.joblib": b"unpublished-model"}, "rejected")
        report = case.root / "reports/packaging/build_validation.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"bundle": "dist/rejected.zip"}))
        return rejected

    def reject_promotion(*args, **kwargs):
        raise OSError("Injected post-build promotion failure")

    monkeypatch.setattr(train, "ROOT", case.root)
    monkeypatch.setattr(train, "ExperimentTracker", lambda _: Tracker())
    monkeypatch.setattr(train, "_train_models", lambda *args: metadata)
    monkeypatch.setattr(validate_model, "validate_model", lambda *args: {})
    monkeypatch.setattr(validate_tracking, "validate_tracking", lambda *args: {})
    monkeypatch.setattr(package_model, "build_bundle", build)
    monkeypatch.setattr(train, "promote_release", reject_promotion)
    with pytest.raises(OSError, match="Injected post-build"):
        train.train_models()
    assert statuses == ["FAILED"]
    assert case.pointer.read_bytes() == original_pointer
    assert (
        json.loads((case.root / "reports/packaging/build_validation.json").read_text())["bundle"]
        == "dist/rejected.zip"
    )
    chosen = prepare_docker.latest_bundle(case.root)
    assert chosen == case.bundle
    monkeypatch.setattr(prepare_docker, "verify_bundle", lambda _: {})
    context = prepare_docker.prepare_context(chosen, project=case.root)
    assert (context / "bundle/model/model.joblib").read_bytes() == b"approved-model"


@pytest.mark.parametrize("selection", ["missing", "legacy"])
def test_missing_approved_selection_never_falls_back_to_build_report(approved, selection):
    if selection == "missing":
        approved.pointer.unlink()
    else:
        approved.pointer.write_text(json.dumps({"run_id": "legacy"}))
    report = approved.root / "reports/packaging/build_validation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"bundle": "dist/approved.zip"}))
    with pytest.raises(ValueError, match="No published model release"):
        prepare_docker.latest_bundle(approved.root)


def test_changed_release_reference_is_rejected(approved):
    (approved.release / "bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="Selected release checksum mismatch"):
        prepare_docker.latest_bundle(approved.root)


def test_changed_approved_archive_is_rejected(approved):
    with zipfile.ZipFile(approved.bundle, "a") as archive:
        archive.writestr("unexpected.txt", "changed after approval")
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        prepare_docker.latest_bundle(approved.root)


def test_missing_approved_archive_fails_closed(approved):
    approved.bundle.unlink()
    with pytest.raises(FileNotFoundError, match="Selected release bundle is missing"):
        prepare_docker.latest_bundle(approved.root)


@pytest.mark.parametrize("change", ["run_id", *releases.MODEL_FILES])
def test_archive_must_match_approved_model_even_with_recorded_hash(approved, change):
    files = approved.files.copy()
    if change != "run_id":
        files[change] = b"different approved-model field"
    write_bundle(approved.bundle, files, "different-run" if change == "run_id" else "approved")
    reseal_reference(
        approved,
        {
            "path": approved.bundle.relative_to(approved.root).as_posix(),
            "sha256": releases.checksum(approved.bundle),
        },
    )
    with pytest.raises(ValueError, match="model identity mismatch|differs from approved model"):
        prepare_docker.latest_bundle(approved.root)


@pytest.mark.parametrize("absolute", [False, True])
def test_approved_bundle_reference_cannot_escape_project(approved, absolute):
    outside = approved.root.parent / "outside.zip"
    shutil.copyfile(approved.bundle, outside)
    reseal_reference(
        approved,
        {
            "path": str(outside) if absolute else "../outside.zip",
            "sha256": releases.checksum(outside),
        },
    )
    with pytest.raises(ValueError, match="bundle path must remain inside the project"):
        prepare_docker.latest_bundle(approved.root)


@pytest.mark.parametrize("entry", [prepare_docker, docker_smoke])
@pytest.mark.parametrize("explicit", [False, True])
def test_both_cli_defaults_use_approved_selection_but_allow_explicit_testing(
    approved,
    monkeypatch,
    entry,
    explicit,
):
    chosen = []
    override = approved.root / "dist/testing.zip"
    argv = ["command", "--bundle", str(override)] if explicit else ["command"]
    monkeypatch.setattr(sys, "argv", argv)
    resolver = prepare_docker.latest_bundle

    def select():
        assert not explicit, "Explicit bundle must bypass default selection"
        return resolver(approved.root)

    monkeypatch.setattr(entry, "latest_bundle", select)
    if entry is prepare_docker:

        def context(bundle, **kwargs):
            chosen.append(bundle)
            return approved.root / "context"

        monkeypatch.setattr(entry, "prepare_context", context)
    else:

        def check(bundle, *args):
            chosen.append(bundle)
            return {"status": "passed"}

        monkeypatch.setattr(entry, "check_docker", check)
    entry.main()
    assert chosen == [override if explicit else approved.bundle]
