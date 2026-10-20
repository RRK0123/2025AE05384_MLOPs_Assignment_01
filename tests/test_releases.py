"""Verify failed validation/publication preserves the last successful selection."""

import json
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from heart_disease import releases


def test_directory_publication_is_complete_and_preserves_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model").write_bytes(b"complete")
    destination = tmp_path / "release"
    releases.publish_directory(source, destination)
    assert (destination / "model").read_bytes() == b"complete"
    (source / "model").write_bytes(b"changed")
    with pytest.raises(OSError):
        releases.publish_directory(source, destination)
    assert (destination / "model").read_bytes() == b"complete"
    assert not list(tmp_path.glob(".publishing-*"))


def test_directory_copy_failure_leaves_no_partial_release(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "release"

    def broken(source, pending):
        pending.mkdir()
        (pending / "partial").write_bytes(b"partial")
        raise OSError("deliberate copy failure")

    monkeypatch.setattr(releases.shutil, "copytree", broken)
    with pytest.raises(OSError, match="deliberate"):
        releases.publish_directory(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".publishing-*"))


@pytest.fixture
def release_case(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    metadata = {
        "run_id": "candidate-run",
        "report_directory": "reports/modeling/candidate-run",
        "serving_directory": "candidate",
        "experiment_tracking": {"parent_run_id": "parent"},
    }
    (candidate / "metadata.json").write_text(json.dumps(metadata))
    (candidate / "feature_schema.json").write_text("{}")
    (candidate / "model.joblib").write_bytes(b"new-model")
    legacy = tmp_path / "models/selected"
    legacy.mkdir(parents=True)
    (legacy / "model.joblib").write_bytes(b"last-successful-model")
    pointer = tmp_path / "reports/modeling/latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"run_id": "old-run", "report": "old.md"}))
    model_check = {
        "run_id": metadata["run_id"],
        "model_test_predictions_verified": True,
        "tuning_tables_and_selected_parameters_reconciled": True,
        "fitted_classifier_parameters_match_tuning": True,
        "model_files_sha256": {
            name: releases.checksum(candidate / name) for name in releases.MODEL_FILES
        },
    }
    tracking = {
        "training_session": metadata["run_id"],
        "parent_run_id": "parent",
        "all_finished": True,
        "tuning_tables_parameters_histories_and_artifacts_reconciled": True,
        "run_count": 102,
    }
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps({"model_run_id": metadata["run_id"]}))
        for name in releases.MODEL_FILES:
            archive.writestr(f"model/{name}", (candidate / name).read_bytes())
    return SimpleNamespace(
        root=tmp_path,
        candidate=candidate,
        metadata=metadata,
        pointer=pointer,
        legacy=legacy,
        model=model_check,
        tracking=tracking,
        bundle=bundle,
    )


def promote(case):
    return releases.promote_release(
        case.candidate, case.model, case.tracking, case.bundle, case.root
    )


def test_success_switches_one_pointer_to_complete_release(release_case, monkeypatch):
    case = release_case
    old = case.pointer.read_bytes()
    original = releases.os.replace

    def inspect_switch(pending, pointer):
        assert case.pointer.read_bytes() == old
        selection = json.loads(pending.read_text())
        directory = case.root / selection["model_directory"]
        assert all((directory / name).is_file() for name in releases.MODEL_FILES)
        assert (directory / "model_validation.json").is_file()
        assert (directory / "tracking_validation.json").is_file()
        original(pending, pointer)

    monkeypatch.setattr(releases.os, "replace", inspect_switch)
    selected = promote(case)
    assert releases.resolve_model_directory(root=case.root) == selected
    assert (selected / "model.joblib").read_bytes() == b"new-model"
    assert (case.legacy / "model.joblib").read_bytes() == b"last-successful-model"


def test_release_retains_exact_bundle_after_build_output_is_removed(release_case):
    case = release_case
    expected = case.bundle.read_bytes()
    selected = promote(case)
    reference = json.loads((selected / "bundle.json").read_text())
    retained = case.root / reference["path"]
    assert retained.parent == case.root / "models/bundles"
    assert retained.name == reference["sha256"] + ".zip"
    assert retained.read_bytes() == expected
    case.bundle.unlink()
    assert retained.read_bytes() == expected
    assert releases.resolve_model_directory(root=case.root) == selected


@pytest.mark.parametrize("name", ["metadata.json", "feature_schema.json"])
def test_bundle_must_match_all_validated_model_files(release_case, name):
    case = release_case
    old = case.pointer.read_bytes()
    with zipfile.ZipFile(case.bundle, "w") as archive:
        archive.writestr(
            "bundle_manifest.json", json.dumps({"model_run_id": case.metadata["run_id"]})
        )
        for item in releases.MODEL_FILES:
            content = b"mismatched" if item == name else (case.candidate / item).read_bytes()
            archive.writestr(f"model/{item}", content)
    with pytest.raises(ValueError, match="validated candidate"):
        promote(case)
    assert case.pointer.read_bytes() == old


def test_bundle_retention_failure_preserves_selection(release_case, monkeypatch):
    case = release_case
    old = case.pointer.read_bytes()

    def fail(*args):
        raise OSError("Injected retention failure")

    monkeypatch.setattr(releases, "retain_bundle", fail)
    with pytest.raises(OSError, match="retention failure"):
        promote(case)
    assert case.pointer.read_bytes() == old


@pytest.mark.parametrize(
    "key",
    [
        "tuning_tables_and_selected_parameters_reconciled",
        "fitted_classifier_parameters_match_tuning",
        "tuning_tables_parameters_histories_and_artifacts_reconciled",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_publication_requires_completed_tuning_checks(release_case, key, missing):
    case = release_case
    proof = case.model if key in case.model else case.tracking
    if missing:
        del proof[key]
    else:
        proof[key] = False
    before = case.pointer.read_bytes()
    with pytest.raises(ValueError):
        promote(case)
    assert case.pointer.read_bytes() == before


def test_existing_retained_bundle_is_never_overwritten(release_case):
    case = release_case
    selected = promote(case)
    reference = json.loads((selected / "bundle.json").read_text())
    retained = case.root / reference["path"]
    retained.write_bytes(b"changed retained artifact")
    pointer = case.pointer.read_bytes()
    with pytest.raises(ValueError, match="Retained bundle differs"):
        promote(case)
    assert retained.read_bytes() == b"changed retained artifact"
    assert case.pointer.read_bytes() == pointer


@pytest.mark.parametrize(
    "failure", ["model_validation", "tracking_validation", "candidate_changed", "wrong_bundle"]
)
def test_rejected_candidate_preserves_selection(release_case, failure):
    case = release_case
    old = case.pointer.read_bytes()
    if failure == "model_validation":
        case.model["model_test_predictions_verified"] = False
    elif failure == "tracking_validation":
        case.tracking["all_finished"] = False
    elif failure == "candidate_changed":
        (case.candidate / "model.joblib").write_bytes(b"changed after validation")
    else:
        with zipfile.ZipFile(case.bundle, "w") as archive:
            archive.writestr("bundle_manifest.json", json.dumps({"model_run_id": "other-run"}))
            archive.writestr("model/model.joblib", b"another model")
    with pytest.raises(ValueError):
        promote(case)
    assert case.pointer.read_bytes() == old
    assert releases.resolve_model_directory(root=case.root) == case.legacy


def test_atomic_pointer_failure_preserves_old_release(release_case, monkeypatch):
    case = release_case
    old = case.pointer.read_bytes()

    def fail(*args):
        raise OSError("Injected publication failure")

    monkeypatch.setattr(releases.os, "replace", fail)
    with pytest.raises(OSError, match="Injected publication failure"):
        promote(case)
    assert case.pointer.read_bytes() == old
    assert releases.resolve_model_directory(root=case.root) == case.legacy


def test_modified_published_release_is_rejected(release_case):
    case = release_case
    selected = promote(case)
    (selected / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Selected release checksum mismatch"):
        releases.resolve_model_directory(root=case.root)


@pytest.mark.parametrize("failure", ["model", "tracking", "bundle", "logging", None])
def test_training_orchestration_validates_before_promotion(release_case, monkeypatch, failure):
    from heart_disease import package_model, train, validate_model, validate_tracking

    case = release_case
    old = case.pointer.read_bytes()
    events, statuses = [], []

    def checkpoint(name, result):
        events.append(name)
        if failure == name:
            raise ValueError(f"Injected {name} failure")
        return result

    class Client:
        def log_artifact(self, *args):
            checkpoint("logging", None)

        def set_terminated(self, run_id, status):
            statuses.append(status)

    class Tracker:
        parent_id = "parent"
        client = Client()

        @contextmanager
        def session_run(self):
            yield self

    monkeypatch.setattr(train, "ROOT", case.root)
    monkeypatch.setattr(train, "ExperimentTracker", lambda *args: Tracker())
    monkeypatch.setattr(train, "_train_models", lambda *args: case.metadata)
    monkeypatch.setattr(
        validate_model, "validate_model", lambda *args: checkpoint("model", case.model)
    )
    monkeypatch.setattr(
        validate_tracking, "validate_tracking", lambda *args: checkpoint("tracking", case.tracking)
    )
    monkeypatch.setattr(
        package_model, "build_bundle", lambda **kwargs: checkpoint("bundle", case.bundle)
    )
    if failure:
        with pytest.raises(ValueError, match=f"Injected {failure} failure"):
            train.train_models()
        assert case.pointer.read_bytes() == old
        assert statuses == ["FAILED"]
    else:
        train.train_models()
        assert events.index("model") < events.index("tracking") < events.index("bundle")
        assert releases.resolve_model_directory(root=case.root) != case.legacy
        assert statuses == []
