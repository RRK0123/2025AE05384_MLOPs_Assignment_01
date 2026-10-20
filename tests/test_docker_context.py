"""Protect the container's model/application identity and small build context."""

import json
import zipfile

import pytest

from scripts import prepare_docker


@pytest.fixture
def context_case(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    for name in prepare_docker.CONTEXT_FILES:
        (project / name).write_text(name)
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("model/model.joblib", "model")
        archive.writestr("requirements-runtime.txt", "requirements-runtime.txt")
    monkeypatch.setattr(prepare_docker, "verify_bundle", lambda path: {})
    return project, bundle


def test_context_reuses_identical_inputs_and_contains_no_dataset(context_case):
    project, bundle = context_case
    context = prepare_docker.prepare_context(bundle, project=project)
    assert prepare_docker.prepare_context(bundle, project=project) == context
    assert (context / "bundle/model/model.joblib").is_file()
    assert not (context / "data").exists()


@pytest.mark.parametrize("reseal", [False, True])
def test_modified_context_rejected_even_with_resealed_local_manifest(context_case, reseal):
    project, bundle = context_case
    context = prepare_docker.prepare_context(bundle, project=project)
    (context / "Dockerfile").write_text("changed")
    if reseal:
        manifest = json.loads((context / "context.json").read_text())
        manifest["files"]["Dockerfile"] = prepare_docker.checksum(context / "Dockerfile")
        (context / "context.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="context was modified"):
        prepare_docker.prepare_context(bundle, project=project)


def test_changed_application_build_files_preserve_previous_context(context_case):
    project, bundle = context_case
    first = prepare_docker.prepare_context(bundle, project=project)
    old = (first / "requirements-api.txt").read_bytes()
    (project / "requirements-api.txt").write_text("updated dependencies")
    second = prepare_docker.prepare_context(bundle, project=project)
    assert first != second
    assert (first / "requirements-api.txt").read_bytes() == old


def test_archive_path_escape_rejected_before_verifying_model(context_case):
    project, bundle = context_case
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("../outside.txt", "unexpected")
    with pytest.raises(ValueError, match="Unsafe bundle archive member"):
        prepare_docker.prepare_context(bundle, project=project)
