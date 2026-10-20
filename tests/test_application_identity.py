"""Bind the installed application to its wheel and preserve immutable releases."""

import zipfile

import pytest

from heart_disease.package_model import publish_archive
from heart_disease.verify_bundle import checksum, verify_application


@pytest.fixture
def application(tmp_path):
    package = tmp_path / "application"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "predict.py").write_text("def predict(): return 1\n")
    wheel = tmp_path / "application.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in package.glob("*.py"):
            archive.write(path, f"heart_disease/{path.name}")
    manifest = {
        "wheel": wheel.name,
        "application_source_sha256": {path.name: checksum(path) for path in package.glob("*.py")},
    }
    return tmp_path, package, wheel, manifest


def test_matching_package_wheel_and_manifest_pass(application):
    folder, package, _, manifest = application
    verify_application(folder, manifest, package)


@pytest.mark.parametrize("change", ["modified", "extra", "missing"])
def test_same_version_changed_application_rejected(application, change):
    folder, package, _, manifest = application
    if change == "modified":
        (package / "predict.py").write_text("def predict(): return 0\n")
    elif change == "extra":
        (package / "old_module.py").write_text("pass\n")
    else:
        (package / "predict.py").unlink()
    with pytest.raises(ValueError, match="Installed application code differs"):
        verify_application(folder, manifest, package)


def test_wheel_must_match_verified_source(application):
    folder, package, wheel, manifest = application
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("heart_disease/predict.py", "def predict(): return 0\n")
    with pytest.raises(ValueError, match="Bundled wheel code differs"):
        verify_application(folder, manifest, package)


def test_release_identity_preserves_older_application(tmp_path):
    first, second = tmp_path / "first.tmp", tmp_path / "second.tmp"
    first.write_bytes(b"application one")
    second.write_bytes(b"application two")
    old, _ = publish_archive(first, tmp_path, "same-model-0.1.1")
    new, _ = publish_archive(second, tmp_path, "same-model-0.1.1")
    assert old != new
    assert old.read_bytes() == b"application one"
    assert new.read_bytes() == b"application two"
    repeated, _ = publish_archive(first, tmp_path, "same-model-0.1.1")
    assert repeated == old


def test_corrupt_existing_release_is_not_overwritten(tmp_path):
    pending = tmp_path / "pending.tmp"
    pending.write_bytes(b"original")
    archive, _ = publish_archive(pending, tmp_path, "model")
    # Replace the directory entry rather than editing the shared hard-linked inode.
    archive.unlink()
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Existing release content"):
        publish_archive(pending, tmp_path, "model")
    assert archive.read_bytes() == b"corrupt"
