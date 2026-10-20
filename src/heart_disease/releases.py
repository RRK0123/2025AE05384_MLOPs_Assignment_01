"""Validate immutable model releases and atomically select one complete release."""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from .config import ROOT

MODEL_FILES = ["model.joblib", "metadata.json", "feature_schema.json"]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_bundle(bundle, root, expected):
    """Keep approved serving bytes in the versioned tree, independent of dist/."""
    directory = Path(root) / "models/bundles"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{expected}.zip"
    descriptor, temporary = tempfile.mkstemp(prefix=".publishing-", dir=directory)
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream, Path(bundle).open("rb") as source:
            shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if checksum(pending) != expected:
            raise ValueError("Bundle changed while retaining its approved bytes.")
        try:
            os.link(pending, destination)
        except FileExistsError:
            if checksum(destination) != expected:
                raise ValueError("Retained bundle differs from its content identity.") from None
    finally:
        pending.unlink(missing_ok=True)
    return destination


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def publish_directory(source, destination):
    """Publish complete bytes with the destination parent's inherited permissions.

    Windows Python 3.13 temporary directories have owner-only ACLs. Renaming
    their children retains those ACLs, preventing the user's Docker process
    from reading a bundle prepared by a sandbox account. copytree copies file
    data/stat, not ACLs, into a sibling created under the normal output parent.
    """
    destination = Path(destination).resolve()
    pending = destination.parent / (".publishing-" + uuid.uuid4().hex)
    try:
        shutil.copytree(source, pending)
        pending.rename(destination)
    finally:
        if pending.exists():
            if pending.resolve().parent != destination.parent:
                raise ValueError("Refusing to clean a publication path outside its output parent")
            shutil.rmtree(pending)


def resolve_model_directory(model_directory=None, root=ROOT):
    root = Path(root).resolve()
    legacy = root / "models/selected"
    requested = Path(model_directory).resolve() if model_directory is not None else legacy
    pointer = root / "reports/modeling/latest.json"
    if requested != legacy or not pointer.exists():
        return requested
    selection = json.loads(pointer.read_text(encoding="utf-8"))
    if "model_directory" not in selection:
        return legacy  # Existing pre-release-layout checkout.
    release = (root / selection["model_directory"]).resolve()
    if not release.is_relative_to(root / "models/releases"):
        raise ValueError("Selected release path is outside models/releases.")
    if checksum(release / "release_manifest.json") != selection["release_manifest_sha256"]:
        raise ValueError("Selected release manifest checksum mismatch.")
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != selection["run_id"]:
        raise ValueError("Selected release run ID mismatch.")
    for name, expected in manifest["files"].items():
        path = (release / name).resolve()
        if not path.is_relative_to(release) or checksum(path) != expected:
            raise ValueError(f"Selected release checksum mismatch: {name}")
    return release


def promote_release(candidate, model_validation, tracking_validation, bundle, root=ROOT):
    root, candidate, bundle = (
        Path(root).resolve(),
        Path(candidate).resolve(),
        Path(bundle).resolve(),
    )
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    hashes = {name: checksum(candidate / name) for name in MODEL_FILES}
    if (
        model_validation["run_id"] != metadata["run_id"]
        or not model_validation["model_test_predictions_verified"]
        or not model_validation.get("tuning_tables_and_selected_parameters_reconciled")
        or not model_validation.get("fitted_classifier_parameters_match_tuning")
        or model_validation["model_files_sha256"] != hashes
    ):
        raise ValueError("Candidate differs from the model that passed validation.")
    if (
        tracking_validation["training_session"] != metadata["run_id"]
        or tracking_validation["parent_run_id"] != metadata["experiment_tracking"]["parent_run_id"]
        or not tracking_validation["all_finished"]
        or not tracking_validation.get(
            "tuning_tables_parameters_histories_and_artifacts_reconciled"
        )
    ):
        raise ValueError("Candidate does not have successful tracking validation.")
    bundle_hash = checksum(bundle)
    with zipfile.ZipFile(bundle) as archive:
        bundle_manifest = json.loads(archive.read("bundle_manifest.json"))
        if bundle_manifest["model_run_id"] != metadata["run_id"] or any(
            hashlib.sha256(archive.read(f"model/{name}")).hexdigest() != hashes[name]
            for name in MODEL_FILES
        ):
            raise ValueError("Bundle does not contain the validated candidate model.")
    bundle = retain_bundle(bundle, root, bundle_hash)
    releases = root / "models/releases"
    releases.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=releases) as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        for name in MODEL_FILES:
            shutil.copyfile(candidate / name, payload / name)
        if {name: checksum(payload / name) for name in MODEL_FILES} != hashes:
            raise ValueError("Candidate changed while staging its release.")
        write_json(payload / "model_validation.json", model_validation)
        write_json(payload / "tracking_validation.json", tracking_validation)
        write_json(
            payload / "bundle.json",
            {
                "path": bundle.relative_to(root).as_posix(),
                "sha256": bundle_hash,
            },
        )
        manifest = {
            "run_id": metadata["run_id"],
            "files": {path.name: checksum(path) for path in sorted(payload.iterdir())},
        }
        write_json(payload / "release_manifest.json", manifest)
        digest = checksum(payload / "release_manifest.json")
        destination = releases / f"{metadata['run_id']}-{digest}"
        if destination.exists():
            if checksum(destination / "release_manifest.json") != digest:
                raise ValueError("Existing immutable release does not match staged release.")
            for name, expected in manifest["files"].items():
                if checksum(destination / name) != expected:
                    raise ValueError("Existing immutable release contains changed files.")
        else:
            publish_directory(payload, destination)
    pointer = root / "reports/modeling/latest.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "run_id": metadata["run_id"],
        "report": f"{metadata['report_directory']}/model_development.md",
        "model_directory": destination.relative_to(root).as_posix(),
        "release_manifest_sha256": digest,
    }
    # All fallible validation/logging is completed before this single visibility switch.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".selection-", dir=pointer.parent)
    pending = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(selection, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, pointer)
    finally:
        pending.unlink(missing_ok=True)
    return destination
