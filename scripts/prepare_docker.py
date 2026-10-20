"""Prepare a small, immutable Docker context from the verified serving bundle."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from heart_disease.releases import MODEL_FILES, publish_directory, resolve_model_directory
from heart_disease.verify_bundle import verify_bundle

PROJECT = Path(__file__).resolve().parents[1]
CONTEXT_FILES = ["Dockerfile", ".dockerignore", "requirements-api.txt", "requirements-runtime.txt"]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_context(bundle, *, project=PROJECT, output=None):
    bundle, project = Path(bundle).resolve(), Path(project).resolve()
    output = Path(output).resolve() if output else project / "build/docker"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "bundle_sha256": checksum(bundle),
        "build_files": {name: checksum(project / name) for name in CONTEXT_FILES},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = output / digest
    with tempfile.TemporaryDirectory(prefix=".context-", dir=output) as temporary:
        staging = Path(temporary) / "context"
        extracted = staging / "bundle"
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(bundle) as archive:
            for name in archive.namelist():
                if not (extracted / name).resolve().is_relative_to(extracted):
                    raise ValueError("Unsafe bundle archive member")
            archive.extractall(extracted)
        verify_bundle(extracted)
        for name in CONTEXT_FILES:
            shutil.copyfile(project / name, staging / name)
        files = {
            p.relative_to(staging).as_posix(): checksum(p)
            for p in staging.rglob("*")
            if p.is_file()
        }
        if (
            any(files[name] != identity["build_files"][name] for name in CONTEXT_FILES)
            or checksum(bundle) != identity["bundle_sha256"]
        ):
            raise ValueError("Docker build inputs changed while preparing the context")
        if checksum(staging / "requirements-runtime.txt") != checksum(
            extracted / "requirements-runtime.txt"
        ):
            raise ValueError("Serving runtime pins differ from the verified bundle")
        if destination.exists():
            recorded = json.loads((destination / "context.json").read_text())
            actual = {
                p.relative_to(destination).as_posix(): checksum(p)
                for p in destination.rglob("*")
                if p.is_file() and p != destination / "context.json"
            }
            if recorded["identity"] != identity or actual != files or recorded["files"] != files:
                raise ValueError("Existing Docker context was modified; refusing to overwrite it")
            return destination
        (staging / "context.json").write_text(
            json.dumps({"identity": identity, "files": files}, indent=2) + "\n", encoding="utf-8"
        )
        # Content-addressed paths retain older model/application build contexts.
        publish_directory(staging, destination)
    return destination


def latest_bundle(project=PROJECT):
    """Resolve the approved release, never an unpromoted last-build report."""
    project = Path(project).resolve()
    release = resolve_model_directory(root=project)
    if not release.is_relative_to(project / "models/releases"):
        raise ValueError(
            "No published model release; train successfully or pass --bundle explicitly"
        )
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    if not set([*MODEL_FILES, "bundle.json"]).issubset(manifest["files"]):
        raise ValueError("Selected release does not bind its model files and bundle reference")
    metadata = json.loads((release / "metadata.json").read_text(encoding="utf-8"))
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("Selected release model identity mismatch")
    reference = json.loads((release / "bundle.json").read_text(encoding="utf-8"))
    relative = Path(reference["path"])
    bundle = (project / relative).resolve()
    if (
        relative.is_absolute()
        or relative.drive
        or ".." in relative.parts
        or not bundle.is_relative_to(project)
    ):
        raise ValueError("Selected bundle path must remain inside the project")
    if not bundle.is_file():
        raise FileNotFoundError(
            f"Selected release bundle is missing: {bundle}; restore models/bundles from "
            "the same repository revision or release backup. Legacy dist references may "
            "require the original archive or successful retraining."
        )
    if checksum(bundle) != reference["sha256"]:
        raise ValueError("Selected release bundle checksum mismatch")
    with zipfile.ZipFile(bundle) as archive:
        bundled_manifest = json.loads(archive.read("bundle_manifest.json"))
        if bundled_manifest["model_run_id"] != manifest["run_id"]:
            raise ValueError("Selected bundle model identity mismatch")
        for name in MODEL_FILES:
            if hashlib.sha256(archive.read(f"model/{name}")).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Selected bundle differs from approved model: {name}")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Explicit bundle for testing; default uses the approved selected release",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context = prepare_context(args.bundle or latest_bundle(), output=args.output)
    print(json.dumps({"context": str(context), "identity": context.name}, indent=2))


if __name__ == "__main__":
    main()
