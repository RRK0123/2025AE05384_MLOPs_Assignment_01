"""Verify a model bundle's integrity, environment and prediction fixtures."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .predict import load_predictor


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported bundle manifest version.")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        path = (directory / name).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(directory):
            raise ValueError("Invalid bundle manifest file path.")
        if not path.is_file() or checksum(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if actual != set(manifest["files"]) | {"bundle_manifest.json"}:
        raise ValueError("Bundle contains missing or unexpected files.")
    return manifest


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = verify_files(directory)
    if list(sys.version_info[:2]) != manifest["python_major_minor"]:
        raise ValueError("Use the bundle's documented Python major/minor version.")
    for name, version in manifest["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime version mismatch: {name}; install requirements-runtime.txt.")
    if importlib.metadata.version("heart-disease-mlops") != manifest["package_version"]:
        raise ValueError("Installed application version differs from the bundle wheel.")
    verify_application(directory, manifest)
    request = json.loads((directory / "fixtures/request.json").read_text(encoding="utf-8"))
    expected = json.loads(
        (directory / "fixtures/expected_response.json").read_text(encoding="utf-8")
    )
    predictor = load_predictor(directory / "model")
    actual = predictor.predict(request)
    if actual["model_run_id"] != expected["model_run_id"]:
        raise ValueError("Prediction fixture refers to another model run.")
    if len(actual["predictions"]) != len(expected["predictions"]):
        raise ValueError("Prediction count differs from the packaged fixtures.")
    for observed, reference in zip(actual["predictions"], expected["predictions"]):
        if observed["prediction"] != reference["prediction"]:
            raise ValueError("Prediction class differs from the packaged fixture.")
        np.testing.assert_allclose(
            [observed["probability_disease"], observed["confidence"]],
            [reference["probability_disease"], reference["confidence"]],
            rtol=1e-12,
            atol=1e-12,
        )
    return {
        "model_run_id": actual["model_run_id"],
        "bundle_files_verified": len(manifest["files"]),
        "pinned_runtime_verified": True,
        "installed_application_code_verified": True,
        "python_version": sys.version.split()[0],
        "prediction_cases_verified": len(actual["predictions"]),
        "missing_value_cases_included": any(
            any(value is None for value in record.values()) for record in request["instances"]
        ),
        "preprocessing_and_classifier_loaded_together": True,
        "prediction_tolerance": "rtol=1e-12, atol=1e-12; exact class equality",
        "installed_module": str(Path(__file__).resolve()),
    }


def verify_application(directory, manifest, package_directory=None):
    """Bind manifest, bundled wheel and imported package to the same source bytes."""
    package_directory = Path(package_directory or Path(__file__).resolve().parent)
    expected = manifest["application_source_sha256"]
    actual = {path.name: checksum(path) for path in package_directory.glob("*.py")}
    if actual != expected:
        different = sorted(
            name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)
        )
        raise ValueError("Installed application code differs from bundle: " + ", ".join(different))
    wheel = Path(directory) / manifest["wheel"]
    with zipfile.ZipFile(wheel) as archive:
        wheel_code = {
            name.removeprefix("heart_disease/"): hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("heart_disease/") and name.endswith(".py")
        }
    if wheel_code != expected:
        raise ValueError("Bundled wheel code differs from the application manifest.")
    # Detect mixed imports as well as a consistently imported wrong package.
    if package_directory.resolve() == Path(__file__).resolve().parent:
        for name, module in tuple(sys.modules.items()):
            if name == "heart_disease" or name.startswith("heart_disease."):
                source = getattr(module, "__file__", None)
                if source and Path(source).resolve().parent != package_directory.resolve():
                    raise ValueError(
                        f"Application module imported outside verified package: {name}"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Extracted bundle directory.")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2))


if __name__ == "__main__":
    main()
