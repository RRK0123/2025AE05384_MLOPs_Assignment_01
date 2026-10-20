"""Build a deterministic wheel and reusable model bundle without retraining."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT
from .predict import load_predictor
from .schema import FEATURES
from .verify_bundle import checksum, verify_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dependency_pins(path):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1 or ";" in line or line.startswith("-"):
            raise ValueError("Runtime requirements must contain explicit package==version pins.")
        name, version = line.split("==")
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Build environment differs from the runtime pin for {name}.")
        result[name] = version
    normalized = {canonicalize_name(name): version for name, version in result.items()}
    for platform_name in ["win32", "linux"]:
        for name in result:
            for raw in importlib.metadata.requires(name) or []:
                dependency = Requirement(raw)
                if dependency.marker and not dependency.marker.evaluate(
                    {
                        "extra": "",
                        "sys_platform": platform_name,
                        "platform_system": "Windows" if platform_name == "win32" else "Linux",
                    }
                ):
                    continue
                version = normalized.get(canonicalize_name(dependency.name))
                if version is None or not dependency.specifier.contains(version):
                    raise ValueError(
                        f"Missing or incompatible runtime dependency pin: {dependency}"
                    )
    return result


def preprocessing_state(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    scaler = numeric.named_steps["scaler"]
    return {
        "input_features": list(pipeline.feature_names_in_),
        "transformed_features": list(preprocessor.get_feature_names_out()),
        "numeric_medians": numeric.named_steps["imputer"].statistics_.tolist(),
        "categorical_modes": categorical.named_steps["imputer"].statistics_.tolist(),
        "one_hot_categories": [
            values.tolist() for values in categorical.named_steps["encoder"].categories_
        ],
        "numeric_scaling": scaler != "passthrough",
        "numeric_mean": scaler.mean_.tolist() if scaler != "passthrough" else None,
        "numeric_scale": scaler.scale_.tolist() if scaler != "passthrough" else None,
        "fitted_on": "development partition only; no fitting occurs when loading this bundle",
    }


def prediction_fixtures():
    train = pd.read_csv(ROOT / "data/processed/train.csv")[FEATURES]
    records, descriptions = [], []
    selections = [(row, "development example") for _, row in train.head(3).iterrows()]
    selections += [
        (row, "development example with original missing value")
        for _, row in train.loc[train.isna().any(axis=1)].head(2).iterrows()
    ]
    for row, description in selections:
        records.append(
            {name: None if pd.isna(row[name]) else float(row[name]) for name in FEATURES}
        )
        descriptions.append(description)
    missing = records[0].copy()
    missing["chol"], missing["thal"] = None, None
    records.extend([missing, dict.fromkeys(FEATURES)])
    descriptions.extend(
        ["synthetic numeric and categorical nulls", "all predictors null; fitted imputation"]
    )
    return {"instances": records}, descriptions


def publish_archive(pending, output, stem):
    """Content-addressed release names preserve prior application/model combinations."""
    digest = checksum(pending)
    archive = Path(output) / f"{stem}-{digest}.zip"
    # Copy into the output parent first so Windows files inherit its normal ACLs,
    # rather than the owner-only ACL of a private TemporaryDirectory.
    descriptor, public_pending_name = tempfile.mkstemp(prefix=".publishing-", dir=output)
    public_pending = Path(public_pending_name)
    try:
        with os.fdopen(descriptor, "wb") as stream, Path(pending).open("rb") as source:
            shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Atomic publication still cannot overwrite an existing release.
            os.link(public_pending, archive)
        except FileExistsError:
            if checksum(archive) != digest:
                raise ValueError(
                    "Existing release content does not match its archive identity."
                ) from None
    finally:
        public_pending.unlink(missing_ok=True)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest


def build_bundle(output=ROOT / "dist", model_directory=None):
    from .releases import resolve_model_directory

    output, model_directory = Path(output).resolve(), resolve_model_directory(model_directory)
    predictor = load_predictor(model_directory)
    metadata = predictor.metadata
    runtime_versions = dependency_pins(ROOT / "requirements-runtime.txt")
    source_epoch = int(datetime.fromisoformat(metadata["created_at_utc"]).timestamp())
    output.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("heart-disease-mlops")
    stem = f"heart-disease-{metadata['run_id']}-{version}"
    with tempfile.TemporaryDirectory(prefix="heart-package-", dir=output) as temporary:
        staging = Path(temporary) / "bundle"
        (staging / "model").mkdir(parents=True)
        (staging / "fixtures").mkdir()
        (staging / "wheels").mkdir()
        build_env = {**os.environ, "SOURCE_DATE_EPOCH": str(source_epoch)}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(staging / "wheels"),
                str(ROOT),
            ],
            check=True,
            env=build_env,
        )
        wheels = list((staging / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one application wheel.")
        for name in ["model.joblib", "metadata.json", "feature_schema.json"]:
            shutil.copyfile(model_directory / name, staging / "model" / name)
        for name in ["requirements-runtime.txt", "docs/packaging.md"]:
            shutil.copyfile(ROOT / name, staging / Path(name).name)
        request, descriptions = prediction_fixtures()
        write_json(staging / "fixtures/request.json", request)
        write_json(staging / "fixtures/case_descriptions.json", descriptions)
        write_json(staging / "fixtures/expected_response.json", predictor.predict(request))
        write_json(
            staging / "model/preprocessing_state.json", preprocessing_state(predictor.pipeline)
        )
        files = {
            path.relative_to(staging).as_posix(): checksum(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "format_version": 1,
            "model_run_id": metadata["run_id"],
            "package_version": version,
            "python_major_minor": list(sys.version_info[:2]),
            "build_python_version": platform.python_version(),
            "build_tools": {
                name: importlib.metadata.version(name)
                for name in ["build", "setuptools", "wheel", "packaging", "pyproject_hooks"]
            },
            "runtime_versions": runtime_versions,
            "wheel": wheels[0].relative_to(staging).as_posix(),
            "source_date_epoch": source_epoch,
            "application_source_sha256": {
                path.name: checksum(path)
                for path in sorted((ROOT / "src/heart_disease").glob("*.py"))
            },
            "training_source_sha256": metadata["code_sha256"],
            "files": files,
        }
        write_json(staging / "bundle_manifest.json", manifest)
        verification = verify_bundle(staging)
        # Normalize paths, timestamps, ordering, permissions and compression settings.
        fixed_date = datetime.fromtimestamp(source_epoch, timezone.utc).timetuple()[:6]
        pending = Path(temporary) / "bundle.zip"
        with zipfile.ZipFile(
            pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zipped:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(
                        path.relative_to(staging).as_posix(), date_time=fixed_date
                    )
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    zipped.writestr(info, path.read_bytes(), compresslevel=9)
        archive, archive_hash = publish_archive(pending, output, stem)
        report = ROOT / "reports/packaging"
        report.mkdir(parents=True, exist_ok=True)
        write_json(report / "bundle_manifest.json", manifest)
        write_json(
            report / "build_validation.json",
            {
                "bundle": archive.relative_to(ROOT).as_posix()
                if archive.is_relative_to(ROOT)
                else str(archive),
                "archive_sha256": archive_hash,
                "build_environment_validation": verification,
            },
        )
        print(
            json.dumps(
                {
                    "bundle": str(archive),
                    "sha256": archive_hash,
                    "verified_prediction_cases": verification["prediction_cases_verified"],
                },
                indent=2,
            )
        )
        return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    build_bundle(args.output, args.model_dir)


if __name__ == "__main__":
    main()
