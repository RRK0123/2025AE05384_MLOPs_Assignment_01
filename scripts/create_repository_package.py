"""Export a clean full repository without committing or copying historical caches."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def repository_files(project=PROJECT):
    project = Path(project)
    selected = json.loads((project / "reports/modeling/latest.json").read_text())
    folders = [
        ".github",
        "src",
        "tests",
        "scripts",
        "configs",
        "docs",
        "deployment",
        "data",
        selected["model_directory"],
        "models/selected",
        f"models/runs/{selected['run_id']}",
        f"reports/modeling/{selected['run_id']}",
    ]
    files = set()
    for name in folders:
        folder = project / name
        for path in folder.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and not any(
                    p in {"__pycache__", ".pytest_cache", ".ruff_cache"} or p.endswith(".egg-info")
                    for p in path.relative_to(project).parts
                )
            ):
                files.add(path)
    for path in project.iterdir():
        if path.is_file() and (
            path.suffix in {".bat", ".toml"}
            or path.name.startswith("requirements")
            or path.name
            in {
                "README.md",
                ".gitignore",
                ".gitattributes",
                ".dockerignore",
                "Dockerfile",
                "MLOps Assignment 1 Instructions.pdf",
            }
        ):
            files.add(path)
    files.add(project / "reports/modeling/latest.json")
    bundle = json.loads((project / selected["model_directory"] / "bundle.json").read_text())
    retained = project / "models/bundles" / (bundle["sha256"] + ".zip")
    if hashlib.sha256(retained.read_bytes()).hexdigest() != bundle["sha256"]:
        raise ValueError("Retained serving bundle hash differs")
    files.add(retained)
    return sorted(files)


def create(destination, project=PROJECT):
    project, destination = Path(project).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in repository_files(project):
            name = path.relative_to(project).as_posix()
            payload = path.read_bytes()
            archive.writestr("heart-disease-mlops/" + name, payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        archive.writestr(
            "heart-disease-mlops/REPOSITORY_FILES.json",
            json.dumps(
                {
                    "purpose": "Full repository and independent Windows/GitHub acceptance tracks",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "files_sha256": hashes,
                    "source_committed_by_exporter": False,
                    "historical_tracking_database_included": False,
                },
                indent=2,
            ),
        )
    with zipfile.ZipFile(destination) as check:
        if check.testzip() is not None:
            raise ValueError("Repository archive failed CRC verification")
        for name, digest in hashes.items():
            if hashlib.sha256(check.read("heart-disease-mlops/" + name)).hexdigest() != digest:
                raise ValueError(f"Exported file differs: {name}")
    return {
        "archive": str(destination),
        "files": len(hashes),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(create(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
