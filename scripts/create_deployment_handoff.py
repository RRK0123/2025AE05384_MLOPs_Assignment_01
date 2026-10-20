"""Create a small serving-only transfer ZIP for the other Windows PC."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .prepare_docker import PROJECT, latest_bundle, prepare_context
else:
    from prepare_docker import PROJECT, latest_bundle, prepare_context


def create_handoff(destination, *, project=PROJECT):
    project, destination = Path(project).resolve(), Path(destination).resolve()
    bundle = latest_bundle(project)
    prepare_context(bundle, project=project)
    pointer = json.loads((project / "reports/modeling/latest.json").read_text(encoding="utf-8"))
    release = (project / pointer["model_directory"]).resolve()
    if not release.is_relative_to(project / "models/releases"):
        raise ValueError("Invalid selected release directory")
    files = {
        project / name
        for name in [
            "RUN_DOCKER_KUBERNETES.bat",
            "COLLECT_DEPLOYMENT_LOGS.bat",
            "README.md",
            "Dockerfile",
            ".dockerignore",
            ".gitattributes",
            ".gitignore",
            "pyproject.toml",
            "reports/modeling/latest.json",
        ]
    }
    files.add(bundle)
    files.update(project.glob("requirements*.txt"))
    for directory in ["src", "scripts", "docs", "deployment"]:
        files.update(
            path
            for path in (project / directory).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".ps1", ".md", ".json", ".yaml"}
            and "__pycache__" not in path.parts
            and not any(p.endswith(".egg-info") for p in path.parts)
        )
    files.update(path for path in release.iterdir() if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = {path.relative_to(project).as_posix(): path for path in files}
    for name in ["START_HERE.md", "CAPTURE_BEFORE_LEAVING.md"]:
        # Put the essential guides beside the launchers, as well as keeping the
        # original documentation tree and its relative links intact.
        entries[name] = project / "deployment/windows" / name
    identity = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(entries.items())
    }
    # Never overwrite a previously prepared handoff.
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
        archive.writestr(
            "HANDOFF_FILES.json",
            json.dumps(
                {
                    "purpose": "Serving-only handoff; not the complete assignment submission",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_run_id": pointer["run_id"],
                    "files_sha256": identity,
                },
                indent=2,
            )
            + "\n",
        )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT
        / "dist"
        / (
            "windows-deployment-handoff-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + ".zip"
        ),
    )
    args = parser.parse_args()
    path = create_handoff(args.output)
    print(
        json.dumps(
            {
                "handoff": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
