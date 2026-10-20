"""Run CI from a fresh data workspace and retain evidence even on failure."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_workspace(project, output):
    """Copy only source, configuration, docs and the attributed raw data cache."""
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    for directory in ["src", "docs", "data/raw"]:
        shutil.copytree(
            project / directory,
            workspace / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    for path in [
        project / "pyproject.toml",
        project / "README.md",
        *project.glob("requirements*.txt"),
    ]:
        shutil.copyfile(path, workspace / path.name)
    return workspace


def run_step(name, command, *, cwd, env, output, results):
    record = {
        "name": name,
        "command": [str(part) for part in command],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(cwd),
        "environment": {
            key: env[key]
            for key in [
                "HEART_DISEASE_PROJECT_ROOT",
                "PYTHONUTF8",
                "MPLBACKEND",
                "OMP_NUM_THREADS",
                "ACCEPTANCE_RUN_STARTED_AT",
            ]
            if key in env
        },
    }
    results.append(record)
    write_json(output / "steps.json", results)
    started = time.monotonic()
    print(f"\n--- {name} ---", flush=True)
    try:
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            code = process.wait()
        record.update(returncode=code, status="passed" if code == 0 else "failed")
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "steps.json", results)


def run_pipeline(output, jobs=2, *, project=PROJECT, docker=False, evidence=False):
    output, project = Path(output).resolve(), Path(project).resolve()
    workspace = prepare_workspace(project, output)
    env = {
        **os.environ,
        "HEART_DISEASE_PROJECT_ROOT": str(workspace),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(output / "mplconfig"),
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    # Optimized Python would disable pytest assertions in application-independent tests.
    env.pop("PYTHONOPTIMIZE", None)
    # Keep subprocess/pytest temporary files within this run, avoiding inherited
    # Windows temp-directory ACLs and preserving diagnostics on the borrowed PC.
    temporary_files = output / "temporary"
    temporary_files.mkdir()
    env.update(TMP=str(temporary_files), TEMP=str(temporary_files), TMPDIR=str(temporary_files))
    results = []
    summary = {
        "status": "running",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workspace": str(workspace),
        "commit": os.environ.get("GITHUB_SHA"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_outputs_copied": False,
    }
    write_json(output / "summary.json", summary)
    if evidence:
        coverage_dir = output.parent / "coverage"
        coverage_dir.mkdir(exist_ok=True)
        coverage_config = coverage_dir / "coverage.ini"
        coverage_config.write_text(
            "[run]\nbranch = True\nparallel = True\nsource = heart_disease,scripts.acceptance\n"
            f"data_file = {(coverage_dir / '.coverage').as_posix()}\n"
            "[paths]\napplication =\n"
            f"    {(project / 'src/heart_disease').as_posix()}\n"
            f"    {(workspace / 'src/heart_disease').as_posix()}\n",
            encoding="utf-8",
        )

    def step(name, *args, executable=sys.executable, cwd=project, environment=None):
        step_env = dict(env if environment is None else environment)
        if evidence and executable == sys.executable and args[0] == "-m":
            if args[1] == "pytest":
                args = (*args, "-p", "scripts.acceptance.pytest_evidence")
                step_env["ACCEPTANCE_TEST_EVIDENCE"] = str(output / f"{name}-details.json")
            if args[1] == "pytest" or str(args[1]).startswith("heart_disease."):
                args = ("-m", "coverage", "run", f"--rcfile={coverage_config}", *args)
        run_step(
            name,
            [executable, *map(str, args)],
            cwd=cwd,
            env=step_env,
            output=output,
            results=results,
        )

    try:
        step("dependencies", "-m", "pip", "check")
        step("environment", "-m", "pip", "freeze")
        step("lint", "-m", "ruff", "check", "src", "tests", "scripts")
        step("format", "-m", "ruff", "format", "--check", "src", "tests", "scripts")
        step(
            "unit-tests",
            "-m",
            "pytest",
            "-m",
            "not integration",
            f"--junitxml={output / 'unit-tests.xml'}",
        )
        step("eda", "-m", "heart_disease.eda")
        step("data-validation", "-m", "heart_disease.validate")
        step("training", "-m", "heart_disease.train", "--jobs", jobs)
        step("model-validation", "-m", "heart_disease.validate_model")
        step("tracking-validation", "-m", "heart_disease.validate_tracking")
        step(
            "integration-tests",
            "-m",
            "pytest",
            "-m",
            "integration",
            f"--junitxml={output / 'integration-tests.xml'}",
        )
        # Training has already built a bundle. An independent rebuild must reuse its bytes.
        pointer = json.loads((workspace / "reports/modeling/latest.json").read_text())
        release = workspace / pointer["model_directory"]
        original = json.loads((release / "bundle.json").read_text())
        step("rebuild", "-m", "heart_disease.package_model")
        rebuilt = json.loads((workspace / "reports/packaging/build_validation.json").read_text())
        bundle = workspace / rebuilt["bundle"]
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if digest != original["sha256"] or digest != rebuilt["archive_sha256"]:
            raise ValueError("Rebuilt bundle differs from the validated training bundle")
        bundle_dir = output / "bundle"
        with zipfile.ZipFile(bundle) as archive:
            for name in archive.namelist():
                if not (bundle_dir / name).resolve().is_relative_to(bundle_dir.resolve()):
                    raise ValueError("Unsafe bundle member")
            archive.extractall(bundle_dir)
        step("bundle-validation", "-m", "heart_disease.verify_bundle", "--bundle", bundle_dir)
        # Install and run the wheel with inference dependencies only, outside the checkout.
        runtime = output / "runtime"
        venv.EnvBuilder(with_pip=True).create(runtime)
        python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        inference_workspace = output / "inference-only"
        inference_workspace.mkdir()
        inference_env = {**env, "HEART_DISEASE_PROJECT_ROOT": str(inference_workspace)}
        manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
        step(
            "runtime-install",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            bundle_dir / "requirements-runtime.txt",
            bundle_dir / manifest["wheel"],
            executable=python,
            cwd=inference_workspace,
            environment=inference_env,
        )
        step(
            "runtime-dependencies",
            "-I",
            "-m",
            "pip",
            "check",
            executable=python,
            cwd=inference_workspace,
            environment=inference_env,
        )
        step(
            "runtime-validation",
            "-I",
            "-m",
            "heart_disease.verify_bundle",
            "--bundle",
            bundle_dir,
            executable=python,
            cwd=inference_workspace,
            environment=inference_env,
        )
        step(
            "api-install",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            project / "requirements-api.txt",
            executable=python,
            cwd=inference_workspace,
            environment=inference_env,
        )
        step(
            "api-dependencies",
            "-I",
            "-m",
            "pip",
            "check",
            executable=python,
            cwd=inference_workspace,
            environment=inference_env,
        )
        step(
            "api-http-validation",
            project / "scripts/verify_api.py",
            "--python",
            python,
            "--bundle",
            bundle_dir,
            "--output",
            output / "api",
        )
        if docker:
            step(
                "docker-validation",
                project / "scripts/docker_smoke.py",
                "--bundle",
                bundle,
                "--output",
                output / "docker",
            )
        summary.update(
            status="passed",
            model_run_id=pointer["run_id"],
            bundle=str(bundle),
            bundle_sha256=digest,
            rebuild_identical=True,
            isolated_wheel_verified=True,
            inference_has_no_training_workspace=True,
            isolated_api_http_verified=True,
            docker_status="passed" if docker else "not_requested",
        )
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["steps"] = results
        write_json(output / "summary.json", summary)
        if path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(path).open("a", encoding="utf-8") as report:
                report.write(f"## CI: {summary['status']}\n\n")
                for result in results:
                    report.write(f"- {result['name']}: {result['status']}\n")
                report.write("\nDownload the evidence artifact for logs and reports.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports/ci" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--evidence", action="store_true", help="Collect test methods and branch coverage"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Require a Linux Docker build and HTTP test"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run_pipeline(args.output, args.jobs, docker=args.docker, evidence=args.evidence)


if __name__ == "__main__":
    main()
