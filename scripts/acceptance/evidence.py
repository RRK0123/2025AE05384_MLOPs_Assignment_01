"""Atomic evidence ledger, live state, process ownership, and bounded logging."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_replace = os.replace


def atomic_replace(source, target):
    # Windows denies replacement while a live dashboard reader briefly owns the
    # old file without delete sharing. Retry the completed snapshot, never expose
    # a half-written JSON document or abort the test pipeline on a brief read.
    for attempt in range(61):
        try:
            _replace(source, target)
            return
        except PermissionError:
            if attempt == 60:
                raise
            time.sleep(0.05)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    atomic_replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def stop(process):
    if process and process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=20,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


class Ledger:
    def __init__(self, output, track, config):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.processes = []
        self.result = {
            "schema_version": 1,
            "run_id": self.output.name,
            "track": track,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "monotonic_origin": self.started,
            "python": sys.version,
            "platform": platform.platform(),
            "commit": os.environ.get("GITHUB_SHA"),
            "config": config,
            "cases": [],
            "historical_results_used": False,
        }
        project = Path(__file__).resolve().parents[2]
        if not self.result["commit"] and (project / ".git").exists() and shutil.which("git"):
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if revision.returncode == 0:
                self.result["commit"] = revision.stdout.strip()
        self.result["source_identity"] = {
            path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ["src", "scripts", "tests", "configs", ".github"]
            for path in (project / folder).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".ps1"}
            and "__pycache__" not in path.parts
        }
        write_json(self.output / "config.json", config)
        self.save()

    def save(self):
        self.result["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        write_json(self.output / "results.json", self.result)

    @contextmanager
    def case(self, identifier, title, requirements, method, expected, inputs=None):
        record = {
            "id": identifier,
            "purpose": title,
            "requirements": requirements,
            "method": method,
            "expected": expected,
            "inputs": inputs or {},
            "status": "running",
            "start_seconds": round(time.monotonic() - self.started, 3),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "evidence": [],
        }
        self.result["cases"].append(record)
        self.result["current_stage"] = title
        self.save()
        print(f"\n[{identifier}] {title}", flush=True)
        started = time.monotonic()
        try:
            yield record
            record["status"] = "passed"
            record.setdefault("actual", "All explicit checks completed successfully.")
        except BaseException as error:
            record.update(status="failed", actual=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)
            self.save()

    def command(self, name, command, *, env=None, timeout=3600, required=True):
        log = self.output / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        code = None
        process = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    list(map(str, command)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.processes.append(process)
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f"{name} exceeded {timeout}s; see {log}")
                    self.save()
                    time.sleep(0.5)
                code = process.returncode
            if code and required:
                raise RuntimeError(f"{name} exited {code}; see {log}")
        finally:
            if process and process.poll() is None:
                stop(process)
        return {
            "returncode": code,
            "log": str(log.relative_to(self.output)),
            "output": log.read_text(encoding="utf-8", errors="replace"),
        }

    def background(self, name, command, *, env=None):
        log = self.output / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                list(map(str, command)),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        self.processes.append(process)
        return process

    def snapshot(self, name, url=None, selector=None, **actions):
        queue = self.output / "media-commands.jsonl"
        with queue.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "name": name,
                        "url": url,
                        "selector": selector,
                        "seconds": round(time.monotonic() - self.started, 3),
                        **actions,
                    }
                )
                + "\n"
            )
        deadline = time.monotonic() + 90
        destination = self.output / "screenshots" / f"{name}.json"
        while time.monotonic() < deadline:
            result = read_json(destination)
            if result:
                if result["status"] != "passed":
                    raise RuntimeError(f"Screenshot {name}: {result.get('error')}")
                return result
            time.sleep(0.25)
        raise TimeoutError(f"Screenshot {name} was not acknowledged")


def inventory(output):
    output = Path(output)
    files = {}
    for path in evidence_paths(output):
        if not included_evidence(path, output):
            continue
        if path.name == "evidence-manifest.json" or path.name.endswith(".tmp"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.relative_to(output).as_posix()] = {
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    write_json(output / "evidence-manifest.json", files)
    return files


def included_evidence(path, output):
    return (
        path.is_file()
        and not path.is_symlink()
        and not any(
            part
            in {
                "runtime",
                "inference-only",
                "workspace",
                "browser-backup",
                "bundle",
                "temporary",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
            }
            for part in path.relative_to(output).parts
        )
        and not path.name.endswith(".tmp")
    )


def evidence_paths(output):
    output = Path(output)
    excluded = {
        "runtime",
        "inference-only",
        "workspace",
        "browser-backup",
        "bundle",
        "temporary",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    for directory, folders, files in os.walk(output):
        folders[:] = sorted(
            name
            for name in folders
            if name not in excluded and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if included_evidence(path, output):
                yield path


def archive_evidence(output):
    """Publish a bounded snapshot whose manifest hashes the actual ZIP entries."""
    output = Path(output)
    destination = output.with_name(output.name + "-evidence.zip")
    temporary = destination.with_suffix(".partial")
    snapshots = {}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in evidence_paths(output):
            relative = path.relative_to(output)
            if path.name == "evidence-manifest.json":
                continue
            digest, byte_count = hashlib.sha256(), 0
            with (
                path.open("rb") as source,
                archive.open(relative.as_posix(), "w", force_zip64=True) as target,
            ):
                remaining = path.stat().st_size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    remaining -= len(chunk)
            snapshots[relative.as_posix()] = {"sha256": digest.hexdigest(), "bytes": byte_count}
        archive.writestr("evidence-manifest.json", json.dumps(snapshots, indent=2))
    atomic_replace(temporary, destination)
    write_json(output / "evidence-manifest.json", snapshots)
    return destination
