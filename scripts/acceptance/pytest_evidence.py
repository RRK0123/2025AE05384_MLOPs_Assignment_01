"""Persist actual pytest outcomes and readable methods without changing assertions."""

import ast
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.acceptance.evidence import atomic_replace as _atomic_replace

REQUIREMENTS = {
    "data": ["R1"],
    "preprocessing": ["R1", "R2", "R4"],
    "model": ["R2", "R4"],
    "cv": ["R2"],
    "tuning": ["R2", "R3"],
    "tracking": ["R3"],
    "packaging": ["R4"],
    "application": ["R4"],
    "release": ["R4"],
    "ci": ["R5"],
    "api": ["R6"],
    "docker": ["R6"],
    "deployment": ["R7"],
    "monitoring": ["R8"],
    "acceptance": ["R5", "R8", "R9"],
}
_records = {}
_destination = None
# Release tests intentionally monkeypatch os.replace. Evidence must not invoke
# the application's injected fault while pytest reports the completed test.


def requirements_for(name):
    return sorted({r for key, values in REQUIREMENTS.items() if key in name for r in values})


def save():
    if _destination:
        _destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(list(_records.values()), indent=2), encoding="utf-8")
        _atomic_replace(temporary, _destination)


def pytest_collection_finish(session):
    global _destination
    target = os.environ.get("ACCEPTANCE_TEST_EVIDENCE")
    if not target:
        return
    _destination = Path(target)
    for item in session.items:
        try:
            source = inspect.getsource(item.obj)
            import textwrap

            tree = ast.parse(textwrap.dedent(source))
            checks = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        except (OSError, TypeError, SyntaxError):
            source, checks = "Source unavailable", []
        _records[item.nodeid] = {
            "id": item.nodeid,
            "purpose": inspect.getdoc(item.obj) or item.name.replace("_", " "),
            "requirements": requirements_for(item.nodeid),
            "method": "Execute pytest test and its declared fixtures; evaluate its assertions.",
            "source": str(item.path),
            "source_line": item.location[1] + 1,
            "inputs": {
                k: repr(v)
                for k, v in getattr(
                    item, "callspec", type("P", (), {"params": {}})()
                ).params.items()
            },
            "fixtures": list(item.fixturenames),
            "fixture_definitions": {},
            "expected": checks or ["Test completes without an unexpected exception."],
            "test_source": source,
            "status": "not_run",
            "seconds": 0,
            "phases": [],
        }
        for name, definitions in item._fixtureinfo.name2fixturedefs.items():
            fixture = definitions[-1].func
            file = inspect.getsourcefile(fixture)
            if file and Path(file).resolve().is_relative_to(
                Path(__file__).resolve().parents[2] / "tests"
            ):
                _records[item.nodeid]["fixture_definitions"][name] = {
                    "source": file,
                    "definition": inspect.getsource(fixture),
                }
    save()


def pytest_runtest_logreport(report):
    record = _records.get(report.nodeid)
    if record is None:
        return
    if "started_at" not in record and getattr(report, "start", None) is not None:
        record["started_at"] = datetime.fromtimestamp(report.start, timezone.utc).isoformat()
        origin = os.environ.get("ACCEPTANCE_RUN_STARTED_AT")
        if origin:
            record["start_seconds"] = max(
                0, report.start - datetime.fromisoformat(origin).timestamp()
            )
    record["seconds"] += report.duration
    record["phases"].append({"phase": report.when, "outcome": report.outcome})
    if report.failed:
        record.update(status="failed", actual=str(report.longrepr))
    elif report.skipped and record["status"] != "failed":
        record.update(status="skipped", actual=str(report.longrepr))
    elif report.when == "call" and record["status"] not in {"failed", "skipped"}:
        record.update(
            status="passed", actual="Pytest completed all executed assertions successfully."
        )
    save()
