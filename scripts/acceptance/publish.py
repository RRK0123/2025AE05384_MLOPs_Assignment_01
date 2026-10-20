"""Publish verified GitHub evidence; never commit or push repository source."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from scripts.acceptance.evidence import inventory, read_json, write_json


def github_api(endpoint, *, method="GET", payload=None):
    command = ["gh", "api", "--method", method, endpoint]
    if payload is None:
        response = subprocess.run(command, capture_output=True, text=True, check=True)
    else:
        with tempfile.TemporaryDirectory(prefix="heart-issue-") as directory:
            body = Path(directory) / "body.json"
            body.write_text(json.dumps(payload), encoding="utf-8")
            response = subprocess.run(
                [*command, "--input", str(body)], capture_output=True, text=True, check=True
            )
    return json.loads(response.stdout) if response.stdout.strip() else {}


def issue_payload(output, repository, run_id):
    request = read_json(Path(output) / "retraining/request.json")
    result = read_json(Path(output) / "results.json")
    if not request or not request.get("request_retraining") or not request.get("demo"):
        raise ValueError("This workflow publishes only explicit DEMO retraining requests")
    return {
        "title": "[DEMO] Review model degradation and retraining request",
        "body": "<!-- heart-mlops-retraining-demo -->\n\nControlled input-corruption demonstration; not a production incident.\n\n"
        f"Execution: https://github.com/{repository}/actions/runs/{run_id}\n\n"
        f"Model run: `{request['model_run_id']}`\n\nCommit: `{result.get('commit')}`\n\n"
        "Download the complete-github-evidence artifact for reports, statistics, screenshots and video.\n\n"
        "```json\n" + json.dumps(request, indent=2) + "\n```\n\n"
        "Requested action: review data quality and retraining criteria. No automatic retraining or model replacement occurred.",
    }


def publish_issue(output):
    repository = os.environ["GITHUB_REPOSITORY"]
    payload = issue_payload(output, repository, os.environ["GITHUB_RUN_ID"])
    issues = github_api(f"repos/{repository}/issues?state=open&per_page=100")
    existing = next(
        (
            item
            for item in issues
            if "<!-- heart-mlops-retraining-demo -->" in (item.get("body") or "")
            and "pull_request" not in item
        ),
        None,
    )
    endpoint = f"repos/{repository}/issues" + (f"/{existing['number']}" if existing else "")
    response = github_api(endpoint, method="PATCH" if existing else "POST", payload=payload)
    write_json(
        Path(output) / "retraining/github-issue.json",
        {
            "number": response["number"],
            "url": response["html_url"],
            "updated_existing": bool(existing),
        },
    )
    print(response["html_url"])


def publish_image(output):
    output = Path(output)
    result = read_json(output / "results.json")
    if result.get("status") != "passed" or not result.get("complete_acceptance"):
        raise ValueError("Only a fully verified image can be published")
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    remote = f"ghcr.io/{repository}:run-{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    source = result["deployed_image"]
    login = subprocess.run(
        [
            "docker",
            "login",
            "ghcr.io",
            "--username",
            os.environ["GITHUB_ACTOR"],
            "--password-stdin",
        ],
        input=os.environ["GH_TOKEN"],
        text=True,
        capture_output=True,
    )
    if login.returncode:
        raise RuntimeError("GHCR login failed; check packages:write permission")
    subprocess.run(["docker", "tag", source, remote], check=True)
    log = output / "deployment/registry-publication.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["docker", "push", remote],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )
    result["published_image"] = remote
    write_json(output / "results.json", result)
    write_json(
        output / "deployment/registry-publication.json",
        {
            "source_image": source,
            "published_image": remote,
            "verification": "Published only after deployment and acceptance passed",
        },
    )
    print(remote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issue", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if not (args.output / "results.json").exists():
        print("No suite result yet; bootstrap logs will be uploaded.")
        return
    if args.issue or args.image:
        if os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
            raise RuntimeError("Publishing is restricted to trusted push/manual workflows")
        result = read_json(args.output / "results.json")
        if result.get("status") != "passed" or result.get("track") != "github":
            raise ValueError("Only a successful independent GitHub run can publish")
    if args.image or args.issue:
        started = time.monotonic()
        case = {
            "id": "GHCR-001" if args.image else "ISSUE-001",
            "requirements": ["R5", "R8"] if args.issue else ["R6"],
            "purpose": "Publish verified image"
            if args.image
            else "Publish DEMO retraining request",
            "method": "Run authenticated GitHub workflow publication using its scoped built-in token.",
            "expected": "Publication succeeds and its identity/link is retained.",
            "status": "running",
        }
        try:
            if args.image:
                publish_image(args.output)
                case["actual"] = read_json(args.output / "deployment/registry-publication.json")
            else:
                publish_issue(args.output)
                case["actual"] = read_json(args.output / "retraining/github-issue.json")
            case["status"] = "passed"
        except Exception as error:
            case.update(status="failed", actual=f"{type(error).__name__}: {error}")
            raise
        finally:
            result = read_json(args.output / "results.json")
            case["seconds"] = round(time.monotonic() - started, 3)
            result["cases"].append(case)
            if case["status"] == "failed":
                result.update(status="failed", complete_acceptance=False)
            write_json(args.output / "results.json", result)
    if args.finalize:
        from scripts.acceptance.reporting import generate
        from scripts.acceptance.run import archive_evidence

        generate(args.output)
        inventory(args.output)
        print(archive_evidence(args.output))
    if args.cleanup:
        result = read_json(args.output / "results.json")
        cluster = result.get("cluster", {})
        if (
            result.get("track") == "github"
            and cluster.get("owned_by_run")
            and cluster.get("creation_attempted")
            and not cluster.get("deleted_after_evidence")
        ):
            subprocess.run(
                ["kind", "delete", "cluster", "--name", cluster["name"]], check=True, timeout=180
            )


if __name__ == "__main__":
    main()
