"""Critical track-separation and publication gates for the hosted workflow."""

from pathlib import Path

import yaml


def test_hosted_deployment_is_independent_and_never_runs_for_pull_requests():
    project = Path(__file__).resolve().parents[1]
    workflow = yaml.load((project / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["deploy"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["needs"] == "quality"
    assert job["if"] == "github.event_name != 'pull_request'"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--track github" in commands
    assert "--native-validation" not in commands
    assert "self-hosted" not in str(job)
    assert "cloud-provider-kind" in commands and "kind@" in commands
    assert job["permissions"] == {"contents": "read", "packages": "write", "issues": "write"}
    assert "git commit" not in commands and "git push" not in commands


def test_evidence_upload_and_cleanup_survive_a_failed_deployment():
    project = Path(__file__).resolve().parents[1]
    workflow = yaml.load((project / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["deploy"]["steps"]
    cleanup = next(step for step in steps if "--cleanup" in step.get("run", ""))
    archive = next(step for step in steps if "--finalize" in step.get("run", ""))
    upload = next(
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert cleanup["if"] == archive["if"] == upload["if"] == "always()"
    assert "*-evidence.zip" in upload["with"]["path"]
