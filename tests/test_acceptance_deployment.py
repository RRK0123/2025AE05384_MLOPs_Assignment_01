"""Regression for diagnostic JSON accidentally submitted as Kubernetes resources."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.acceptance import deployment


@pytest.mark.parametrize("track", ["github", "windows"])
@pytest.mark.parametrize("reject_validation", [False, True])
def test_deployment_applies_only_named_manifests_and_preserves_validation_gate(
    tmp_path, monkeypatch, track, reject_validation
):
    """Extra evidence files never reach kubectl; rejected dry runs block apply."""
    output = tmp_path / "run with spaces"
    manifests = output / "deployment"
    manifests.mkdir(parents=True)
    (manifests / "service.json").write_text('{"diagnostic": true}', encoding="utf-8")
    (manifests / "unrelated.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: unrelated\n", encoding="utf-8"
    )
    commands = []
    expected_files = [manifests / f"{name}.yaml" for name in ("namespace", "deployment", "service")]

    def command(name, args, **kwargs):
        args = [str(value) for value in args]
        commands.append((name, args))
        if name == "cluster-versions":
            return {
                "output": json.dumps(
                    {
                        "clientVersion": {"major": "1", "minor": "37"},
                        "serverVersion": {"major": "1", "minor": "37"},
                    }
                )
            }
        if name in {"server-validation", "apply"}:
            supplied = [Path(args[index + 1]) for index, value in enumerate(args) if value == "-f"]
            assert supplied == expected_files
            assert all(path.is_file() for path in supplied)
            assert [yaml.safe_load(path.read_text())["kind"] for path in supplied] == [
                "Namespace",
                "Deployment",
                "Service",
            ]
            assert (manifests / "versions.json").is_file()
        if name == "server-validation":
            assert "--dry-run=server" in args and "--validate=strict" in args
            if reject_validation:
                raise RuntimeError("simulated server validation rejection")
        if name == "rollout":
            raise RuntimeError("simulation stopped before live rollout")
        return {"output": ""}

    ledger = SimpleNamespace(
        output=output,
        result={"track": track, "run_id": "regression-12345678"},
        save=lambda: None,
        command=command,
        background=lambda *args: None,
    )
    monkeypatch.setattr(deployment.shutil, "which", lambda name: f"/stub/{name}")
    message = (
        "simulated server validation rejection"
        if reject_validation
        else "simulation stopped before live rollout"
    )
    with pytest.raises(RuntimeError, match=message):
        deployment.deploy(
            ledger,
            Path(__file__).resolve().parents[1],
            "heart-disease:regression",
            "model-regression",
        )
    names = [name for name, _ in commands]
    assert names.index("namespace") < names.index("server-validation")
    if reject_validation:
        assert "apply" not in names and "rollout" not in names
    else:
        assert names.index("server-validation") < names.index("apply") < names.index("rollout")
    rendered = yaml.safe_load((manifests / "deployment.yaml").read_text())
    assert (
        rendered["spec"]["template"]["spec"]["containers"][0]["image"] == "heart-disease:regression"
    )
