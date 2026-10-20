"""Build and exercise the Linux serving container; retain real Docker evidence."""

import argparse
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .prepare_docker import PROJECT, latest_bundle, prepare_context
    from .smoke_api import verify_service, wait_ready
else:
    from prepare_docker import PROJECT, latest_bundle, prepare_context
    from smoke_api import verify_service, wait_ready


def check_docker(bundle, output, image=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "build_executed": False, "container_started": False}
    container, docker, engine_ready = None, None, False

    def command(name, *args):
        log_path = output / f"{name}.log"
        try:
            # The child writes directly to disk, so a stalled build can be
            # diagnosed while it is running and abrupt exits retain its output.
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [docker, *args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
        except subprocess.TimeoutExpired as error:
            # TimeoutExpired can contain bytes even with text=True. Preserve the
            # partial build output, which is often the most useful diagnostic.
            partial = [
                part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part or ""
                for part in [error.stdout, error.stderr]
            ]
            with log_path.open("a", encoding="utf-8") as log:
                log.write("".join(partial) + "\nCommand timed out after 1200 seconds.\n")
            raise TimeoutError(
                f"Docker {name} timed out; see {output / (name + '.log')}"
            ) from error
        if completed.returncode:
            raise RuntimeError(f"Docker {name} failed; see {output / (name + '.log')}")
        return log_path.read_text(encoding="utf-8", errors="replace").strip()

    try:
        context = prepare_context(bundle)
        result["context"] = str(context)
        docker = shutil.which("docker")
        if docker is None:
            raise FileNotFoundError(
                "Docker CLI is not installed. Install/start a Linux Docker engine."
            )
        command("version", "version")
        if command("engine", "info", "--format", "{{.OSType}}") != "linux":
            raise RuntimeError("The Docker engine must use Linux containers")
        engine_ready = True
        image = image or "heart-disease:local-" + context.name[:16]
        result["image_tag"] = image
        result["build_executed"] = True
        command("build", "build", "--progress=plain", "--tag", image, str(context))
        image_info = json.loads(command("image", "image", "inspect", image))[0]
        result["image_id"] = image_info["Id"]
        if image_info["Config"]["User"] != "10001:10001":
            raise ValueError("Container image does not select the expected non-root user")
        container = "heart-smoke-" + uuid.uuid4().hex[:12]
        command(
            "run",
            "run",
            "--detach",
            "--name",
            container,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--publish",
            "127.0.0.1::8000",
            image,
        )
        result["container_started"] = True
        inspection = json.loads(command("inspect", "inspect", container))[0]
        port = inspection["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"]
        base_url = f"http://127.0.0.1:{port}"
        wait_ready(base_url)
        result["http_validation"] = verify_service(base_url, context / "bundle")
        for _ in range(60):
            health = command("health", "inspect", "--format", "{{.State.Health.Status}}", container)
            if health == "healthy":
                result["docker_healthcheck_verified"] = True
                break
            if health == "unhealthy":
                raise RuntimeError("Docker healthcheck reported an unhealthy container")
            time.sleep(1)
        else:
            raise TimeoutError("Docker healthcheck did not become healthy")
        # Verify the actual container user's identity and absence of training tooling.
        identity = command("uid", "exec", container, "id", "-u")
        if identity != "10001":
            raise ValueError("Serving process is not running as the intended non-root user")
        check = (
            "import importlib.util,json; print(json.dumps({n:importlib.util.find_spec(n) is None "
            "for n in ['mlflow','matplotlib','pytest','build']}))"
        )
        absent = json.loads(
            command("runtime-tools", "exec", container, "python", "-I", "-c", check)
        )
        if not all(absent.values()):
            raise ValueError("Training/development tools leaked into the serving image")
        result.update(status="passed", non_root_verified=True, training_tools_absent=absent)
    except BaseException as error:
        result.update(
            status="failed" if engine_ready else "blocked", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        if container and docker:
            for name, args in [
                ("container-logs", ["logs", container]),
                ("container-final", ["inspect", container]),
                ("cleanup", ["rm", "--force", container]),
            ]:
                try:
                    command(name, *args)
                except Exception as error:
                    result.setdefault("cleanup_errors", []).append(str(error))
            if result.get("cleanup_errors") and result["status"] == "passed":
                result["status"] = "failed"
        (output / "validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if result["status"] != "passed":
        raise RuntimeError("Container cleanup failed; see validation.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Explicit bundle for testing; default uses the approved selected release",
    )
    parser.add_argument("--image")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT
        / "reports/docker"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    args = parser.parse_args()
    print(
        json.dumps(check_docker(args.bundle or latest_bundle(), args.output, args.image), indent=2)
    )


if __name__ == "__main__":
    main()
