"""Record the actual desktop (Windows/Xvfb) and capture real browser pages."""

import argparse
import functools
import html
import json
import os
import shutil
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from scripts.acceptance.evidence import read_json, write_json

LIVE_PAGE = """<!doctype html><meta charset="utf-8"><title>Acceptance execution</title>
<style>body{font:20px system-ui;background:#101e2b;color:#e8f1f5;margin:32px}h1{color:#62d5c8}
pre{font:15px Consolas,monospace;white-space:pre-wrap;background:#182c3b;padding:20px;max-height:540px;overflow:auto}
.tag{color:#f0c66c}li{margin:8px}</style><h1>Heart Disease MLOps | live execution</h1>
<div id="head"></div><pre id="log">Starting evidence capture...</pre><div id="cases"></div>
<script>async function update(){try{let s=await(await fetch('/live')).json();
document.getElementById('head').textContent=s.title;
document.getElementById('log').textContent=s.log;
document.getElementById('cases').textContent=s.cases;
}catch(e){} }setInterval(update,1000);update();</script>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/stage/"):
            name = self.path.removeprefix("/stage/")
            root = Path(self.directory)
            steps = read_json(root / "ci/steps.json", [])
            record = next((step for step in steps if step["name"] == name), None)
            if record is None:
                self.send_error(404)
                return
            log = (root / "ci" / (name + ".log")).read_text(encoding="utf-8", errors="replace")
            body = (
                "<!doctype html><meta charset='utf-8'><style>body{font:20px system-ui;margin:30px;background:#101e2b;color:#e8f1f5}pre{font:15px Consolas;white-space:pre-wrap}</style>"
                + "<h1>Completed stage: "
                + html.escape(name)
                + "</h1><pre>"
                + html.escape(json.dumps(record, indent=2))
                + "</pre><pre>"
                + html.escape(log[-14000:])
                + "</pre>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/live":
            root = Path(self.directory)
            result = read_json(root / "results.json", {})
            steps = read_json(root / "ci/steps.json", [])
            stage = result.get("current_stage", "Starting")
            log = ""
            if stage == "Fresh model pipeline and automated tests" and steps:
                stage += " / " + steps[-1]["name"]
                path = root / "ci" / (steps[-1]["name"] + ".log")
            elif stage == "Publish verified image":
                path = root / "deployment/registry-publication.log"
            else:
                paths = list((root / "logs").glob("*.log"))
                path = max(paths, key=lambda p: p.stat().st_mtime) if paths else None
            if path and path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 12000))
                    log = stream.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - result.get("monotonic_origin", time.monotonic())
            body = json.dumps(
                {
                    "title": f"{result.get('track')} | {int(elapsed)}s | {stage}",
                    "log": log[-9000:],
                    "cases": " | ".join(
                        f"{c['id']}: {c['status']}" for c in result.get("cases", [])
                    ),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


def browser_options():
    for candidate in [
        os.environ.get("ACCEPTANCE_BROWSER"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]:
        if candidate and Path(candidate).is_file():
            return {"executable_path": candidate}
    return {}


def capture(output):
    from imageio_ffmpeg import get_ffmpeg_exe
    from playwright.sync_api import sync_playwright

    output = Path(output).resolve()
    video = output / "video"
    screenshots = output / "screenshots"
    video.mkdir(exist_ok=True)
    screenshots.mkdir(exist_ok=True)
    (output / "live.html").write_text(LIVE_PAGE, encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(output))
    )
    Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    recorder, browser, context = None, None, None
    log = (video / "recorder.log").open("w", encoding="utf-8")
    result = {"status": "starting", "live_url": url + "/live.html"}
    try:
        ffmpeg = shutil.which("ffmpeg") or get_ffmpeg_exe()
        if os.name == "nt":
            source = ["-f", "gdigrab", "-framerate", "8", "-draw_mouse", "0", "-i", "desktop"]
        else:
            if not os.environ.get("DISPLAY"):
                raise RuntimeError("A DISPLAY/Xvfb session is required for real execution video")
            source = [
                "-f",
                "x11grab",
                "-framerate",
                "8",
                "-video_size",
                "1440x900",
                "-i",
                os.environ["DISPLAY"],
            ]
        recorder = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                *source,
                "-vf",
                "scale=1280:-2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "29",
                "-pix_fmt",
                "yuv420p",
                str(video / "full-execution.mp4"),
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result["capture_start_seconds"] = (
            time.monotonic() - read_json(output / "results.json")["monotonic_origin"]
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=False, **browser_options(), args=["--window-size=1440,900"]
            )
            context = browser.new_context(
                viewport={"width": 1400, "height": 800},
            )
            page = context.new_page()
            page.goto(url + "/live.html")
            page.bring_to_front()
            page.wait_for_timeout(1500)
            if recorder.poll() is not None:
                raise RuntimeError("Screen recorder exited; see video/recorder.log")
            result.update(
                status="ready",
                browser_version=browser.version,
                capture="Actual desktop screen, continuous, without time compression",
                recorder_pid=recorder.pid,
            )
            write_json(output / "media-status.json", result)
            consumed = 0
            captured_steps = set()
            mlflow_origins = set()
            while not (output / "stop-media.json").exists():
                queue = output / "media-commands.jsonl"
                commands = queue.read_text(encoding="utf-8").splitlines() if queue.exists() else []
                for line in commands[consumed:]:
                    command = json.loads(line)
                    name = command["name"]
                    try:
                        destination = command.get("url") or url + "/live.html"
                        if destination.startswith("/"):
                            destination = url + destination
                        page.goto(destination, wait_until="domcontentloaded", timeout=30000)
                        page.bring_to_front()
                        if command.get("action") == "mlflow":
                            origin = destination.split("/#", 1)[0]
                            if origin not in mlflow_origins:
                                page.get_by_text("Model training", exact=True).click(timeout=30000)
                                mlflow_origins.add(origin)
                                if command.get("view") != "experiment":
                                    page.goto(destination, wait_until="domcontentloaded")
                            page.get_by_text(command["run_name"], exact=True).first.wait_for(
                                timeout=30000
                            )
                            close = page.get_by_role("button", name="Close", exact=True)
                            if close.count() and close.first.is_visible():
                                close.first.click()
                            if command.get("view") in {"metrics", "parameters"}:
                                page.get_by_role("tab", name="Overview", exact=True).click()
                                field = (
                                    "selected_test_roc_auc"
                                    if command["view"] == "metrics"
                                    else "selected_parameters"
                                )
                                page.get_by_text(
                                    field, exact=True
                                ).first.scroll_into_view_if_needed(timeout=30000)
                            elif command.get("view") == "artifacts":
                                page.get_by_role("tab", name="Artifacts", exact=True).click()
                                page.get_by_text("metadata.json", exact=False).first.wait_for(
                                    timeout=30000
                                )
                                page.get_by_text("metadata.json", exact=False).first.click()
                                page.get_by_text("selected_family", exact=False).first.wait_for(
                                    timeout=30000
                                )
                        if command.get("action") == "predict":
                            block = page.locator(".opblock-post").filter(has_text="/predict")
                            block.locator(".opblock-summary").click()
                            block.get_by_role("button", name="Try it out").click()
                            block.locator("textarea").fill(json.dumps(command["payload"], indent=2))
                            block.get_by_role("button", name="Execute", exact=True).click()
                            block.locator(".responses-inner").get_by_text(
                                command["model_run_id"], exact=False
                            ).first.wait_for(timeout=20000)
                        if command.get("selector"):
                            page.locator(command["selector"]).first.wait_for(timeout=20000)
                        page.evaluate(
                            """label => { const banner=document.createElement('div');
                            banner.textContent=label; banner.style.cssText='position:fixed;bottom:0;left:0;right:0;padding:7px 14px;background:#10343e;color:white;font:14px system-ui;z-index:2147483647';
                            document.body.appendChild(banner); }""",
                            f"Evidence: {name} | Run: {output.name}",
                        )
                        page.wait_for_timeout(2000)
                        page.screenshot(
                            path=str(screenshots / f"{name}.png"),
                            full_page=command.get("full_page", True),
                        )
                        import coverage

                        if coverage.Coverage.current():
                            coverage.Coverage.current().save()
                        write_json(
                            screenshots / f"{name}.json",
                            {
                                "status": "passed",
                                **command,
                                "file": f"screenshots/{name}.png",
                                "url": destination,
                            },
                        )
                    except Exception as error:
                        write_json(
                            screenshots / f"{name}.json", {"status": "failed", "error": str(error)}
                        )
                    finally:
                        page.goto(url + "/live.html")
                    consumed += 1
                for step in read_json(output / "ci/steps.json", []):
                    if step["status"] == "passed" and step["name"] not in captured_steps:
                        page.goto(url + "/stage/" + step["name"])
                        page.wait_for_timeout(750)
                        page.screenshot(path=str(screenshots / f"stage-{step['name']}.png"))
                        captured_steps.add(step["name"])
                        page.goto(url + "/live.html")
                if recorder.poll() is not None:
                    raise RuntimeError("Screen recorder stopped before execution finished")
                page.bring_to_front()
                page.wait_for_timeout(500)
            context.close()
            context = None
            browser.close()
            browser = None
        recorder.communicate(input=b"q\n", timeout=45)
        if recorder.returncode or not (video / "full-execution.mp4").is_file():
            raise RuntimeError("Screen recording failed to finalize")
        chapters = [";FFMETADATA1", "title=Heart Disease MLOps acceptance execution"]
        records = read_json(output / "results.json", {}).get("cases", [])
        for item in records:
            if "start_seconds" not in item:
                continue
            start = max(0, int((item["start_seconds"] - result["capture_start_seconds"]) * 1000))
            end = start + max(1, int(item.get("seconds", 1) * 1000))
            title = (
                (item["id"] + " " + item["purpose"])
                .replace("=", " ")
                .replace(";", " ")
                .replace("#", " ")
            )
            chapters += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start}",
                f"END={end}",
                f"title={title}",
            ]
        (video / "chapters.ffmeta").write_text("\n".join(chapters) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video / "full-execution.mp4"),
                "-i",
                str(video / "chapters.ffmeta"),
                "-map_metadata",
                "1",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(video / "final-demonstration.mp4"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError("Demonstration video chapter assembly failed")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if context:
            try:
                context.close()
            except Exception as error:
                result["context_cleanup_error"] = str(error)
        if browser:
            try:
                browser.close()
            except Exception as error:
                result["browser_cleanup_error"] = str(error)
        if recorder and recorder.poll() is None:
            try:
                recorder.communicate(input=b"q\n", timeout=45)
            except (subprocess.TimeoutExpired, OSError) as error:
                recorder.kill()
                recorder.wait(timeout=10)
                result.update(status="failed", recorder_cleanup_error=str(error))
        server.shutdown()
        log.close()
        write_json(output / "media-status.json", result)
    return result["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if capture(args.output) else 1)


if __name__ == "__main__":
    main()
