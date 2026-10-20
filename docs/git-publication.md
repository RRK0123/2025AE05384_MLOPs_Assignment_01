# Publish the repository yourself

Codex does not initialize, stage, commit, push, create a remote repository or
register a self-hosted runner. You perform the commands below.

Prefer the clean full repository ZIP to the old serving-only Windows kit. Extract
it into a fresh directory. Its `REPOSITORY_FILES.json` lists source hashes; it has
no old virtual environments, MLflow databases or unrelated historical outputs.

Create an **empty public GitHub repository** with Issues enabled. Do not initialize
it with another README. Replace the path and URL below with your own values.

```powershell
Set-Location 'C:\YOUR\PATH\heart-disease-mlops'
git init -b main
git status --short
git add .
git diff --cached --stat
git commit -m "Add reproducible MLOps pipeline and independent acceptance tracks"
git remote add origin https://github.com/YOUR-USER/YOUR-REPOSITORY.git
git push -u origin main
```

The push triggers CI and, after quality checks pass, independent deployment into
a temporary Kubernetes cluster on GitHub. Pull requests run quality checks without
publishing images or issues. Actions can also be started from `workflow_dispatch`.
GitHub's hosted runner does not connect to your friend's PC.

Under **Actions**, open **Independent GitHub CI and CD**. Download the quality
artifact and the complete GitHub evidence artifact. The latter contains its own
reports, statistical evidence, screenshots, video and diagnostic ZIP. Bootstrap
logs are retained even if environment installation fails.

Trusted successful runs publish the tested image to GHCR and create/update the
DEMO retraining issue. If a repository/organization policy blocks the built-in
token's package or issue permission, the corresponding workflow step fails with
evidence; it is not reported as successful. New GHCR packages may initially be
private; change package visibility if the evaluator needs direct pull access.
The Kubernetes demonstration itself loads the exact locally built image, so it
does not require registry credentials or depend on an earlier published image.

Windows execution is separate: clone/download this revision on the capable PC,
prepare prerequisites, and double-click `RUN_ALL.bat`. Its evidence is generated
independently, even if GitHub has never executed the workflow.
