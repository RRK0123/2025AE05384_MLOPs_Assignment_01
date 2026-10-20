# Acceptance suite implementation

The two execution tracks are independent: Windows uses Minikube; GitHub Actions
uses a temporary kind cluster. Neither consumes the other track's run results.
No commit, push, remote creation, or runner registration is performed by Codex.

Implemented:

1. Per-test methods/assertions, fixture definitions, timestamps and branch coverage.
2. Labeled endpoint metrics/intervals, load statistics, drift and retraining requests.
3. Coordinator, isolated training workspaces and Minikube/kind adapters.
4. Real browser screenshots, actual desktop recording and chaptered final MP4.
5. Detailed HTML/PDF/JUnit and a verified ten-page assignment PDF.
6. One-click Windows bootstrap, scoped cleanup/diagnostics and independent GitHub
   CI/CD with image publication and a deduplicated DEMO retraining issue.
7. Clean repository export with a source inventory and student-run Git commands.

Development verification on the original Windows PC included a fresh native
pipeline with 265 unit tests and 10 integration tests passing, 102 verified
MLflow runs, matching rebuilt bundle bytes, an isolated inference installation,
real HTTP evaluation/monitoring, and continuous/chaptered videos. Later focused
regressions cover the final reporting, workflow, bootstrap and file-sharing changes.
Real browser capture checks verify MLflow's actual training rows, metrics,
parameters, artifacts and both final models rather than loading placeholders.

The native rehearsal explicitly reports PARTIAL: Docker/Kubernetes are untested
on this PC. The new complete Windows workflow and the hosted GitHub workflow must
execute on their intended targets to produce full deployment proof. The source
package contains automation, not a claim that those target executions passed.

All generated evidence belongs to its current run. Historical reports are never
copied into a fresh acceptance workspace. Controlled degradation is always labeled
DEMO and never changes the reference labels or the deployed model artifact.
