# Execution and evidence architecture

```mermaid
flowchart TD
  Source[Same repository revision and pinned dependencies]
  Source --> W[Windows one-click coordinator]
  Source --> G[GitHub Actions CI/CD]
  W --> WT[Fresh Windows workspace: EDA, training, MLflow, tests]
  G --> GT[Fresh Linux workspace: EDA, training, MLflow, tests]
  WT --> WB[Verified bundle and Docker image]
  GT --> GB[Verified bundle and Docker image]
  WB --> WM[Dedicated Minikube cluster and LoadBalancer]
  GB --> GK[Temporary kind cluster and LoadBalancer]
  WM --> WE[Labeled evaluation, robustness, load, monitoring]
  GK --> GE[Labeled evaluation, robustness, load, monitoring]
  WE --> WR[Local DEMO retraining request]
  GE --> GR[Artifact request and DEMO GitHub issue]
  WE --> WD[Windows reports, statistics, coverage, screenshots, video]
  GE --> GD[GitHub reports, statistics, coverage, screenshots, video]
  GE --> Registry[Publish verified image to GHCR]
```

Each track trains its own model; model identity and hashes follow it into its own
deployment. Neither track consumes the other one's model or reports. Evaluation
uses known labels; synthetic corruption is visibly marked DEMO. Requests require
human review and do not automatically retrain/promote a model. The live operational
API dashboard and the labeled acceptance monitoring report have distinct scopes.
