"""Produce traceable HTML, JUnit and ten-page assignment evidence from this run."""

import csv
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts.acceptance.evidence import read_json, write_json

REQUIREMENTS = {
    "R1": ("Data acquisition, cleaning and EDA", 5),
    "R2": ("Feature engineering, tuning and model evaluation", 8),
    "R3": ("MLflow experiment tracking", 5),
    "R4": ("Packaging and reproducibility", 7),
    "R5": ("CI and automated tests", 8),
    "R6": ("Docker prediction API", 5),
    "R7": ("Kubernetes deployment and exposure", 7),
    "R8": ("Logging and monitoring", 3),
    "R9": ("Documentation and delivery evidence", 2),
}
STYLE = """body{font:16px/1.5 system-ui;color:#203342;max-width:1200px;margin:32px auto;padding:0 24px}
h1,h2{color:#125b64}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #cdd9dd;padding:9px;text-align:left;vertical-align:top}
th{background:#edf4f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:12px;font-size:13px}
details{margin:12px 0;border:1px solid #cdd9dd;padding:12px}summary{cursor:pointer;font-weight:600}
.passed{color:#09664c}.failed,.blocked{color:#a62a27}.skipped,.partial{color:#855b00}
img{max-width:100%}a{color:#086d8f}.note{background:#edf4f5;padding:16px} @media print{details{break-inside:avoid}}"""


def esc(value):
    return html.escape(str(value))


def combined_cases(output, result):
    cases = [{"kind": "acceptance_check", **case} for case in result.get("cases", [])]
    for path in sorted((output / "ci").glob("*-details.json")):
        for case in read_json(path, []):
            case["kind"] = "python_test"
            if case["status"] == "not_run" and result.get("status") != "running":
                case.update(
                    status="blocked", actual="Execution ended before this collected test could run."
                )
            cases.append(case)
    # The CI commands remain individually visible, including pipeline validators.
    for step in read_json(output / "ci/steps.json", []):
        from scripts.acceptance.pytest_evidence import requirements_for

        cases.append(
            {
                "id": "PIPELINE-" + step["name"],
                "kind": "pipeline_command",
                "purpose": step["name"],
                "requirements": requirements_for(step["name"]),
                "method": "Run the recorded subprocess against the fresh workspace.",
                "inputs": {
                    "command": step["command"],
                    "cwd": step.get("cwd"),
                    "environment": step.get("environment", {}),
                },
                "started_at": step.get("started_at"),
                "expected": "Exit code 0 and stage-specific validator assertions pass.",
                "actual": step.get(
                    "error", f"Exit code {step.get('returncode')}; see complete log."
                ),
                "status": step["status"],
                "seconds": step.get("seconds", 0),
                "evidence": [f"ci/{step['name']}.log"],
            }
        )
    for case in cases:
        if "start_seconds" not in case and case.get("started_at") and result.get("started_at"):
            case["start_seconds"] = max(
                0,
                (
                    datetime.fromisoformat(case["started_at"])
                    - datetime.fromisoformat(result["started_at"])
                ).total_seconds(),
            )
    return cases


def requirement_matrix(cases):
    rows = []
    for identifier, (title, marks) in REQUIREMENTS.items():
        matching = [case for case in cases if identifier in case.get("requirements", [])]
        outcomes = Counter(case["status"] for case in matching)
        status = (
            "failed"
            if outcomes["failed"]
            else "blocked"
            if outcomes["blocked"]
            else "covered"
            if outcomes["passed"]
            else "not_executed"
        )
        mandatory = {"R6": "DOCKER-001", "R7": "K8S-001"}.get(identifier)
        if mandatory:
            execution = next((case for case in cases if case["id"] == mandatory), None)
            if execution is None or execution["status"] == "skipped":
                status = "not_executed"
            elif execution["status"] != "passed":
                status = execution["status"]
        rows.append(
            {
                "requirement": identifier,
                "title": title,
                "rubric_marks": marks,
                "status": status,
                "passed": outcomes["passed"],
                "failed": outcomes["failed"],
                "skipped": outcomes["skipped"],
                "blocked": outcomes["blocked"],
                "test_ids": "; ".join(case["id"] for case in matching),
            }
        )
    return rows


def generate(output):
    output = Path(output).resolve()
    result = read_json(output / "results.json", {})
    cases = combined_cases(output, result)
    report = output / "report"
    report.mkdir(exist_ok=True)
    statistics = read_json(output / "statistics.json", {})
    coverage = read_json(output / "coverage/coverage.json", {})
    matrix = requirement_matrix(cases)
    write_json(report / "test-cases.json", cases)
    write_json(report / "requirement-coverage.json", matrix)
    with (report / "requirement-coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    counts = dict(Counter(case["status"] for case in cases))
    levels = {
        kind: dict(Counter(case["status"] for case in cases if case["kind"] == kind))
        for kind in ["python_test", "acceptance_check", "pipeline_command"]
    }
    links = [
        "report/test-report.html",
        "report/detailed-test-report.pdf",
        "report/assignment-submission.pdf",
        "coverage/html/index.html",
        "video/final-demonstration.mp4",
        "video/full-execution.mp4",
        "statistics.json",
        "monitoring/decisions.json",
        "retraining/request.json",
        "services.json",
        "evidence-manifest.json",
    ]
    landing = f"<!doctype html><meta charset='utf-8'><title>Run {esc(result.get('run_id'))}</title><style>{STYLE}</style>"
    landing += f"<h1>Heart Disease MLOps acceptance evidence</h1><p>Run: {esc(result.get('run_id'))} | Track: {esc(result.get('track'))}</p>"
    landing += f"<h2 class='{esc(result.get('status'))}'>{esc(result.get('status', 'unknown')).upper()}</h2><p>{esc(result.get('error', ''))}</p>"
    landing += (
        "<ul>"
        + "".join(
            f"<li><a href='{link}'>{esc(link)}</a></li>"
            for link in links
            if (output / link).exists()
        )
        + "</ul>"
    )
    services = read_json(output / "services.json", {})
    if (
        result.get("track") == "windows"
        and result.get("status") == "passed"
        and result.get("complete_acceptance")
    ):
        api = services.get("deployed_api")
        landing += "<h2>Live demonstration on this PC</h2><ul>"
        for title, url in [
            ("MLflow experiment tracking", services.get("mlflow")),
            ("Prediction API / Swagger", api + "/docs" if api else None),
            ("API monitoring", api + "/monitoring" if api else None),
        ]:
            if url:
                landing += (
                    f"<li><a target='_blank' rel='noopener' href='{esc(url)}'>{esc(title)}</a></li>"
                )
        landing += "</ul>"
    landing += "<p>Temporary GitHub clusters stop after evidence collection. Windows demo services remain available until STOP_DEMO is run.</p>"
    (output / "START_HERE.html").write_text(landing, encoding="utf-8")
    content = f"<!doctype html><meta charset='utf-8'><title>Detailed test report</title><style>{STYLE}</style>"
    content += f"<h1>Detailed acceptance test report</h1><p>Run {esc(result.get('run_id'))}, track {esc(result.get('track'))}, status {esc(result.get('status'))}.</p>"
    content += "<p class='note'>Tests, model performance and requirement coverage are different measures. Controlled degradation is DEMO evidence. Synthetic inputs do not establish clinical accuracy. Missing student details remain explicitly marked.</p>"
    content += (
        "<details><summary>Environment, configuration and source identities</summary><pre>"
        + esc(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
        + "</pre></details>"
    )
    content += (
        "<h2>Outcome totals by check type</h2><pre>" + esc(json.dumps(levels, indent=2)) + "</pre>"
    )
    content += "<p>Acceptance gates and command checks are reported separately from individual Python tests. Model evaluation sample counts are separate again.</p>"
    content += "<h2>Requirement coverage</h2><table><tr><th>ID</th><th>Requirement</th><th>Status</th><th>Pass / fail / skip / blocked</th></tr>"
    for row in matrix:
        content += f"<tr><td>{row['requirement']}</td><td>{esc(row['title'])}</td><td>{row['status']}</td><td>{row['passed']} / {row['failed']} / {row['skipped']} / {row['blocked']}</td></tr>"
    content += "</table><p>A covered requirement has executed evidence; this is not an automatic grading claim. Skipped cases remain listed below.</p>"
    content += "<h2>Code coverage</h2><p>Instrumented native Python tests and CLI stages; production containers and isolated wheel/API processes are black-box checks. No coverage percentage is inferred for uninstrumented code.</p>"
    content += (
        "<pre>"
        + esc(json.dumps(coverage.get("totals", {"status": "not collected"}), indent=2))
        + "</pre>"
    )
    if (output / "coverage/html/index.html").exists():
        content += "<a href='../coverage/html/index.html'>Open line and branch details</a>"
    content += (
        "<details><summary>Model, data and deployment statistics</summary><pre>"
        + esc(json.dumps({k: v for k, v in statistics.items() if k != "tables"}, indent=2))
        + "</pre></details>"
    )
    content += "<h2>Test cases</h2>"
    suite = ET.Element(
        "testsuite",
        name="acceptance",
        tests=str(len(cases)),
        failures=str(counts.get("failed", 0)),
        skipped=str(counts.get("skipped", 0) + counts.get("blocked", 0)),
    )
    for case in cases:
        content += f"<details id='{esc(case['id'])}'><summary class='{esc(case['status'])}'>{esc(case['id'])} - {esc(case['status'])} ({case.get('seconds', 0):.3f}s)</summary>"
        content += "<pre>" + esc(json.dumps(case, indent=2)) + "</pre>"
        for evidence in case.get("evidence", []):
            if (output / evidence).is_file():
                content += f"<a href='../{esc(evidence)}'>{esc(evidence)}</a><br>"
        if "start_seconds" in case:
            offset = read_json(output / "media-status.json", {}).get("capture_start_seconds", 0)
            video_seconds = max(0, case["start_seconds"] - offset)
            content += f"<p><a href='../video/final-demonstration.mp4#t={int(video_seconds)}'>Video at {video_seconds:.1f} seconds</a></p>"
        content += "</details>"
        item = ET.SubElement(
            suite,
            "testcase",
            name=case["id"],
            classname="acceptance",
            time=str(case.get("seconds", 0)),
        )
        if case["status"] == "failed":
            ET.SubElement(item, "failure", message=str(case.get("actual", "Failure"))).text = str(
                case.get("actual", "")
            )
        elif case["status"] in {"skipped", "blocked", "not_run"}:
            ET.SubElement(item, "skipped", message=str(case.get("actual", case["status"])))
    ET.ElementTree(suite).write(report / "junit.xml", encoding="utf-8", xml_declaration=True)
    content += "<h2>Screenshots</h2>"
    for image in sorted((output / "screenshots").glob("*.png")):
        content += f"<details><summary>{esc(image.stem)}</summary><img src='../screenshots/{esc(image.name)}'></details>"
    (report / "test-report.html").write_text(content, encoding="utf-8")
    generate_pdfs(output, result, cases, statistics, matrix, coverage)
    write_json(
        report / "summary.json",
        {
            "counts": counts,
            "counts_by_type": levels,
            "requirements": matrix,
            "code_coverage": coverage.get("totals"),
            "assignment_pages": 10,
            "status": result.get("status"),
        },
    )
    return counts


def generate_pdfs(output, result, cases, statistics, matrix, coverage):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "SmallText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )

    def paragraph(text, style="BodyText"):
        return Paragraph(esc(text).replace("\n", "<br/>"), styles[style])

    def table(rows, widths=None):
        value = Table(
            [[paragraph(str(cell), "SmallText") for cell in row] for row in rows],
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        value.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f1f3")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bccbd2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return value

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            42, 25, f"{result.get('run_id')} | {result.get('track')} | {result.get('status')}"
        )
        canvas.drawRightString(550, 25, f"Page {document.page}")

    def document(path, story):
        SimpleDocTemplate(
            str(path),
            pagesize=(595, 842),
            rightMargin=42,
            leftMargin=42,
            topMargin=40,
            bottomMargin=45,
        ).build(story, onFirstPage=footer, onLaterPages=footer)

    detailed = [
        paragraph("Detailed automated test report", "Title"),
        paragraph(f"Run {result.get('run_id')} - {result.get('status')}"),
        paragraph(
            "This report records actual test methods, inputs, assertions, fixture definitions, outcomes and evidence references. The HTML edition provides clickable links; JSON preserves machine-readable detail."
        ),
    ]
    detailed += [paragraph("Execution environment and configuration", "Heading1")]
    environment = json.dumps(
        {
            k: result.get(k)
            for k in [
                "run_id",
                "track",
                "status",
                "started_at",
                "finished_at",
                "python",
                "platform",
                "commit",
                "config",
                "model_run_id",
                "model_sha256",
                "bundle_sha256",
                "complete_acceptance",
            ]
        },
        indent=2,
    )
    for start in range(0, len(environment), 1800):
        detailed.append(paragraph(environment[start : start + 1800], "SmallText"))
    detailed += [
        paragraph("Requirement coverage", "Heading1"),
        table(
            [
                ["ID", "Requirement", "Status"],
                *[[r["requirement"], r["title"], r["status"]] for r in matrix],
            ],
            [35, 365, 110],
        ),
        paragraph("Code coverage", "Heading1"),
        paragraph(
            json.dumps(coverage.get("totals", {"status": "Not collected"}), indent=2), "SmallText"
        ),
        paragraph(
            "Instrumented native tests, CLI stages and acceptance coordinator; production containers and isolated inference remain black-box. See coverage/html for every missing line and branch.",
            "SmallText",
        ),
        paragraph("Individual test cases", "Heading1"),
    ]
    for case in cases:
        detailed += [
            Spacer(1, 10),
            paragraph(case["id"], "Heading3"),
            paragraph(
                f"Status: {case['status']} | Duration: {case.get('seconds', 0):.3f}s | Requirements: {', '.join(case.get('requirements', []))}",
                "SmallText",
            ),
        ]
        for field in [
            "purpose",
            "method",
            "inputs",
            "fixtures",
            "fixture_definitions",
            "source",
            "source_line",
            "expected",
            "actual",
            "evidence",
            "test_source",
        ]:
            if field not in case and field not in {
                "purpose",
                "method",
                "inputs",
                "expected",
                "actual",
                "evidence",
            }:
                continue
            value = case.get(field, "See test source and fixture references in HTML")
            value = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
            # Long assertion lists are divided to keep ReportLab flowables splittable.
            detailed.append(paragraph(field.upper(), "SmallText"))
            for start in range(0, len(value), 1800):
                detailed.append(paragraph(value[start : start + 1800], "SmallText"))
    document(output / "report/detailed-test-report.pdf", detailed)

    metadata = statistics.get("metadata", {})
    evaluation = statistics.get("deployed_evaluation", statistics.get("native_evaluation", {}))
    student = result.get("config", {}).get("submission", {})
    quality = statistics.get("data_quality", {})
    data = statistics.get("data", {})
    data_rows = [
        ["Statistic", "Value"],
        ["Source records", data.get("records", "Not executed")],
        ["Raw predictors", data.get("input_feature_count", "Not executed")],
        ["Class counts (0 / 1)", str(data.get("class_counts", {}))],
        ["Development / test rows", f"{metadata.get('train_rows')} / {metadata.get('test_rows')}"],
        [
            "Missing cells / duplicate rows",
            f"{quality.get('missing_cells')} / {quality.get('duplicate_rows')}",
        ],
        [
            "Missing values by feature",
            str({k: v for k, v in data.get("missing_by_feature", {}).items() if v}),
        ],
    ]
    comparison = [["Family", "Outer ROC-AUC mean", "Outer ROC-AUC SD"]]
    for row in statistics.get("tables", {}).get("model_comparison.csv", []):
        comparison.append(
            [row["family"], f"{row['roc_auc_mean']:.4f}", f"{row['roc_auc_std']:.4f}"]
        )
    tracking = statistics.get("tracking", {})
    tracking_rows = [
        ["Tracked evidence", "Result"],
        ["Experiment", tracking.get("experiment_name", "Not executed")],
        ["Total finished runs", tracking.get("run_count", "Not executed")],
    ]
    tracking_rows += [[phase, count] for phase, count in tracking.get("phase_counts", {}).items()]
    tracking_rows += [
        [
            "All runs have parameters, metrics, artifacts and plots",
            tracking.get("all_runs_have_parameters_metrics_artifacts_plots"),
        ],
        [
            "CV predictions/reports reconciled",
            tracking.get("outer_cv_predictions_and_reports_reconciled"),
        ],
        [
            "Both final models reload identically",
            tracking.get("both_final_models_logged_and_roundtrip_verified"),
        ],
    ]
    from reportlab.graphics.shapes import Drawing, Line, Rect, String

    architecture = Drawing(510, 190)
    nodes = [
        "Raw data + EDA",
        "Nested CV + tuning",
        "MLflow + selection",
        "Verified model bundle",
        "Docker image + tests",
        "Kubernetes exposure",
        "Labeled HTTP testing",
        "Monitoring decisions",
        "Retraining request",
    ]
    positions = [
        ((index % 3 if (index // 3) % 2 == 0 else 2 - index % 3) * 172, 140 - (index // 3) * 65)
        for index in range(len(nodes))
    ]
    for (px, py), (x, y) in zip(positions, positions[1:]):
        architecture.add(
            Line(px + 81, py + 20, x + 81, y + 20, strokeColor=colors.HexColor("#125b64"))
        )
    for index, (text, (x, y)) in enumerate(zip(nodes, positions)):
        architecture.add(
            Rect(
                x,
                y,
                162,
                40,
                fillColor=colors.HexColor("#e9f1f3"),
                strokeColor=colors.HexColor("#125b64"),
            )
        )
        architecture.add(
            String(x + 81, y + 16, f"{index + 1}. {text}", textAnchor="middle", fontSize=8)
        )
    scores = evaluation.get("metrics", {})
    load = statistics.get("load", {})
    load_rows = [
        ["Operational measurement", "Observed"],
        ["Requests / concurrency", f"{load.get('requests')} / {load.get('concurrency')}"],
        [
            "Throughput (requests/second)",
            load.get("throughput_requests_per_second", "Not executed"),
        ],
        ["Client latency percentiles (ms)", str(load.get("latency_ms", "Not executed"))],
        ["Request error rate", load.get("error_rate", "Not executed")],
    ]
    request = read_json(output / "retraining/request.json", {})
    trigger_rows = [["DEMO metric", "Reference", "Observed", "Drop"]]
    for trigger in request.get("triggers", []):
        trigger_rows.append(
            [
                trigger["metric"],
                f"{trigger['baseline']:.4f}",
                f"{trigger['observed']:.4f}",
                f"{trigger['drop']:.4f}",
            ]
        )
    intervals = evaluation.get("confidence_intervals", {}).get("intervals", {})
    score_rows = [["Metric", "Observed", "95% bootstrap interval"]]
    for name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc_average_precision",
        "brier_score",
    ]:
        value = scores.get(name)
        bounds = intervals.get(name)
        score_rows.append(
            [
                name,
                f"{value:.4f}" if value is not None else "Not available",
                f"{bounds[0]:.4f} - {bounds[1]:.4f}" if bounds else "Not available",
            ]
        )
    sections = [
        (
            "1. Assignment and execution identity",
            [
                paragraph("Heart Disease MLOps", "Title"),
                table(
                    [["Field", "Recorded value"], *[[k, v] for k, v in student.items()]], [150, 360]
                ),
                paragraph(
                    f"Track: {result.get('track')}. Run: {result.get('run_id')}. Commit: {result.get('commit') or 'Not available in this checkout'}. Result: {result.get('status')}.",
                    "SmallText",
                ),
                paragraph(
                    "This document summarizes this execution only. Its detailed technical report, machine-readable statistics and evidence manifest are supplied separately. Unsupplied student details must be completed before submission."
                ),
            ],
        ),
        (
            "2. Architecture and reproducibility",
            [
                paragraph(
                    "Attributed raw data -> training-only EDA -> fold-fitted preprocessing -> nested model comparison -> MLflow -> verified model bundle -> Docker -> Kubernetes LoadBalancer -> labeled endpoint evaluation -> monitoring -> retraining request."
                ),
                architecture,
                paragraph(
                    "The Windows track uses its own Minikube cluster. The GitHub track independently trains and deploys into a temporary kind cluster. Model, source, data and image identities are recorded; neither track consumes the other track's evidence."
                ),
                paragraph(
                    "Pinned dependencies and reusable CLI stages support reproduction. The serving pipeline includes its fitted imputation, encoding and classifier. Containers perform inference without retraining or requiring MLflow."
                ),
                paragraph(
                    json.dumps(
                        {
                            k: result.get(k)
                            for k in [
                                "model_run_id",
                                "model_sha256",
                                "bundle_sha256",
                                "deployed_image",
                                "cluster",
                            ]
                        },
                        indent=2,
                    ),
                    "SmallText",
                ),
            ],
        ),
        (
            "3. Dataset and exploratory analysis",
            [
                paragraph(
                    "UCI processed Cleveland: the source contains 13 predictors plus diagnosis num. The binary target is num > 0. Diagnosis and target are excluded from inputs. The assignment's 14+ features wording differs from the 13 raw predictors; one-hot encoding does not create additional raw measurements."
                ),
                table(data_rows, [240, 270]),
                paragraph("Predictors: " + ", ".join(data.get("predictors", [])), "SmallText"),
                paragraph(
                    "Full-source counts are integrity audits. Feature distributions and relationships are analyzed on development data; missing values are retained until training-fitted imputation. Outliers are flagged rather than removed."
                ),
                paragraph(
                    "Source: Janosi et al., Heart Disease, UCI Machine Learning Repository (1989), DOI 10.24432/C52P4X; CC BY 4.0.",
                    "SmallText",
                ),
            ],
        ),
        (
            "4. Features, training and model selection",
            [
                paragraph(
                    "All 13 original predictors are retained. Five continuous features use median imputation; eight coded features use mode imputation and one-hot encoding. Logistic Regression additionally scales continuous inputs. There are 28 transformed features."
                ),
                paragraph(
                    "A majority baseline, Logistic Regression and Random Forest are compared using five stratified outer folds. Inner three-fold tuning stays inside each outer training partition. Each family is retuned on development data using five folds. The winner is chosen by mean outer ROC-AUC before reserved-test evaluation."
                ),
                paragraph(
                    f"Selected family: {metadata.get('selected_family', 'Not executed')}. Development rows: {metadata.get('train_rows')}. Test rows: {metadata.get('test_rows')}. Seed: {metadata.get('seed')}.",
                    "SmallText",
                ),
                table(comparison, [220, 145, 145]),
                paragraph(
                    "Small score differences do not establish statistical superiority. The selected parameters and every search candidate are retained in the detailed statistical evidence.",
                    "SmallText",
                ),
                paragraph(
                    "Selected parameters: "
                    + str(metadata.get("selected_parameters", "Not executed")),
                    "SmallText",
                ),
            ],
        ),
        (
            "5. MLflow experiment tracking",
            [
                paragraph(
                    "Each completed training session records its parent, outer validation folds, hyperparameter candidates and both final family models. Parameters, fold metric histories, search tables, plots and native model exports are reconciled with training evidence."
                ),
                table(tracking_rows, [360, 150]),
                paragraph(
                    "The local database is generated per run. The UI and screenshots demonstrate the actual session; portable tracking tables and validation results accompany the submission. Historical tracking databases are not required to reproduce training."
                ),
            ],
        ),
        (
            "6. Labeled evaluation and uncertainty",
            [
                paragraph(
                    "The deployed endpoint receives original-feature reserved-test records. Responses are compared with known labels and with the saved model. Reported metrics include accuracy, precision, recall, F1, ROC-AUC, average precision and Brier score."
                ),
                table(score_rows, [220, 100, 190]),
                paragraph(
                    f"Records: {scores.get('records', 'Not executed')}; class counts: {scores.get('class_counts')}. Confusion matrix (actual rows, predicted columns): {scores.get('confusion_matrix_actual_rows_predicted_columns')}.",
                    "SmallText",
                ),
                paragraph(
                    f"Bootstrap replications: {evaluation.get('confidence_intervals', {}).get('samples', 'Not executed')}; seed: {evaluation.get('confidence_intervals', {}).get('seed', 'Not executed')}.",
                    "SmallText",
                ),
                paragraph(
                    "Intervals use a seeded stratified percentile bootstrap. The historical sample is small. Model probabilities are uncalibrated estimates of recorded disease presence; these results do not establish clinical suitability or future disease risk.",
                    "SmallText",
                ),
            ],
        ),
        (
            "7. Automated testing and CI/CD",
            [
                paragraph(
                    "Tests exercise input validation, leakage boundaries, model serialization, CV/tuning reconciliation, MLflow records, packaging, HTTP behavior and deployment identity. Expected assertions, actual outcomes and test methods are retained per case."
                ),
                paragraph(
                    json.dumps(dict(Counter(c["status"] for c in cases)), indent=2), "SmallText"
                ),
                paragraph(
                    "GitHub Actions independently installs, tests, trains, builds, deploys to temporary Kubernetes, evaluates and uploads evidence. Source commits and pushes are performed by the student. The Windows track uses the same code in a standalone run."
                ),
                paragraph(
                    json.dumps(
                        coverage.get("totals", {"status": "Coverage unavailable"}), indent=2
                    ),
                    "SmallText",
                ),
                paragraph(
                    "Coverage measures instrumented native Python execution; production containers and isolated inference are black-box checks. Requirement coverage is reported separately and is not a predicted grade.",
                    "SmallText",
                ),
            ],
        ),
        (
            "8. Docker and Kubernetes deployment",
            [
                paragraph(
                    "The current run's verified bundle builds the serving image. Tests check Linux container startup, model identity, inference and non-root runtime. A separately named cluster loads that exact image and applies validated Namespace, Deployment and LoadBalancer manifests."
                ),
                paragraph(
                    "Startup, readiness and liveness probes, resource limits, dropped capabilities and a read-only filesystem are configured. Tests verify ready pods, runtime image IDs, service exposure and predictions through the real LoadBalancer."
                ),
                paragraph(
                    f"Endpoint: {result.get('base_url', 'Not deployed')}. Image: {result.get('deployed_image', 'Not deployed')}.",
                    "SmallText",
                ),
                table(load_rows, [260, 250]),
                paragraph(
                    "Windows services remain available until explicit cleanup. The GitHub cluster is temporary and is deleted after diagnostics. Real execution failures and blocked stages remain visible in the report."
                ),
            ],
        ),
        (
            "9. Monitoring and retraining requests",
            [
                paragraph(
                    "Operational monitoring checks request rates, latency, error statuses and prediction counters. Labeled monitoring compares recall and ROC-AUC with a recorded reference. Numeric KS effect sizes and categorical total variation identify distribution changes."
                ),
                paragraph(
                    json.dumps(
                        read_json(output / "monitoring/summary.json", {"status": "Not executed"}),
                        indent=2,
                    ),
                    "SmallText",
                ),
                table(trigger_rows, [210, 100, 100, 100]),
                paragraph(
                    "Healthy replay must not request retraining. A clearly marked DEMO input-corruption replay preserves labels and the deployed artifact, and verifies the degradation response. Replayed rows do not constitute independent production observations."
                ),
                paragraph(
                    "Requests are stored in evidence. GitHub may create or update a DEMO issue from an authorized workflow. No automatic retraining or model replacement occurs. Infrastructure failures produce operational alerts."
                ),
            ],
        ),
        (
            "10. Requirement evidence and limitations",
            [
                table(
                    [
                        ["ID", "Requirement", "Status"],
                        *[[r["requirement"], r["title"], r["status"]] for r in matrix],
                    ],
                    [35, 365, 110],
                ),
                paragraph(
                    "Deliverables: full repository, dependency pins, automated tests, Docker/Kubernetes configuration, tracking evidence, detailed report, statistics, coverage, screenshots and continuous/chaptered execution videos. Consult START_HERE.html and evidence-manifest.json."
                ),
                paragraph(
                    "Limitations: 303 historical records; small reserved test set; original 13-predictor schema; binary diagnosis mapping; uncalibrated probabilities; controlled monitoring replay rather than longitudinal production validation. Missing execution evidence is reported, never inferred.",
                    "SmallText",
                ),
            ],
        ),
    ]
    story = []
    for index, (title, body) in enumerate(sections):
        if index:
            story.append(PageBreak())
        story += [paragraph(title, "Heading1"), *body]
        # One real execution image on selected pages, sized to preserve ten pages.
        choices = {
            2: "eda",
            3: "model-evaluation",
            4: "mlflow",
            5: "model-02",
            7: "kubernetes",
            8: "monitoring",
        }
        if index in choices:
            candidates = list((output / "screenshots").glob(f"{choices[index]}*.png"))
            if candidates:
                image = Image(str(candidates[0]))
                factor = min(510 / image.imageWidth, 180 / image.imageHeight)
                image.drawWidth, image.drawHeight = (
                    image.imageWidth * factor,
                    image.imageHeight * factor,
                )
                story += [Spacer(1, 0.1 * inch), image]
    document(output / "report/assignment-submission.pdf", story)
    from pypdf import PdfReader

    pages = len(PdfReader(output / "report/assignment-submission.pdf").pages)
    if pages != 10:
        raise ValueError(f"Assignment report overflowed: {pages} pages; expected exactly 10")
