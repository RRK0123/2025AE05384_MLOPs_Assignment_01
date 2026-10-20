"""Write analysis findings from computed tables."""

import pandas as pd

from .config import SEED
from .schema import DISPLAY, NUMERIC, SOURCE_PAGE, TARGET


def write_findings(raw, clean, train, test, quality, reports):
    counts = quality["class_counts"]
    rates = pd.read_csv(reports / "tables" / "categorical_disease_rates.csv")
    outliers = pd.read_csv(reports / "tables" / "iqr_outlier_audit.csv")
    numeric = train.groupby(TARGET)[NUMERIC].median()
    absent, present = numeric.loc[0], numeric.loc[1]
    missing = {k: v for k, v in quality["missing_by_column"].items() if v}
    chest = rates.loc[rates["feature"] == "cp"]
    chest_lines = "\n".join(
        f"- {row.label}: {row.disease_percent:.1f}% disease present (n={row.n})."
        for row in chest.itertuples()
    )
    outlier_lines = "\n".join(
        f"- `{row.feature}`: {row.flagged_count} flagged observations ({row.flagged_percent:.1f}%)."
        for row in outliers.itertuples()
    )
    text = f"""# Cleveland Heart Disease: EDA findings

## Scope and provenance

This is the data acquisition and EDA stage, not a model evaluation report.
The original UCI processed Cleveland subset contains {len(raw)} rows, 13 predictors,
and one diagnosis column. The assignment's '14+ features' wording differs from
UCI's 13-predictor / 14-column schema. We use the documented source as supplied,
without inventing additional predictors. Source: {SOURCE_PAGE}; DOI: 10.24432/C52P4X.
The dataset is attributed to Janosi, Steinbrunn, Pfisterer, and Detrano (1989),
UCI Machine Learning Repository, under CC BY 4.0.
The raw download and SHA-256 checksum are recorded in `data/raw/source.json`.

## Data quality and cleaning

- Shape: {len(clean)} rows x {len(clean.columns)} cleaned columns (13 predictors + binary target).
- Missing cells in the original source: {quality["missing_cells"]}; affected rows: {quality["rows_with_missing"]}.
- Missing values by column: {missing}.
- Exact duplicate source rows: {quality["duplicate_rows"]}.
- Category-code and measurement validation passed; missing diagnosis labels are rejected.
- Diagnosis `num` is mapped to `target = int(num > 0)` and removed from predictors.
- All source rows are retained. `?` becomes a missing value, not a new category.
- Categorical codes are stored as nullable integers; continuous features remain numeric.
- Cleaned data keeps missingness visible. Imputation is a learned operation performed
  using the training partition only, not a global cleaning operation.

## Class balance and evaluation boundary

There are {counts["0"]} disease-absent and {counts["1"]} disease-present observations
({counts["0"] / len(clean):.1%} versus {counts["1"] / len(clean):.1%}). This is not a severe
imbalance. Resampling is not justified solely by these counts.
An 80/20 stratified split with random seed {SEED} reserves {len(test)} observations
for final testing and {len(train)} for development. Source row positions are saved
in `data/processed/split_manifest.json`. Quality and overall target counts use
the full source; all feature distributions, associations, outlier flags, and
descriptive statistics use training rows only. Do not explore test relationships
or choose settings using test outcomes.

## Continuous measurements: observations, not causal claims

Training medians (disease absent versus disease present):

| Measurement | Absent | Present |
|---|---:|---:|
"""
    for column in NUMERIC:
        text += f"| {DISPLAY[column]} | {absent[column]:.1f} | {present[column]:.1f} |\n"
    text += f"""
Histograms and boxplots show both differences and overlap between groups.
These summaries do not establish a clinically useful prediction rule.
The correlation heatmap uses the five continuous features only. Integer category
codes are not interpreted as continuous measurements. Correlation does not imply
causation, and no predictor is dropped solely because of this exploratory plot.

## Categorical measurements

Training disease frequencies for chest-pain categories:

{chest_lines}

The eight categorical panels include group sizes and descriptive 95% Wilson
intervals. Small groups produce uncertain proportions. Categories with missing
values are excluded from observed group rates; no exploratory imputation is used.
These comparisons describe this sample and are not adjusted for confounding or
multiple comparisons. The sex-category counts describe recorded dataset codes,
not a claim about population prevalence.

## Extreme-value audit

Training observations outside 1.5 x IQR fences:

{outlier_lines}

These are statistical flags, not established data-entry errors. All values are
retained; deleting unusual observations without evidence could remove meaningful
variation. Histograms also help identify skewed measurements.

## Preprocessing demonstration and next-stage decisions

`build_preprocessor()` creates an unfitted scikit-learn transformer with median
imputation and standard scaling for continuous features, and mode imputation
plus one-hot encoding for eight coded features. Categories follow UCI's documented
codebook, so the representation does not depend on learning test categories.
For the EDA demonstration it is fitted on training rows and applied unchanged to
test rows. The 28 transformed columns, missing-value checks, training medians,
and categorical imputation values are recorded in `reports/preprocessing_audit.json`.
The transformed preview files have no diagnosis or target columns. Labels and row
order remain in the matching `train.csv` and `test.csv` files.

When we train models, put a fresh preprocessor inside each model Pipeline and fit
it separately in each cross-validation fold. Do not cross-validate directly on
the already transformed EDA preview: its statistics use the entire training set.
Keep all 13 original predictors initially, use stratified cross-validation,
compare at least two classifiers, and evaluate the reserved test set only after
selection. Scaling can be disabled for tree-based models when appropriate.

## Limitations

This is a small historical Cleveland cohort. Its composition and collection
context do not establish representativeness for current populations. The label
indicates recorded disease presence, not the probability of developing disease
in the future. No model performance or deployment claims are made at this stage.

## Outputs and rerun

Run `python -m heart_disease.download`, then `python -m heart_disease.eda` from
the repository root after installing the package. Run
`python -m heart_disease.validate` to check the generated artifacts.
Figures are in `reports/figures/`, numerical tables are in
`reports/tables/`, and cleaned data plus stable split definitions are in
`data/processed/`. See the README and data dictionary for setup and field meanings.
"""
    (reports / "eda_findings.md").write_text(text, encoding="utf-8")
