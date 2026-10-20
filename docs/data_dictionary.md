# Data dictionary: UCI processed Cleveland

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease),
DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
Janosi, Steinbrunn, Pfisterer, and Detrano (1989); CC BY 4.0.
Keep the original codes: other redistributed heart datasets sometimes recode them.

| Field | Meaning | Treatment |
|---|---|---|
| age | Age in years | Continuous; median imputation, scaling |
| sex | Recorded sex: 0 female, 1 male | Categorical; mode imputation, one-hot encoding |
| cp | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic | Categorical |
| trestbps | Resting blood pressure, mm Hg, on admission | Continuous |
| chol | Serum cholesterol, mg/dl | Continuous |
| fbs | Fasting blood sugar >120 mg/dl: 0 false, 1 true | Categorical |
| restecg | Resting ECG: 0 normal, 1 ST-T abnormality, 2 probable/definite LVH | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise-induced angina: 0 no, 1 yes | Categorical |
| oldpeak | ST depression induced by exercise relative to rest | Continuous |
| slope | Peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping | Categorical |
| ca | Number of major vessels colored by fluoroscopy: 0-3 | Categorical for the initial baseline |
| thal | 3 normal, 6 fixed defect, 7 reversible defect | Categorical; do not treat codes as measurements |
| num | Original diagnosis code: 0-4 | Source label only; never a predictor |
| target | Derived label: 0 when num=0; 1 when num>0 | Binary classification target |

There are **13 predictors**, not 14 predictors. The original file contains 14
columns including `num`; the cleaned file replaces `num` with `target`.
Missing `?` values become nulls. Observations are retained. Learned imputation,
encoding, and scaling must be fitted on training data only, within CV folds when
evaluating models. The EDA preprocessing export is a demonstration, not a
cross-validation-ready training matrix.
