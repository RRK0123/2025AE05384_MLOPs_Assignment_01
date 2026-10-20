"""UCI feature definitions and category codes shared by all stages."""

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORIES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "<=120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "Probable/definite LVH"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
FEATURES = RAW_COLUMNS[:-1]
TARGET = "target"
LABELS = {0: "Disease absent", 1: "Disease present"}
DISPLAY = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Maximum heart rate achieved",
    "oldpeak": "Exercise-induced ST depression",
}
