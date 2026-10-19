from pathlib import Path

# 1. Project root - Points to C:\Users\harik\ml.project
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Data directory - Used by model_train.py to find the CSV
DATA_DIR = BASE_DIR / "ml_model" / "data" / "raw"

# 3. Model artifacts - Where the .joblib files are stored
MODEL_DIR = BASE_DIR / "ml_model" / "saved_model" 
MODEL_PATH = MODEL_DIR / "final_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

# 4. Reports directory - For evaluation results
REPORTS_DIR = BASE_DIR / "ml_model" / "reports"

# 5. Expected input features (X)
# CRITICAL: We include "math_score" as an input.
# CRITICAL: We EXCLUDE "writing_score" because it is the target (y) we want to predict.
FEATURE_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "math_score",
    "reading_score",
]

# Ensure directories exist so the code doesn't crash when saving
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def check_models_exist():
    """Returns True if the trained model files are found on disk."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return False
    return True