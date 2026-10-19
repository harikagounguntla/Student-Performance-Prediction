import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from report_utils import save_metrics, save_evaluation_summary
from generate_reports import generate_reports
from pathlib import Path
import sys

# --- Setup paths and imports ---
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
sys.path.append(str(BASE_DIR / 'src'))
ARTIFACTS_DIR = BASE_DIR / "ml_model" / "artifacts"
report_dir = BASE_DIR / "ml_model" / "reports"
def load_datasets():
    """Loads processed training and testing datasets"""
    # 1. Setup absolute paths using BASE_DIR (Option 3)
    # Based on your structure: ml.project > ml_model > data > processed
    train_path = BASE_DIR / "ml_model" / "data" / "processed" / "train_processed.csv"
    test_path = BASE_DIR / "ml_model" / "data" / "processed" / "test_processed.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f" Training file not found: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f" Test file not found: {test_path}")

    print(f" Loading training data from: {train_path}")
    print(f" Loading testing data from: {test_path}")
    try:
        # 2. Load the CSV files
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded {len(train_df)} training rows and {len(test_df)} testing rows.")
        print(f" Train data loaded: {train_df.shape}")
        print(f" Test data loaded: {test_df.shape}")
        return train_df, test_df

    except Exception as e:
        print(f"Error loading CSV files: {e}")
        raise
# =========================
# MAIN TRAINING PIPELINE
# =========================
def train_model():
    """Contains the actual logic for loading data and training."""
    print("Starting Student Performance Training...")
    # 1. Load dataset
    train_df, test_df = load_datasets()
    # 2. Split features and target
    #clean cloumns
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    print("Columns cleaned. Available columns:", train_df.columns.tolist())
    TARGET_COLUMN = "writing score"
    if TARGET_COLUMN not in train_df.columns:
        print(f" Error: {TARGET_COLUMN} not found! Check spelling in CSV.")
        return
    # Columns causing data leakage
    DROP_COLUMNS = ["total_score", "average_score", "high_performer"]

    # Split train data (REMOVE leakage columns)
    X_train = train_df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors='ignore')
    y_train = train_df[TARGET_COLUMN]

    # Split test data (REMOVE leakage columns)
    X_test = test_df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors='ignore')
    y_test = test_df[TARGET_COLUMN]

    # 3. Identify column types
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numerical_features = [
        col for col in numerical_features 
        if col not in DROP_COLUMNS and col != TARGET_COLUMN
    ]
    print("✅ Final Features Used:", X_train.columns.tolist())
    print(" Categorical Features:", categorical_features)
    
    # 4. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    # 5. Model
    model = LinearRegression()
    # 6. Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    # 8. Train model
    print("Training model pipeline (Encoding + Scaling + Regression)")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    # 9. Predictions
    y_pred = pipeline.predict(X_test)
    # 10. Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }
    print("\n MODEL PERFORMANCE:")
    print(f"   RMSE: {metrics['rmse']}")
    print(f"   MAE:  {metrics['mae']}")
    print(f"   R²:   {metrics['r2']}")
    # =========================
    # SAVE ARTIFACTS
    # =========================
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(pipeline, f"{ARTIFACTS_DIR}/student_model.joblib")
    print("Model saved")
    # =========================
    # REPORTS
    # =========================
    os.makedirs(report_dir, exist_ok=True)
    save_evaluation_summary(y_test, y_pred, report_dir)
    save_metrics(y_test, y_pred, report_dir)
    generate_reports(y_test, y_pred, report_dir=report_dir)
    

    print(" Reports generated successfully")
    print(" Training pipeline completed")

def main():
    """The entry point that controls the high-level flow."""
    print(" Starting Linear Regression Training Pipeline")
    train_model()
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()