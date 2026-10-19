"""
Core training logic for Student Performance Prediction
Algorithm: Linear Regression
"""
import pandas as pd
import joblib
import os
from pathlib import Path

# Only import what you need from config
from config import MODEL_PATH, PREPROCESSOR_PATH, BASE_DIR

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model():
    try:
        print("Training started (Linear Regression)...")
        
        # 1. Dataset path (using DATA_DIR from config)
        # Instead of DATA_DIR / "StudentsPerformance.csv"
        dataset_path = BASE_DIR / "ml_model" / "data" / "raw" / "StudentsPerformance.csv"
        
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        # 2. Load and Clean Dataset
        df = pd.read_csv(dataset_path)
        
        # Normalize column names (Lower case, underscores instead of spaces)
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("/", "_")
        )
        print(f"Dataset loaded successfully with shape: {df.shape}")

        # 3. Define target & features
        TARGET_COLUMN = "writing_score" # The thing we want to predict
        
        categorical_features = [
            "gender", "race_ethnicity", "parental_level_of_education",
            "lunch", "test_preparation_course"
        ]
        numerical_features = ["math_score", "reading_score"]
        
        FEATURES = categorical_features + numerical_features
        
        X = df[FEATURES]
        y = df[TARGET_COLUMN]

        # 4. Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        # 5. Model & Pipeline creation
        model = LinearRegression()
        
        # We wrap the preprocessor AND the model into one Pipeline object
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print("Training Linear Regression pipeline...")
        clf.fit(X, y)

        # 6. Save artifacts
        # We save the WHOLE pipeline (clf) so predict.py is easier to write
        os.makedirs(MODEL_PATH.parent, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        
        print(f"Model saved to: {MODEL_PATH}")
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()