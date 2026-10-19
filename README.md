"""
Prediction logic for student performance
"""

import pandas as pd
from config import FEATURE_COLUMNS
from model_loader import load_model

class PredictPipeline:
    def __init__(self):
        # Load model and preprocessor ONCE when class is initialized
        self.model, self.preprocessor = load_model()

    def predict(self, input_data_dict):
        """
        Args:
            input_data_dict: Dict from UI (e.g., {"gender": "female", "reading_score": 70...})
        Returns:
            Dict containing the prediction result
        """
        try:
            # ... (Steps 1, 2, and the transformation logic remain the same)
            # 1. Convert dict to DataFrame
            df = pd.DataFrame([input_data_dict])
            # 2. Ensure columns are in the exact order the model expects
            df = df[FEATURE_COLUMNS]
            # 3. Transform and Predict
            data_to_transform = df.astype(object) 
            data_scaled = self.preprocessor.transform(data_to_transform)
            "this is option if you want you can use"
            preds = self.model.predict(data_scaled)
            return {
                "prediction": round(float(preds[0]), 2),
                "status": "success"
            }
            # 4. Return user-friendly result (Update here for cleaner output)
            # Clamp score between 0 and 100
            
            # --------------------------
        except Exception as e:
            return {
                "prediction": None,
                "status": f"Error: {str(e)}"
            }
# --- ADDED THIS FUNCTION ---
# Initialize the pipeline once globally
_pipeline = None

def predict_score(data):
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictPipeline()
    return _pipeline.predict(data)
# This allows for a quick demo check
if __name__ == "__main__":
    # ... (rest of your test code)
    pipeline = PredictPipeline()
    # Sample data matching your FEATURE_COLUMNS
    test_data = {col: 0 for col in FEATURE_COLUMNS} 
    print(pipeline.predict(test_data))
if __name__ == "__main__":
    # Test 1: A "High Performing" Student
    high_student = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "master's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 95,
        "writing_score": 98
    }

    # Test 2: A "Low Performing" Student
    low_student = {
        "gender": "male",
        "race_ethnicity": "group A",
        "parental_level_of_education": "some high school",
        "lunch": "reduced",
        "test_preparation_course": "none",
        "reading_score": 20,
        "writing_score": 20
    }

    print("High Student Prediction:",PredictPipeline(high_student))
    print("Low Student Prediction:", PredictPipeline(low_student))