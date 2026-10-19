"""
Prediction logic for student performance.
Handles data transformation and model inference via Pipeline.
"""
import pandas as pd
try:
    from .config import FEATURE_COLUMNS
    from .model_loader import load_model
except ImportError:
    from config import FEATURE_COLUMNS
    from model_loader import load_model

class PredictPipeline:
    def __init__(self):
        # Load the saved Pipeline (which includes both preprocessor + model)
        self.model, _ = load_model()

    def predict(self, input_data_dict):
        try:
            # 1. Convert dict to DataFrame
            df = pd.DataFrame([input_data_dict])
            
            # 2. Ensure columns are in the exact order the model expects
            df = df[FEATURE_COLUMNS]
            
            # 3. FIX: Predict directly on the RAW DataFrame.
            # The Pipeline handles the 'StandardScaler' and 'OneHotEncoder' automatically.
            prediction = self.model.predict(df)[0]
            
            # 4. Return a clean, rounded dictionary
            return {
                "prediction": round(float(prediction), 2),
                "status": "success"    
            }
        except Exception as e:
            return {
                "prediction": None,
                "status": f"Error during prediction: {str(e)}"
            }

# --- NO CHANGES NEEDED BELOW THIS LINE ---

_pipeline = None

def predict_score(data):
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictPipeline()
    return _pipeline.predict(data)

if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print(" STARTING FULL PIPELINE VALIDATION")
    print("="*60)

    test_cases = [
        ("🏆 High Performer", {
            "gender": "female", "race_ethnicity": "group E",
            "parental_level_of_education": "master's degree",
            "lunch": "standard", "test_preparation_course": "completed",
            "math_score": 95, "reading_score": 92
        }),
        ("📊 Average Student", {
            "gender": "male", "race_ethnicity": "group C",
            "parental_level_of_education": "associate's degree",
            "lunch": "standard", "test_preparation_course": "none",
            "math_score": 70, "reading_score": 72
        }),
        ("📉 Low Performer", {
            "gender": "male", "race_ethnicity": "group A",
            "parental_level_of_education": "some high school",
            "lunch": "free/reduced", "test_preparation_course": "none",
            "math_score": 30, "reading_score": 35
        })
    ]

    print("\n MODEL LOGIC CHECK:")
    print("-" * 50)
    for label, data in test_cases:
        result = predict_score(data)
        print(f"{label:20}: {result['prediction']} (Status: {result['status']})")

    print("\n ERROR HANDLING TESTS:")
    print("-" * 50)
    
    missing_data = {"gender": "male", "reading_score": 50} 
    print(f"{' Missing Columns':20}: {predict_score(missing_data)['status']}")
    
    bad_type_data = test_cases[1][1].copy()
    bad_type_data["math_score"] = "InvalidType"
    print(f"{' Invalid Data Type':20}: {predict_score(bad_type_data)['status']}")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60 + "\n")