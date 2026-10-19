import os
import joblib
import pandas as pd

class PredictPipeline:
    def __init__(self):
        # Use relative paths so the code works everywhere
        self.model_path = os.path.join("saved_model", "final_model.joblib")
        self.preprocessor_path = os.path.join("saved_model", "preprocessor.joblib")

        # Load artifacts with basic error checking
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
        except FileNotFoundError:
            raise FileNotFoundError("Model or Preprocessor not found in 'saved_model' folder. Please run the training pipeline first.")

    def predict(self, input_data: dict):
        """
        input_data: dictionary of features from your web form
        """
        try:
            # 1. Convert input dict → DataFrame
            df = pd.DataFrame([input_data])

            # 2. Apply preprocessing (Scaling/Encoding)
            X_processed = self.preprocessor.transform(df)

            # 3. Predict
            prediction = self.model.predict(X_processed)

            return prediction[0]
            
        except Exception as e:
            raise Exception(f"Prediction Error: {str(e)}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    pipeline = PredictPipeline()

    sample_input = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 85,
        "writing_score": 80
    }

    predicted_score = pipeline.predict(sample_input)
    print("Predicted Math Score:", predicted_score)
