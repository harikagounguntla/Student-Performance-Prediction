#Load trained model and preprocessor for student performance predictions
import joblib
from config import MODEL_PATH, PREPROCESSOR_PATH

class ModelLoadError(Exception):
    """Custom exception for model loading failures"""
    pass
def load_model():
    """
    Load the trained model and preprocessor.
    Returns:
        tuple: (model, preprocessor)
    """
    try:
        # 1. Strict Path Validation (From Option 1 logic)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor file missing at: {PREPROCESSOR_PATH}")

        # 2. Detailed Logging (From Option 2 logic)
        print(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        print("Success: Model and preprocessor are ready for use!")
        return model, preprocessor
    
    except Exception as e:
        # Re-raise the error so the calling app (Flask/FastAPI) can handle it
        raise ModelLoadError(f"Failed to load ML artifacts: {str(e)}")