import os
import json
from datetime import datetime
import logging

# Set up logging to match your project style
logger = logging.getLogger(__name__)

def save_model_metadata(model_name, version, metrics, model_dir="saved_model"):
    """
    Creates a model_metadata.json file containing training details and performance.
    
    Args:
        model_name (str): Name of the model (e.g., "Student Performance Predictor")
        version (str): Semantic versioning (e.g., "1.0.1")
        metrics (dict): Dictionary of evaluation scores (e.g., {'r2': 0.85, 'mae': 3.2})
        model_dir (str): Path to save the metadata file.
    """
    try:
        metadata = {
            "model_details": {
                "name": model_name,
                "version": version,
                "author": "ML Team",
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_metrics": metrics,
            "bundle_contents": {
                "model_file": "final_model.joblib",
                "preprocessor_file": "preprocessor.joblib",
                "feature_schema": "feature_names.json"
            }
        }

        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"✅ Model metadata saved successfully to {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.error(f"Error occurred while saving metadata: {str(e)}")
        raise e