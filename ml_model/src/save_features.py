import os
import json
import logging

# Set up logging to match your preprocessor
logger = logging.getLogger(__name__)

def save_feature_names(preprocessor_obj, model_dir="saved_model"):
    """
    Extracts final feature names from a fitted preprocessor and saves to JSON.
    
    Args:
        preprocessor_obj: The fitted DataPreprocessor or ColumnTransformer instance.
        model_dir: Path to the production bundle folder.
    """
    try:
        # Extract the final names (this handles One-Hot Encoding expansion)
        # Using the method from your DataPreprocessor class
        feature_names = preprocessor_obj.get_feature_names_out()
        
        # Ensure it's a list for JSON compatibility
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)

        os.makedirs(model_dir, exist_ok=True)
        feature_path = os.path.join(model_dir, "feature_names.json")

        with open(feature_path, "w") as f:
            json.dump(feature_names, f, indent=4)
            
        logger.info(f"✅ Successfully saved {len(feature_names)} features to {feature_path}")
        
    except Exception as e:
        logger.error(f"Error saving feature names: {str(e)}")
        raise e