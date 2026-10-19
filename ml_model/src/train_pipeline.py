import os
import sys
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluation
from src.save_features import save_feature_names
from src.model_metadata import save_model_metadata
from src.utils import load_artifact as load_object

if __name__ == "__main__":
    try:
        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        
        # Now this will correctly unpack 5 values without error
        X_train, X_test, y_train, y_test, preprocessor_obj = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Extraction will now work because preprocessor_obj is the transformer
        feature_names = list(preprocessor_obj.get_feature_names_out())
        print(f"Extracted {len(feature_names)} features.")

        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        # ✅ FIX: Extract feature names from preprocessor before using them
        feature_names = list(preprocessor_obj.get_feature_names_out())

        # 3️⃣ Model Training
        trainer = ModelTrainer()
        # Cleaned call: We pass X, y, and the preprocessor to save the model bundle
        model_path = trainer.train_and_save(X_train, y_train,preprocessor_obj,feature_names)

        # 4️⃣ Model Evaluation
        model = load_object(model_path)
        evaluation = ModelEvaluation()
        metrics = evaluation.evaluate(model, X_test, y_test)
        print(f"\n Model Evaluation Metrics: {metrics}")

        # 5️⃣ Create the Production Bundle
        print("\nCreating production bundle in 'saved_model'...")

        # Save Feature Names dynamically 
        save_feature_names(preprocessor_obj, model_dir="saved_model")

        # Save Metadata (records performance and training date)
        save_model_metadata(
            model_name="Student Performance Predictor",
            version="1.0.0",
            metrics=metrics,
            model_dir="saved_model"
        )

        print("\n Training Pipeline Complete. 'saved_model' folder is fully populated!")

    except Exception as e:
        print(f"Error occurred in pipeline: {e}")
        sys.exit(1)