#“This folder is a package, allow imports from here.”
"""
Student Performance Prediction Package

This package contains modules for:
- Data ingestion
- Data transformation
- Model training
- Model evaluation
- Prediction pipeline
"""
__version__ = "1.0.0"

# Expose lightweight, reusable components
from .data_ingestion import DataIngestion
from .data_transformation import DataTransformation
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluation
from .utils import load_artifact, save_artifact

__all__ = [
    "DataIngestion",
    "DataTransformation",
    "ModelTrainer",
    "ModelEvaluation",
    "load_artifact",
    "save_artifact",
    "__version__"
]

