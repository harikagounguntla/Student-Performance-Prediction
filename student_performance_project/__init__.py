# student_performance_project/__init__.py

"""
Student Performance Prediction Package

Usage:
    from student_performance_project import predict_score, load_model
"""
from predict import PredictPipeline,predict_score
from model_loader import load_model
from config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS

__version__ = "1.0.0"

__all__ = [
    "PredictPipeline",
    "predict_score",
    "load_model",
    "MODEL_PATH",
    "PREPROCESSOR_PATH",
    "FEATURE_COLUMNS",
]