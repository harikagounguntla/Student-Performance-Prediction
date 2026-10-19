# src/utils.py
import os
import joblib

def save_artifact(obj, file_path):
    """Save any Python object as an artifact"""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    joblib.dump(obj, file_path)

def load_artifact(file_path):
    """Load a saved artifact"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found at: {file_path}")
    return joblib.load(file_path)
