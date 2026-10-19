"""
Student Performance Prediction - Data Preprocessing Pipeline
Handles numeric scaling + categorical encoding for student features
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, model_dir: str = "saved_model"):
        """
        Initialize preprocessor with target save directory
        Args:
            model_dir: Directory to save fitted preprocessor
        """
        self.model_dir = model_dir
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        self._preprocessor = None
        
    def get_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numeric and categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Detected {len(numeric_features)} numeric features: {numeric_features}")
        logger.info(f"Detected {len(categorical_features)} categorical features: {categorical_features}")
        
        return numeric_features, categorical_features
    
    def build_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Build complete preprocessing pipeline
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            
        Returns:
            Fitted ColumnTransformer pipeline
        """
        # Numeric pipeline: Standard scaling
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: One-hot encoding (handle unknown categories)
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(
                handle_unknown='ignore', 
                drop='first',  # Avoid multicollinearity
                sparse_output=False
            ))
        ])
        
        # Combine both transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough',  # Keep any unprocessed columns
            verbose_feature_names_out=False  # Keep original names
        )
        self._preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor on training data and transform
        Args:
            X: Training DataFrame  
        Returns:
            Transformed numpy array
        """
        if self._preprocessor is None:
            raise ValueError("Must call build_preprocessor() first")
            
        logger.info("Fitting preprocessor on training data...")
        X_transformed = self._preprocessor.fit_transform(X)
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor
        Args:
            X: New data DataFrame   
        Returns:
            Transformed numpy array
        """
        if self._preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first")
            
        logger.info("Transforming new data...")
        X_transformed = self._preprocessor.transform(X)
        return X_transformed
    def get_feature_names_out(self) -> List[str]:
        if self._preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        return list(self._preprocessor.get_feature_names_out())
    def save(self) -> str:
        """
        Save fitted preprocessor to disk
        Returns:
            Path to saved preprocessor
        """
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self._preprocessor, self.preprocessor_path)
        logger.info(f"Preprocessor saved to {self.preprocessor_path}")
        return self.preprocessor_path
    
    @classmethod
    def load(cls, model_dir: str = "saved_model") -> 'DataPreprocessor':
        """
        Load saved preprocessor
        Args:
            model_dir: Directory containing saved preprocessor   
        Returns:
            Loaded DataPreprocessor instance
        """
        preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        preprocessor = joblib.load(preprocessor_path)
        
        instance = cls(model_dir)
        instance._preprocessor = preprocessor
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return instance

# Usage example (remove in production)
if __name__ == "__main__":
    # Example student features
    sample_data = pd.DataFrame({
        'study_hours': [2, 5, 8],
        'previous_scores': [60, 85, 92],
        'attendance': [75, 95, 98],
        'gender': ['M', 'F', 'M'],
        'parental_education': ['High School', 'Graduate', 'High School']
    })
    
    preprocessor = DataPreprocessor()
    num_features = ['study_hours', 'previous_scores', 'attendance']
    cat_features = ['gender', 'parental_education']
    
    # Build, fit, save
    prep_pipeline = preprocessor.build_preprocessor(num_features, cat_features)
    X_transformed = preprocessor.fit_transform(sample_data)
    preprocessor.save()
    
    print("Preprocessor ready for production!")
    print(f"Output shape: {X_transformed.shape}")
