import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataTransformation:
    def __init__(self):
        # Numerical columns
        self.numerical_cols = [
            "reading_score",
            "writing_score"
        ]

        # Categorical columns
        self.categorical_cols = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]

        # Artifact paths
        self.artifact_dir = "artifacts"
        self.preprocessor_path = os.path.join(
            self.artifact_dir, "preprocessor.joblib"
        )
        self.feature_path = os.path.join(
            self.artifact_dir, "feature_names.json"
        )

    def get_preprocessor(self):
        """Create preprocessing pipeline"""

        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        """Read data, transform, and save artifacts"""

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Split features and target
        X_train = train_df.drop("math_score", axis=1)
        y_train = train_df["math_score"]

        X_test = test_df.drop("math_score", axis=1)
        y_test = test_df["math_score"]

        # Get preprocessor
        preprocessor = self.get_preprocessor()

        # Fit & transform
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Create artifacts directory
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Save preprocessor
        joblib.dump(preprocessor, self.preprocessor_path)

        # ✅ EXTRACT ACTUAL FEATURE NAMES (CORRECT WAY)
        num_features = self.numerical_cols

        cat_features = (
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(self.categorical_cols)
        )

        feature_names = num_features + list(cat_features)

        # Save feature names
        with open(self.feature_path, "w") as f:
            json.dump(feature_names, f, indent=4)

        return (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            preprocessor
        )
