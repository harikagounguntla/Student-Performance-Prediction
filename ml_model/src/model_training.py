import os
import joblib
import json
import shutil
from sklearn.linear_model import LinearRegression

class ModelTrainer:
    def __init__(self):
        # Define directories
        self.art_dir = "artifacts"
        self.save_dir = "saved_model"
        
        os.makedirs(self.art_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_and_save(self, X_train, y_train, preprocessor, feature_names):
        # 1. Train
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 2. Save everything to 'artifacts' first (The Lab)
        model_art_path = os.path.join(self.art_dir, "student_model.joblib")
        prep_art_path = os.path.join(self.art_dir, "preprocessor.joblib")
        feat_art_path = os.path.join(self.art_dir, "feature_names.json")
        
        joblib.dump(model, model_art_path)
        joblib.dump(preprocessor, prep_art_path)
        with open(feat_art_path, 'w') as f:
            json.dump(list(feature_names), f)

        # 3. Copy/Save to 'saved_model' for Production (The Store)
        joblib.dump(model, os.path.join(self.save_dir, "final_model.joblib"))
        shutil.copy(prep_art_path, os.path.join(self.save_dir, "preprocessor.joblib"))
        shutil.copy(feat_art_path, os.path.join(self.save_dir, "feature_names.json"))

        print(f" Full bundle saved to {self.save_dir}")
        return os.path.join(self.save_dir, "final_model.joblib")

