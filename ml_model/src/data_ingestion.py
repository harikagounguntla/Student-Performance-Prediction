import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataIngestion:
    def __init__(self):
        self.data_path = os.path.join("data", "raw", "StudentsPerformance.csv")
        self.processed_dir = os.path.join("data", "processed")
        self.train_path = os.path.join(self.processed_dir, "train_processed.csv")
        self.test_path = os.path.join(self.processed_dir, "test_processed.csv")

    def initiate_data_ingestion(self):
        # Read raw data
        df = pd.read_csv(self.data_path)
        df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_"))

        

        # Split data
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )

        # Create processed directory
        os.makedirs(self.processed_dir, exist_ok=True)

        # Save files
        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        return self.train_path, self.test_path
