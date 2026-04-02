import json
import pandas as pd
import os

class DataLoader:
    """Handles loading and initial preprocessing of the R2R datasets."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def load_json(self):
        """Loads the combined_dataset.json and returns a DataFrame."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)} samples from {df['source'].unique()} sources.")
        return df

if __name__ == "__main__":
    # Test loading
    loader = DataLoader("data/combined_dataset.json")
    df = loader.load_json()
    print(df.head())