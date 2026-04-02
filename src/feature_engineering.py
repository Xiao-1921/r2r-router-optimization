import pandas as pd
import numpy as np
import re

class RouterFeatureExtractor:
    """Extracts linguistic and model-based features for routing decisions."""
    
    def __init__(self):
        pass

    def extract_linguistic_features(self, df):
        """Extract basic features from the question text."""
        # 1. Question Length
        df['feat_length_char'] = df['question'].apply(len)
        
        # 2. Contains specific keywords (Reasoning vs Factual)
        df['feat_has_why'] = df['question'].str.contains('why', case=False).astype(int)
        df['feat_has_how'] = df['question'].str.contains('how', case=False).astype(int)
        
        # 3. Simple NER count placeholder (e.g., Capitalized words not at start)
        df['feat_ner_estimate'] = df['question'].apply(lambda x: len(re.findall(r'\b[A-Z][a-z]+\b', x)))
        
        return df

    def integrate_model_uncertainty(self, df, logprobs):
        """
        Integrates token-level logprobs to calculate entropy.
        TODO: Implement this once inference logs are available.
        """
        # Placeholder for Task 1: Entropy Calculation
        # df['feat_entropy'] = -np.sum(probs * np.log(probs))
        return df

if __name__ == "__main__":
    # Example usage
    extractor = RouterFeatureExtractor()
    # Assuming df is loaded from data_utils
    # df_features = extractor.extract_linguistic_features(df)