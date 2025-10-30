"""
Data loading module for bank churn dataset.
Handles Kaggle dataset download and CSV loading.
"""

import os
import pandas as pd
import kagglehub


class DataLoader:
    """Load bank churn dataset from Kaggle."""
    
    def __init__(self, dataset_id="harshitstark/bank-churn-train"):
        """
        Initialize data loader.
        
        Args:
            dataset_id: Kaggle dataset identifier
        """
        self.dataset_id = dataset_id
        self.data_path = None
    
    def download_dataset(self):
        """
        Download dataset from Kaggle.
        
        Returns:
            str: Path to downloaded dataset
        """
        print("=" * 70)
        print("[Step 1] Downloading Dataset")
        print("=" * 70)
        
        self.data_path = kagglehub.dataset_download(self.dataset_id)
        files = os.listdir(self.data_path)
        
        print(f"[Path] {self.data_path}")
        print(f"[Files] {files}")
        
        return self.data_path
    
    def load_csv(self, filename="train.csv"):
        """
        Load CSV file into pandas DataFrame.
        
        Args:
            filename: Name of CSV file to load
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("\n" + "=" * 70)
        print("[Step 2] Loading Data")
        print("=" * 70)
        
        if self.data_path is None:
            self.download_dataset()
        
        csv_path = os.path.join(self.data_path, filename)
        df = pd.read_csv(csv_path)
        
        print(f"[Info] Records: {len(df)}, Columns: {len(df.columns)}")
        
        return df
