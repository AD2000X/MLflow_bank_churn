"""
Data preprocessing module.
Handles data cleaning, encoding, and train-test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess bank churn dataset."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor.
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None
    
    def preprocess(self, df):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        print("\n" + "=" * 70)
        print("[Step 3] Data Preprocessing")
        print("=" * 70)
        
        print(f"[Shape] Original: {df.shape}")
        
        # Drop unnecessary columns
        columns_to_drop = ['id', 'CustomerId', 'Surname']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"[Info] Dropped columns: {columns_to_drop}")
        
        # Check missing values
        missing = df.isnull().sum().sum()
        print(f"[Info] Missing values: {missing}")
        
        # Check target distribution
        if 'Exited' in df.columns:
            print(f"[Info] Churn rate: {df['Exited'].mean():.2%}")
        
        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)
        print(f"[Info] After encoding: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=['Exited']).astype(float)
        y = df['Exited']
        
        self.feature_names = X.columns.tolist()
        print(f"[Info] Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 70)
        print("[Step 5] Splitting Data")
        print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"[Info] Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
