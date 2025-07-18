import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import joblib
import os

class DataPreprocessor:
    """
    A class to handle data preprocessing for employee salary prediction.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'salary'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data exploration."""
        exploration = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'categorical_unique': {col: df[col].nunique() for col in df.select_dtypes(include=['object']).columns}
        }
        return exploration
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_clean = df.copy()
        
        # Fill numerical missing values with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_encoded[col] = df_encoded[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df_encoded[col] = -1
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing ones."""
        df_features = df.copy()
        
        # Experience to age ratio
        df_features['experience_age_ratio'] = df_features['years_experience'] / df_features['age']
        
        # Performance score (normalized)
        df_features['performance_score'] = (df_features['performance_rating'] - 1) / 4
        
        # Productivity score
        df_features['productivity_score'] = (
            df_features['projects_completed'] / (df_features['years_experience'] + 1)
        )
        
        # Overtime ratio
        df_features['overtime_ratio'] = df_features['overtime_hours'] / 160  # Assuming 160 hours/month
        
        return df_features
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        # Exclude target column and categorical columns
        exclude_cols = [self.target_column, 'employee_id'] if 'employee_id' in df.columns else [self.target_column]
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
        else:
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
        
        return df_scaled
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        print("Starting data preprocessing...")
        
        # 1. Handle missing values
        df_clean = self.handle_missing_values(df)
        print("[OK] Missing values handled")
        
        # 2. Create additional features
        df_features = self.create_features(df_clean)
        print("[OK] Additional features created")
        
        # 3. Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_features, fit=fit)
        print("[OK] Categorical variables encoded")
        
        # 4. Scale features
        df_scaled = self.scale_features(df_encoded, fit=fit)
        print("[OK] Features scaled")
        
        if fit:
            # Store feature columns (excluding target)
            self.feature_columns = [col for col in df_scaled.columns if col != self.target_column]
            if 'employee_id' in self.feature_columns:
                self.feature_columns.remove('employee_id')
        
        print(f"Preprocessing complete! Final shape: {df_scaled.shape}")
        return df_scaled
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        # Remove employee_id if present
        if 'employee_id' in df.columns:
            df = df.drop('employee_id', axis=1)
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split complete!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor objects."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load the preprocessor objects."""
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_column = preprocessor_data['target_column']
        print(f"Preprocessor loaded from {filepath}")

def main():
    """Main function to demonstrate data preprocessing."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/employee_salary_data.csv')
    
    if df is not None:
        # Explore data
        print("\n=== Data Exploration ===")
        exploration = preprocessor.explore_data(df)
        print(f"Dataset shape: {exploration['shape']}")
        print(f"Columns: {exploration['columns']}")
        print(f"Missing values: {exploration['missing_values']}")
        
        # Preprocess data
        print("\n=== Data Preprocessing ===")
        df_processed = preprocessor.preprocess_data(df)
        
        # Split data
        print("\n=== Data Splitting ===")
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Save preprocessed data
        print("\n=== Saving Processed Data ===")
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        print("Data preprocessing complete!")

if __name__ == "__main__":
    main()
