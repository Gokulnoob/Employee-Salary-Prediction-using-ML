import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
import joblib
import os
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A class to handle model training and evaluation for salary prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models for comparison."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            print("[WARNING] xgboost is not installed. Skipping XGBoost model.")
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        else:
            print("[WARNING] lightgbm is not installed. Skipping LightGBM model.")
        print(f"Initialized {len(self.models)} models for training")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models with the training data."""
        print("Training models...")
        
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"[OK] {name} trained successfully")
            except Exception as e:
                print(f"[ERROR] Error training {name}: {e}")
        
        print(f"Training complete! {len(self.trained_models)} models trained successfully")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models on test data."""
        print("Evaluating models...")
        
        for name, model in self.trained_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                self.model_scores[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                }
                
                print(f"[OK] {name} evaluated")
                
            except Exception as e:
                print(f"[ERROR] Error evaluating {name}: {e}")
        
        return self.model_scores
    
    def cross_validate_models(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5):
        """Perform cross-validation on all models."""
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = {}
        
        for name, model in self.trained_models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                cv_scores[name] = {
                    'mean_cv_score': scores.mean(),
                    'std_cv_score': scores.std(),
                    'cv_scores': scores.tolist()
                }
                print(f"[OK] {name} - CV R2 Score: {scores.mean():.4f} (+/-{scores.std():.4f})")
            except Exception as e:
                print(f"[ERROR] Error in CV for {name}: {e}")
        
        return cv_scores
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str = 'random_forest') -> Dict[str, Any]:
        """Perform hyperparameter tuning for specified model."""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids or model_name not in self.models:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        try:
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, 
                scoring='r2', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update the trained model with best parameters
            self.trained_models[f'{model_name}_tuned'] = grid_search.best_estimator_
            
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"[OK] Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"[OK] Best CV score: {grid_search.best_score_:.4f}")
            
            return tuning_results
            
        except Exception as e:
            print(f"[ERROR] Error in hyperparameter tuning for {model_name}: {e}")
            return {}
    
    def select_best_model(self, metric: str = 'R2') -> str:
        """Select the best model based on specified metric."""
        if not self.model_scores:
            print("No model scores available. Please evaluate models first.")
            return None
        
        if metric == 'R2':
            # Higher R2 is better
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x]['R2'])
        elif metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            # Lower is better for these metrics
            best_model_name = min(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x][metric])
        else:
            print(f"Metric '{metric}' not supported")
            return None
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        print(f"Best model selected: {best_model_name}")
        print(f"Best {metric} score: {self.model_scores[best_model_name][metric]:.4f}")
        
        return best_model_name
    
    def get_feature_importance(self, X_train: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found in trained models")
            return None
        
        model = self.trained_models[model_name]
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature importance")
            return None
    
    def save_models(self, save_dir: str = 'models'):
        """Save all trained models."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for name, model in self.trained_models.items():
            filepath = os.path.join(save_dir, f'{name}_model.pkl')
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save model scores
        scores_filepath = os.path.join(save_dir, 'model_scores.pkl')
        joblib.dump(self.model_scores, scores_filepath)
        print(f"Saved model scores to {scores_filepath}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_filepath = os.path.join(save_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_filepath)
            print(f"Saved best model to {best_model_filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def print_model_comparison(self):
        """Print a comparison of all model performances."""
        if not self.model_scores:
            print("No model scores available")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.model_scores).T
        comparison_df = comparison_df.round(4)
        
        print(comparison_df.to_string())
        
        # Highlight best model for each metric
        print("\n" + "="*40)
        print("BEST MODELS PER METRIC")
        print("="*40)
        
        for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
            if metric == 'R2':
                best_model = comparison_df[metric].idxmax()
                best_score = comparison_df[metric].max()
            else:
                best_model = comparison_df[metric].idxmin()
                best_score = comparison_df[metric].min()
            
            print(f"{metric:>6}: {best_model:<20} ({best_score:.4f})")

def main():
    """Main function to demonstrate model training."""
    # Load preprocessed data
    try:
        X_train = pd.read_csv('data/X_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_train = pd.read_csv('data/y_train.csv').squeeze()
        y_test = pd.read_csv('data/y_test.csv').squeeze()
        
        print(f"Data loaded successfully!")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Initialize and train models
    trainer.initialize_models()
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    trainer.evaluate_models(X_test, y_test)
    
    # Cross-validation
    trainer.cross_validate_models(X_train, y_train)
    
    # Print comparison
    trainer.print_model_comparison()
    
    # Select best model
    best_model_name = trainer.select_best_model('R2')
    
    # Get feature importance
    if best_model_name:
        feature_importance = trainer.get_feature_importance(X_train, best_model_name)
        if feature_importance is not None:
            print(f"\nTop 10 Important Features ({best_model_name}):")
            print(feature_importance.head(10))
    
    # Hyperparameter tuning for best model (if it supports it)
    if best_model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
        print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
        tuning_results = trainer.hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # Re-evaluate with tuned model
        tuned_model_name = f'{best_model_name}_tuned'
        if tuned_model_name in trainer.trained_models:
            trainer.evaluate_models(X_test, y_test)
            trainer.print_model_comparison()
    
    # Save models
    trainer.save_models()
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()
