import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    A class to handle model evaluation and visualization.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')  # Fallback to default style
        
    def plot_model_comparison(self, model_scores: Dict[str, Dict[str, float]], 
                            save_path: str = None):
        """Plot comparison of model performance metrics."""
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(model_scores).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot R2 Score
        axes[0,0].bar(comparison_df.index, comparison_df['R2'], color='skyblue')
        axes[0,0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot RMSE
        axes[0,1].bar(comparison_df.index, comparison_df['RMSE'], color='lightcoral')
        axes[0,1].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot MAE
        axes[1,0].bar(comparison_df.index, comparison_df['MAE'], color='lightgreen')
        axes[1,0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        axes[1,0].set_ylabel('MAE')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot MAPE
        axes[1,1].bar(comparison_df.index, comparison_df['MAPE'], color='gold')
        axes[1,1].set_title('Mean Absolute Percentage Error (Lower is Better)', fontweight='bold')
        axes[1,1].set_ylabel('MAPE (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  model_name: str = 'Model', save_path: str = None):
        """Plot predictions vs actual values."""
        plt.figure(figsize=self.figsize)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel('Actual Salary', fontsize=12)
        plt.ylabel('Predicted Salary', fontsize=12)
        plt.title(f'{model_name}: Predictions vs Actual Values', fontsize=14, fontweight='bold')
        plt.legend()
        
        # Add statistics
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.0f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray, 
                      model_name: str = 'Model', save_path: str = None):
        """Plot residuals for model evaluation."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 10, save_path: str = None):
        """Plot feature importance."""
        plt.figure(figsize=(10, 8))
        
        # Select top N features
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Invert y-axis to show most important at top
        
        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_salary_distribution(self, salary_data: pd.Series, 
                                title: str = 'Salary Distribution', save_path: str = None):
        """Plot salary distribution."""
        plt.figure(figsize=self.figsize)
        
        # Create histogram
        plt.hist(salary_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Salary')
        plt.ylabel('Frequency')
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add statistics
        mean_salary = salary_data.mean()
        median_salary = salary_data.median()
        plt.axvline(mean_salary, color='red', linestyle='--', label=f'Mean: ${mean_salary:,.0f}')
        plt.axvline(median_salary, color='green', linestyle='--', label=f'Median: ${median_salary:,.0f}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None):
        """Plot correlation matrix of features."""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray,
                           train_sizes: np.ndarray, save_path: str = None):
        """Plot learning curves to analyze model performance vs training size."""
        plt.figure(figsize=self.figsize)
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_evaluation_report(self, y_true: pd.Series, y_pred: np.ndarray, 
                               model_name: str, feature_importance: pd.DataFrame = None):
        """Create a comprehensive evaluation report."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print("="*80)
        print(f"EVALUATION REPORT - {model_name.upper()}")
        print("="*80)
        print(f"R² Score:                    {r2:.4f}")
        print(f"Mean Squared Error:          {mse:.2f}")
        print(f"Root Mean Squared Error:     {rmse:.2f}")
        print(f"Mean Absolute Error:         {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print("="*80)
        
        # Additional insights
        print("\nINSIGHTS:")
        if r2 > 0.8:
            print("[EXCELLENT] Excellent model performance (R2 > 0.8)")
        elif r2 > 0.6:
            print("[GOOD] Good model performance (R2 > 0.6)")
        elif r2 > 0.4:
            print("[WARNING] Moderate model performance (R2 > 0.4)")
        else:
            print("[POOR] Poor model performance (R2 <= 0.4)")
        
        print(f"• On average, predictions are off by ${mae:.0f}")
        print(f"• Model explains {r2*100:.1f}% of the variance in salary")
        
        if feature_importance is not None:
            print(f"\nTOP 5 IMPORTANT FEATURES:")
            for i, row in feature_importance.head(5).iterrows():
                print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        print("="*80)

def main():
    """Main function to demonstrate model evaluation."""
    # Load test data and trained model
    try:
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv').squeeze()
        
        # Load best model
        best_model = joblib.load('models/best_model.pkl')
        model_scores = joblib.load('models/model_scores.pkl')
        
        print("Data and models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading data/models: {e}")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Create evaluation report
    evaluator.create_evaluation_report(y_test, y_pred, 'Best Model')
    
    # Plot visualizations
    evaluator.plot_model_comparison(model_scores, 'models/model_comparison.png')
    evaluator.plot_predictions_vs_actual(y_test, y_pred, 'Best Model', 'models/predictions_vs_actual.png')
    evaluator.plot_residuals(y_test, y_pred, 'Best Model', 'models/residuals.png')
    evaluator.plot_salary_distribution(y_test, 'Test Set Salary Distribution', 'models/salary_distribution.png')
    
    # Plot feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        evaluator.plot_feature_importance(feature_importance, save_path='models/feature_importance.png')
    
    print("\nModel evaluation complete! Check the 'models' folder for visualizations.")

if __name__ == "__main__":
    main()
