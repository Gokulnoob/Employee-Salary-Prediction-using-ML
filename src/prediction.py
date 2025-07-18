import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    """
    A class to make salary predictions using the trained model.
    """
    
    def __init__(self, model_path: str = 'models/best_model.pkl', 
                 preprocessor_path: str = 'models/preprocessor.pkl'):
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Dict[str, Any]] = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
    def load_model_and_preprocessor(self):
        """Load the trained model and preprocessor."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"[OK] Model loaded from {self.model_path}")
            
            preprocessor_data = joblib.load(self.preprocessor_path)
            self.preprocessor = preprocessor_data
            print(f"[OK] Preprocessor loaded from {self.preprocessor_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading model/preprocessor: {e}")
            return False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        # Ensure preprocessor is loaded
        if self.preprocessor is None:
            if not self.load_model_and_preprocessor():
                raise RuntimeError("Failed to load preprocessor")
        
        # Validate required fields
        required_fields = ['age', 'years_experience', 'department', 'job_title', 
                          'education_level', 'location', 'performance_rating', 
                          'overtime_hours', 'projects_completed']
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate numeric fields
        try:
            input_data['age'] = max(18, min(65, int(input_data['age'])))  # Age between 18-65
            input_data['years_experience'] = max(0, min(40, int(input_data['years_experience'])))  # Experience 0-40
            input_data['performance_rating'] = max(1.0, min(5.0, float(input_data['performance_rating'])))  # Rating 1-5
            input_data['overtime_hours'] = max(0, min(100, int(input_data['overtime_hours'])))  # Overtime 0-100
            input_data['projects_completed'] = max(0, min(50, int(input_data['projects_completed'])))  # Projects 0-50
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric input: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Create additional features
        df['experience_age_ratio'] = df['years_experience'] / df['age'].clip(lower=1)  # Prevent division by zero
        df['performance_score'] = (df['performance_rating'] - 1) / 4
        df['productivity_score'] = df['projects_completed'] / (df['years_experience'] + 1)
        df['overtime_ratio'] = df['overtime_hours'] / 160
        
        # Encode categorical variables
        assert self.preprocessor is not None, "Preprocessor should be loaded by now"
        for col, encoder in self.preprocessor['label_encoders'].items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen category
                    df[col] = -1
        
        # Ensure all required columns are present
        for col in self.preprocessor['feature_columns']:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the feature columns used in training
        df = df[self.preprocessor['feature_columns']]
        
        # Scale features
        df_scaled = self.preprocessor['scaler'].transform(df)
        
        return pd.DataFrame(df_scaled, columns=self.preprocessor['feature_columns'])
    
    def predict_salary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make salary prediction for a single employee."""
        if self.model is None or self.preprocessor is None:
            if not self.load_model_and_preprocessor():
                return {"error": "Failed to load model or preprocessor"}
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            assert self.model is not None, "Model should be loaded by now"
            prediction = self.model.predict(processed_data)[0]
            
            # Get prediction interval (if supported)
            prediction_interval = None
            if hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
                # For tree-based models, calculate prediction interval
                try:
                    all_predictions = []
                    for estimator in self.model.estimators_:
                        pred = estimator.predict(processed_data)[0]
                        all_predictions.append(pred)
                    
                    std_pred = np.std(all_predictions)
                    prediction_interval = {
                        'lower': prediction - 1.96 * std_pred,
                        'upper': prediction + 1.96 * std_pred
                    }
                except:
                    pass
            
            result = {
                'predicted_salary': round(prediction, 2),
                'formatted_salary': f"${prediction:,.2f}",
                'prediction_interval': prediction_interval,
                'input_data': input_data
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make salary predictions for multiple employees."""
        results = []
        
        for input_data in input_data_list:
            result = self.predict_salary(input_data)
            results.append(result)
        
        return results
    
    def get_salary_insights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights about salary prediction."""
        prediction_result = self.predict_salary(input_data)
        
        if "error" in prediction_result:
            return prediction_result
        
        predicted_salary = prediction_result['predicted_salary']
        
        insights = {
            'predicted_salary': predicted_salary,
            'salary_range': self._get_salary_range(predicted_salary),
            'experience_impact': self._analyze_experience_impact(input_data),
            'recommendations': self._get_recommendations(input_data, predicted_salary)
        }
        
        return insights
    
    def _get_salary_range(self, salary: float) -> str:
        """Categorize salary into ranges."""
        if salary < 50000:
            return "Entry Level (< $50K)"
        elif salary < 75000:
            return "Mid Level ($50K - $75K)"
        elif salary < 100000:
            return "Senior Level ($75K - $100K)"
        elif salary < 150000:
            return "Lead Level ($100K - $150K)"
        else:
            return "Executive Level (> $150K)"
    
    def _analyze_experience_impact(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of experience on salary."""
        current_exp = input_data['years_experience']
        
        # Create scenarios with different experience levels
        scenarios = {}
        for exp_change in [-2, -1, 1, 2]:
            new_exp = max(0, current_exp + exp_change)
            modified_data = input_data.copy()
            modified_data['years_experience'] = new_exp
            
            result = self.predict_salary(modified_data)
            if "error" not in result:
                # Get current prediction for comparison
                current_result = self.predict_salary(input_data)
                current_salary = current_result.get('predicted_salary', 0) if "error" not in current_result else 0
                
                scenarios[f'{exp_change:+d}_years'] = {
                    'experience': new_exp,
                    'predicted_salary': result['predicted_salary'],
                    'salary_change': result['predicted_salary'] - current_salary
                }
        
        return scenarios
    
    def _get_recommendations(self, input_data: Dict[str, Any], predicted_salary: float) -> List[str]:
        """Get recommendations to improve salary."""
        recommendations = []
        
        # Experience-based recommendations
        if input_data['years_experience'] < 5:
            recommendations.append("Gain more work experience through challenging projects")
        
        # Performance-based recommendations
        if input_data['performance_rating'] < 4:
            recommendations.append("Focus on improving performance rating")
        
        # Education-based recommendations
        if input_data['education_level'] in ['High School', 'Bachelor']:
            recommendations.append("Consider pursuing higher education or certifications")
        
        # Productivity recommendations
        if input_data['projects_completed'] < 5:
            recommendations.append("Take on more projects to demonstrate productivity")
        
        # Department/role recommendations
        if input_data['department'] in ['HR', 'Operations']:
            recommendations.append("Consider transitioning to higher-paying departments like Engineering or Finance")
        
        return recommendations

def create_sample_inputs():
    """Create sample inputs for testing."""
    samples = [
        {
            'age': 28,
            'years_experience': 5,
            'department': 'Engineering',
            'job_title': 'Mid-Level',
            'education_level': 'Bachelor',
            'location': 'San Francisco',
            'performance_rating': 4.2,
            'overtime_hours': 8,
            'projects_completed': 12
        },
        {
            'age': 35,
            'years_experience': 10,
            'department': 'Finance',
            'job_title': 'Senior',
            'education_level': 'Master',
            'location': 'New York',
            'performance_rating': 4.5,
            'overtime_hours': 5,
            'projects_completed': 8
        },
        {
            'age': 42,
            'years_experience': 15,
            'department': 'Engineering',
            'job_title': 'Lead',
            'education_level': 'Master',
            'location': 'Seattle',
            'performance_rating': 4.8,
            'overtime_hours': 10,
            'projects_completed': 20
        }
    ]
    
    return samples

def main():
    """Main function to demonstrate salary prediction."""
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Load model and preprocessor
    if not predictor.load_model_and_preprocessor():
        print("Failed to load model. Please run model training first.")
        return
    
    # Create sample inputs
    sample_inputs = create_sample_inputs()
    
    print("="*80)
    print("SALARY PREDICTION DEMONSTRATION")
    print("="*80)
    
    for i, input_data in enumerate(sample_inputs, 1):
        print(f"\n--- EMPLOYEE {i} ---")
        print("Input Data:")
        for key, value in input_data.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        result = predictor.predict_salary(input_data)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPredicted Salary: {result['formatted_salary']}")
            
            if result['prediction_interval']:
                lower = result['prediction_interval']['lower']
                upper = result['prediction_interval']['upper']
                print(f"Prediction Interval: ${lower:,.2f} - ${upper:,.2f}")
        
        # Get insights
        insights = predictor.get_salary_insights(input_data)
        if "error" not in insights:
            print(f"Salary Range: {insights['salary_range']}")
            print("Recommendations:")
            for rec in insights['recommendations']:
                print(f"  • {rec}")
    
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION")
    print("="*80)
    
    # Interactive prediction
    print("\nYou can now make your own predictions!")
    print("Enter 'quit' to exit.")
    
    while True:
        try:
            print("\nEnter employee details:")
            age = input("Age: ")
            if age.lower() == 'quit':
                break
            
            years_exp = input("Years of Experience: ")
            department = input("Department (Engineering/Sales/Marketing/HR/Finance/Operations/IT): ")
            job_title = input("Job Title (Junior/Mid-Level/Senior/Lead/Manager/Director): ")
            education = input("Education Level (High School/Bachelor/Master/PhD): ")
            location = input("Location (New York/San Francisco/Chicago/Austin/Boston/Seattle/Remote): ")
            performance = input("Performance Rating (1-5): ")
            overtime = input("Overtime Hours per Month: ")
            projects = input("Projects Completed: ")
            
            # Create input dictionary
            user_input = {
                'age': int(age),
                'years_experience': int(years_exp),
                'department': department,
                'job_title': job_title,
                'education_level': education,
                'location': location,
                'performance_rating': float(performance),
                'overtime_hours': int(overtime),
                'projects_completed': int(projects)
            }
            
            # Make prediction
            result = predictor.predict_salary(user_input)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nPredicted Salary: {result['formatted_salary']}")
                
                # Get insights
                insights = predictor.get_salary_insights(user_input)
                if "error" not in insights:
                    print(f"Salary Range: {insights['salary_range']}")
                    print("Recommendations:")
                    for rec in insights['recommendations']:
                        print(f"  • {rec}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with valid inputs.")
    
    print("\nThank you for using the Salary Predictor!")

if __name__ == "__main__":
    main()
