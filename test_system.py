"""
Simple test script to verify the Employee Salary Prediction project is working correctly.
"""

from src.prediction import SalaryPredictor
import pandas as pd

def test_prediction():
    """Test the salary prediction functionality."""
    print("=" * 60)
    print("TESTING EMPLOYEE SALARY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = SalaryPredictor()
        print("[OK] Predictor initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize predictor: {e}")
        return False
    
    # Test data
    test_employee = {
        'age': 30,
        'years_experience': 5,
        'department': 'Engineering',
        'job_title': 'Senior',
        'education_level': 'Master',
        'location': 'San Francisco',
        'performance_rating': 4.2,
        'overtime_hours': 10,
        'projects_completed': 8
    }
    
    print("\nTest Employee Data:")
    for key, value in test_employee.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    try:
        result = predictor.predict_salary(test_employee)
        
        if "error" in result:
            print(f"[ERROR] Prediction failed: {result['error']}")
            return False
        
        print(f"\n[SUCCESS] Predicted Salary: {result['formatted_salary']}")
        
        if result['prediction_interval']:
            lower = result['prediction_interval']['lower']
            upper = result['prediction_interval']['upper']
            print(f"Confidence Interval: ${lower:,.2f} - ${upper:,.2f}")
        
        # Test multiple predictions
        print("\nTesting multiple scenarios...")
        
        scenarios = [
            {
                'name': 'Junior Developer',
                'data': {
                    'age': 24, 'years_experience': 1, 'department': 'Engineering',
                    'job_title': 'Junior', 'education_level': 'Bachelor',
                    'location': 'Austin', 'performance_rating': 3.8,
                    'overtime_hours': 5, 'projects_completed': 3
                }
            },
            {
                'name': 'Senior Manager',
                'data': {
                    'age': 45, 'years_experience': 15, 'department': 'Finance',
                    'job_title': 'Manager', 'education_level': 'Master',
                    'location': 'New York', 'performance_rating': 4.5,
                    'overtime_hours': 8, 'projects_completed': 12
                }
            }
        ]
        
        for scenario in scenarios:
            result = predictor.predict_salary(scenario['data'])
            if "error" not in result:
                print(f"  {scenario['name']}: {result['formatted_salary']}")
            else:
                print(f"  {scenario['name']}: [ERROR] {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Prediction test failed: {e}")
        return False

def test_data_files():
    """Test if all required data files exist."""
    print("\n" + "=" * 60)
    print("TESTING DATA FILES")
    print("=" * 60)
    
    required_files = [
        'data/employee_salary_data.csv',
        'data/X_train.csv',
        'data/X_test.csv', 
        'data/y_train.csv',
        'data/y_test.csv',
        'models/best_model.pkl',
        'models/preprocessor.pkl',
        'models/model_scores.pkl'
    ]
    
    all_good = True
    for file_path in required_files:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"[OK] {file_path} - Shape: {df.shape}")
            else:
                import joblib
                joblib.load(file_path)
                print(f"[OK] {file_path} - Loaded successfully")
        except Exception as e:
            print(f"[ERROR] {file_path} - {e}")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("EMPLOYEE SALARY PREDICTION - SYSTEM TEST")
    print("This will verify that all components are working correctly.\n")
    
    # Test 1: Data files
    data_test = test_data_files()
    
    # Test 2: Prediction functionality
    prediction_test = test_prediction()
    
    # Final result
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if data_test and prediction_test:
        print("[SUCCESS] All tests passed! Your ML project is working perfectly!")
        print("\nNext steps:")
        print("  1. Run 'python launcher.py' for guided experience")
        print("  2. Run 'python src/prediction.py' for interactive predictions")
        print("  3. Open Jupyter notebook for data exploration")
        print("  4. Read BEGINNER_GUIDE.md for detailed explanations")
    else:
        print("[FAILURE] Some tests failed. Please check the error messages above.")
        if not data_test:
            print("  - Data files issue: Run the preprocessing and training scripts")
        if not prediction_test:
            print("  - Prediction issue: Check if models are trained correctly")

if __name__ == "__main__":
    main()
