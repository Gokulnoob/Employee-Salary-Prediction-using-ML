#!/usr/bin/env python3
"""
Test script for the salary prediction functionality
"""
import sys
sys.path.append('src')
from prediction import SalaryPredictor

def test_prediction():
    """Test the prediction functionality"""
    print("🧪 Testing Salary Prediction System...")
    
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Test loading
    if not predictor.load_model_and_preprocessor():
        print("❌ Failed to load model/preprocessor")
        return False
    
    print("✅ Model and preprocessor loaded successfully")
    
    # Test with sample data
    sample_data = {
        'age': 28,
        'years_experience': 5,
        'department': 'Engineering',
        'job_title': 'Mid-Level',
        'education_level': 'Bachelor',
        'location': 'San Francisco',
        'performance_rating': 4.2,
        'overtime_hours': 8,
        'projects_completed': 12
    }
    
    print("\n📊 Testing prediction with sample data...")
    result = predictor.predict_salary(sample_data)
    
    if 'error' in result:
        print(f"❌ Prediction Error: {result['error']}")
        return False
    else:
        print(f"✅ Prediction successful!")
        print(f"   Predicted Salary: {result['formatted_salary']}")
        if result['prediction_interval']:
            lower = result['prediction_interval']['lower']
            upper = result['prediction_interval']['upper']
            print(f"   Confidence Interval: ${lower:,.2f} - ${upper:,.2f}")
        return True

def test_insights():
    """Test the insights functionality"""
    print("\n🔍 Testing salary insights...")
    
    predictor = SalaryPredictor()
    predictor.load_model_and_preprocessor()
    
    sample_data = {
        'age': 35,
        'years_experience': 10,
        'department': 'Finance',
        'job_title': 'Senior',
        'education_level': 'Master',
        'location': 'New York',
        'performance_rating': 4.5,
        'overtime_hours': 5,
        'projects_completed': 8
    }
    
    insights = predictor.get_salary_insights(sample_data)
    
    if 'error' in insights:
        print(f"❌ Insights Error: {insights['error']}")
        return False
    else:
        print(f"✅ Insights generated successfully!")
        print(f"   Salary Range: {insights['salary_range']}")
        print(f"   Recommendations: {len(insights['recommendations'])} items")
        return True

def main():
    """Main test function"""
    print("="*60)
    print("🚀 EMPLOYEE SALARY PREDICTION - SYSTEM TEST")
    print("="*60)
    
    success = True
    
    # Run tests
    success &= test_prediction()
    success &= test_insights()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED! The prediction system is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()
