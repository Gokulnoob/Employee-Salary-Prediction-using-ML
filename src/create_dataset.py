import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create sample employee data
def create_sample_data():
    """Create a sample employee dataset for salary prediction"""
    
    # Define possible values for categorical variables
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'IT']
    job_titles = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director']
    education_levels = ['Bachelor', 'Master', 'PhD', 'High School']
    locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Boston', 'Seattle', 'Remote']
    
    # Generate sample data
    n_samples = 10000
    
    data = {
        'employee_id': range(1, n_samples + 1),
        'age': np.random.randint(22, 65, n_samples),
        'years_experience': np.random.randint(0, 35, n_samples),
        'department': np.random.choice(departments, n_samples),
        'job_title': np.random.choice(job_titles, n_samples),
        'education_level': np.random.choice(education_levels, n_samples),
        'location': np.random.choice(locations, n_samples),
        'performance_rating': np.random.uniform(1, 5, n_samples).round(2),
        'overtime_hours': np.random.randint(0, 20, n_samples),
        'projects_completed': np.random.randint(0, 15, n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create salary based on realistic factors
    base_salary = 40000
    
    # Department multiplier
    dept_multiplier = {
        'Engineering': 1.4, 'IT': 1.3, 'Finance': 1.2, 
        'Sales': 1.1, 'Marketing': 1.0, 'HR': 0.9, 'Operations': 0.8
    }
    
    # Job title multiplier
    title_multiplier = {
        'Junior': 1.0, 'Mid-Level': 1.3, 'Senior': 1.6, 
        'Lead': 1.9, 'Manager': 2.2, 'Director': 2.8
    }
    
    # Education multiplier
    edu_multiplier = {
        'High School': 1.0, 'Bachelor': 1.2, 'Master': 1.4, 'PhD': 1.6
    }
    
    # Location multiplier
    loc_multiplier = {
        'New York': 1.3, 'San Francisco': 1.4, 'Seattle': 1.2, 
        'Boston': 1.2, 'Chicago': 1.1, 'Austin': 1.0, 'Remote': 0.9
    }
    
    # Calculate salary
    df['salary'] = (
        base_salary * 
        df['department'].map(dept_multiplier) *
        df['job_title'].map(title_multiplier) *
        df['education_level'].map(edu_multiplier) *
        df['location'].map(loc_multiplier) *
        (1 + df['years_experience'] * 0.03) *
        (1 + df['performance_rating'] * 0.1) *
        (1 + df['overtime_hours'] * 0.02) *
        (1 + df['projects_completed'] * 0.05)
    ).round(0).astype(int)
    
    # Add some random noise
    df['salary'] += np.random.normal(0, 5000, n_samples).astype(int)
    
    # Ensure minimum salary
    df['salary'] = np.maximum(df['salary'], 35000)
    
    return df

if __name__ == "__main__":
    # Create the dataset
    df = create_sample_data()
    
    # Save to CSV
    output_path = os.path.join('data', 'employee_salary_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nSalary statistics:")
    print(df['salary'].describe())
