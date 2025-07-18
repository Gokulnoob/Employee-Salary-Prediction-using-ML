# Employee Salary Prediction - Machine Learning Project

A comprehensive machine learning project for predicting employee salaries based on various features such as experience, education, department, and performance metrics.

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline for predicting employee salaries. It's designed as an educational project to understand the end-to-end process of building, training, and deploying a machine learning model.

## 📁 Project Structure

```
Employee Salary Prediction/
├── data/                          # Dataset files
│   ├── employee_salary_data.csv   # Raw dataset
│   ├── X_train.csv                # Training features
│   ├── X_test.csv                 # Testing features
│   ├── y_train.csv                # Training targets
│   └── y_test.csv                 # Testing targets
├── notebooks/                     # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── src/                          # Source code
│   ├── create_dataset.py         # Dataset generation
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── model_training.py         # Model training and evaluation
│   ├── model_evaluation.py       # Model evaluation and visualization
│   └── prediction.py             # Salary prediction interface
├── models/                       # Trained models and artifacts
│   ├── best_model.pkl            # Best performing model
│   ├── preprocessor.pkl          # Data preprocessing objects
│   └── model_scores.pkl          # Model performance metrics
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project:**

   ```bash
   # If using git
   git clone <repository-url>
   cd "Employee Salary Prediction"

   # Or simply download and extract the folder
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data:**

   ```bash
   python src/create_dataset.py
   ```

4. **Run the complete pipeline:**

   ```bash
   # Data preprocessing
   python src/data_preprocessing.py

   # Model training
   python src/model_training.py

   # Model evaluation
   python src/model_evaluation.py

   # Make predictions
   python src/prediction.py
   ```

## 📊 Dataset Features

The dataset includes the following features:

| Feature              | Type        | Description                                                                       |
| -------------------- | ----------- | --------------------------------------------------------------------------------- |
| `age`                | Numerical   | Employee age (22-65 years)                                                        |
| `years_experience`   | Numerical   | Years of work experience (0-35 years)                                             |
| `department`         | Categorical | Department (Engineering, Sales, Marketing, HR, Finance, Operations, IT)           |
| `job_title`          | Categorical | Job level (Junior, Mid-Level, Senior, Lead, Manager, Director)                    |
| `education_level`    | Categorical | Education (High School, Bachelor, Master, PhD)                                    |
| `location`           | Categorical | Work location (New York, San Francisco, Chicago, Austin, Boston, Seattle, Remote) |
| `performance_rating` | Numerical   | Performance score (1.0-5.0)                                                       |
| `overtime_hours`     | Numerical   | Monthly overtime hours (0-20)                                                     |
| `projects_completed` | Numerical   | Number of projects completed (0-15)                                               |
| `salary`             | Numerical   | **Target variable** - Annual salary in USD                                        |

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing

- **Missing value handling**: Imputation using median/mode
- **Feature engineering**: Creating derived features like experience-to-age ratio
- **Categorical encoding**: Label encoding for categorical variables
- **Feature scaling**: StandardScaler for numerical features
- **Train-test split**: 80-20 split with stratification

### 2. Model Training

The project trains and compares multiple algorithms:

- **Linear Regression**: Baseline linear model
- **Ridge Regression**: L2 regularized linear model
- **Lasso Regression**: L1 regularized linear model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosted tree ensemble
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Light gradient boosting
- **Support Vector Regression**: SVM for regression

### 3. Model Evaluation

Models are evaluated using:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Cross-validation**: 5-fold cross-validation

### 4. Hyperparameter Tuning

- Grid search for optimal parameters
- Cross-validation for model selection
- Feature importance analysis

## 📈 Model Performance

The best performing model typically achieves:

- **R² Score**: ~0.85-0.90
- **RMSE**: ~$8,000-12,000
- **MAE**: ~$6,000-9,000
- **MAPE**: ~8-12%

## 🎮 Usage Examples

### Basic Prediction

```python
from src.prediction import SalaryPredictor

# Initialize predictor
predictor = SalaryPredictor()

# Employee data
employee_data = {
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

# Make prediction
result = predictor.predict_salary(employee_data)
print(f"Predicted Salary: {result['formatted_salary']}")
```

### Interactive Prediction

```bash
python src/prediction.py
```

### Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 📊 Key Insights

From the exploratory data analysis, we discovered:

1. **Department Impact**: Engineering and IT departments have the highest average salaries
2. **Experience Matters**: Strong positive correlation (0.7+) between experience and salary
3. **Education Premium**: PhD holders earn 30-40% more than high school graduates
4. **Location Factor**: San Francisco and New York offer 20-30% salary premiums
5. **Performance Correlation**: Performance rating shows moderate correlation with salary
6. **Career Progression**: Clear salary progression from Junior to Director levels

## 🛠️ Technical Details

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **plotly**: Interactive visualizations
- **xgboost/lightgbm**: Gradient boosting algorithms
- **jupyter**: Interactive notebooks

### Model Architecture

- **Input**: 15+ engineered features
- **Output**: Continuous salary value
- **Best Algorithm**: Random Forest or Gradient Boosting (varies by run)
- **Validation**: 5-fold cross-validation

## 🎯 Business Applications

This model can be used for:

- **Salary Benchmarking**: Compare salaries across different roles
- **Compensation Planning**: Set competitive salary ranges
- **Budget Forecasting**: Predict salary costs for new hires
- **Performance Evaluation**: Assess if current salaries align with market
- **Career Guidance**: Understand salary progression paths

## 🔮 Future Enhancements

Potential improvements:

- **Deep Learning**: Neural networks for complex patterns
- **Real-time Data**: Integration with live salary databases
- **Geographic Expansion**: Include more locations and cost-of-living adjustments
- **Industry Analysis**: Extend to multiple industries
- **Skill-based Features**: Include technical skills and certifications
- **Web Interface**: Create a web app for easy predictions

## 🤝 Contributing

This is an educational project. Feel free to:

- Experiment with different algorithms
- Add new features to the dataset
- Improve the preprocessing pipeline
- Enhance visualizations
- Add more evaluation metrics

## 📝 Learning Objectives

By working with this project, you will learn:

- Complete ML pipeline development
- Data preprocessing techniques
- Feature engineering strategies
- Model selection and evaluation
- Hyperparameter tuning
- Cross-validation techniques
- Model interpretation and insights
- Python ML ecosystem (pandas, sklearn, etc.)

## 🎓 Educational Value

This project covers:

- **Data Science**: EDA, feature engineering, statistical analysis
- **Machine Learning**: Regression algorithms, model evaluation
- **Software Engineering**: Modular code, documentation, testing
- **Business Intelligence**: Insights generation, reporting

## 📞 Support

For questions or issues:

1. Check the code comments and documentation
2. Review the Jupyter notebook for detailed explanations
3. Experiment with different parameters
4. Refer to the scikit-learn documentation

## 🏆 Acknowledgments

- **Dataset**: Synthetically generated for educational purposes
- **Algorithms**: Implemented using scikit-learn and other open-source libraries
- **Inspiration**: Real-world salary prediction challenges in HR analytics

---

**Happy Learning! 🚀**

_This project is designed to be a comprehensive learning experience in machine learning and data science. Feel free to modify, extend, and experiment with the code to deepen your understanding._
