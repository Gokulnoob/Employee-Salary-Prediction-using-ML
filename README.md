# 💰 Employee Salary Prediction

A machine learning project that predicts employee salaries and ships as an interactive **Streamlit** web app.

## 📁 Project Structure

```
Employee Salary Prediction/
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── data/                          # Datasets (generated)
├── models/                        # Trained model artifacts
├── notebooks/                     # Jupyter EDA notebook
│   └── 01_exploratory_data_analysis.ipynb
└── src/                           # ML pipeline scripts
    ├── create_dataset.py          # Generate sample data
    ├── data_preprocessing.py      # Clean & feature‑engineer
    ├── model_training.py          # Train & compare models
    ├── model_evaluation.py        # Evaluation charts
    └── prediction.py              # Prediction utilities
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ML pipeline (only needed once)
python src/create_dataset.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py

# 3. Launch the Streamlit app
streamlit run app.py
```

## 🖥️ Streamlit App Pages

| Page                     | Description                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| **🎯 Salary Predictor**  | Enter employee details and get an instant salary estimate with a 95 % confidence interval.     |
| **📊 Data Explorer**     | Visualise salary distributions, department breakdowns, and experience‑vs‑salary scatter plots. |
| **🏆 Model Performance** | Compare R², RMSE, MAE, MAPE across all trained models; view feature importance.                |

## 📊 Dataset Features

| Feature              | Type        | Description                                                       |
| -------------------- | ----------- | ----------------------------------------------------------------- |
| `age`                | Numerical   | Employee age (22–65)                                              |
| `years_experience`   | Numerical   | Work experience (0–35 yrs)                                        |
| `department`         | Categorical | Engineering, Sales, Marketing, HR, Finance, Operations, IT        |
| `job_title`          | Categorical | Junior, Mid‑Level, Senior, Lead, Manager, Director                |
| `education_level`    | Categorical | High School, Bachelor, Master, PhD                                |
| `location`           | Categorical | New York, San Francisco, Chicago, Austin, Boston, Seattle, Remote |
| `performance_rating` | Numerical   | 1.0–5.0                                                           |
| `overtime_hours`     | Numerical   | Monthly overtime (0–20)                                           |
| `projects_completed` | Numerical   | 0–15                                                              |
| `salary`             | Numerical   | **Target** — annual salary (USD)                                  |

## 🔬 ML Pipeline

1. **Preprocessing** — missing‑value imputation, feature engineering, label encoding, scaling.
2. **Training** — Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR.
3. **Evaluation** — R², RMSE, MAE, MAPE, 5‑fold cross‑validation, grid‑search tuning.
4. **Deployment** — best model served through the Streamlit app.

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file path to `app.py`.
4. Click **Deploy**.

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
