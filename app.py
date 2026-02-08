"""
Employee Salary Prediction — Streamlit App
Interactive ML dashboard for predicting employee salaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ---------------------------------------------------------------------------
# Helper – cached data / model loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    path = "data/employee_salary_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    path = "models/best_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_preprocessor():
    path = "models/preprocessor.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_model_scores():
    path = "models/model_scores.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ---------------------------------------------------------------------------
# Prediction helper (mirrors src/prediction.py logic)
# ---------------------------------------------------------------------------
def predict_salary(input_data: dict, model, preprocessor) -> dict:
    """Run the full preprocess → predict pipeline for a single employee."""
    df = pd.DataFrame([input_data])

    # Feature engineering (same as data_preprocessing.py)
    df["experience_age_ratio"] = df["years_experience"] / df["age"].clip(lower=1)
    df["performance_score"] = (df["performance_rating"] - 1) / 4
    df["productivity_score"] = df["projects_completed"] / (df["years_experience"] + 1)
    df["overtime_ratio"] = df["overtime_hours"] / 160

    # Encode categoricals
    for col, encoder in preprocessor["label_encoders"].items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                df[col] = -1

    # Ensure all feature columns present
    for col in preprocessor["feature_columns"]:
        if col not in df.columns:
            df[col] = 0
    df = df[preprocessor["feature_columns"]]

    # Scale
    df_scaled = preprocessor["scaler"].transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=preprocessor["feature_columns"])

    prediction = model.predict(df_scaled)[0]

    # Confidence interval for tree‑ensemble models
    interval = None
    if hasattr(model, "estimators_"):
        try:
            preds = np.array([est.predict(df_scaled)[0] for est in model.estimators_])
            std = np.std(preds)
            interval = (prediction - 1.96 * std, prediction + 1.96 * std)
        except Exception:
            pass

    return {"salary": round(prediction, 2), "interval": interval}


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
data = load_dataset()
model = load_model()
preprocessor = load_preprocessor()
model_scores = load_model_scores()

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "IT"]
JOB_TITLES = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]
LOCATIONS = ["New York", "San Francisco", "Chicago", "Austin", "Boston", "Seattle", "Remote"]

# ---------------------------------------------------------------------------
# Sidebar — Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/money-bag.png", width=80)
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["🎯 Salary Predictor", "📊 Data Explorer", "🏆 Model Performance"],
    )
    st.divider()
    st.caption("Employee Salary Prediction · ML Project")

# ===================================================================
# PAGE 1 – Salary Predictor
# ===================================================================
if page == "🎯 Salary Predictor":
    st.title("🎯 Employee Salary Predictor")
    st.markdown("Fill in the employee details below and click **Predict** to get an estimated salary.")

    if model is None or preprocessor is None:
        st.error("⚠️ Trained model not found. Please run the ML pipeline first:\n"
                 "```\npython src/create_dataset.py\npython src/data_preprocessing.py\npython src/model_training.py\n```")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 30)
        years_experience = st.slider("Years of Experience", 0, 40, 5)
        performance_rating = st.slider("Performance Rating", 1.0, 5.0, 3.5, step=0.1)

    with col2:
        department = st.selectbox("Department", DEPARTMENTS)
        job_title = st.selectbox("Job Title", JOB_TITLES)
        education_level = st.selectbox("Education Level", EDUCATION_LEVELS)

    with col3:
        location = st.selectbox("Location", LOCATIONS)
        overtime_hours = st.number_input("Overtime Hours / Month", 0, 100, 5)
        projects_completed = st.number_input("Projects Completed", 0, 50, 8)

    if st.button("💰 Predict Salary", type="primary", use_container_width=True):
        input_data = {
            "age": age,
            "years_experience": years_experience,
            "department": department,
            "job_title": job_title,
            "education_level": education_level,
            "location": location,
            "performance_rating": performance_rating,
            "overtime_hours": overtime_hours,
            "projects_completed": projects_completed,
        }

        result = predict_salary(input_data, model, preprocessor)
        salary = result["salary"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Salary", f"${salary:,.0f}")

        if result["interval"]:
            lo, hi = result["interval"]
            m2.metric("Lower Bound (95 %)", f"${lo:,.0f}")
            m3.metric("Upper Bound (95 %)", f"${hi:,.0f}")

        # Quick recommendations
        tips: list[str] = []
        if years_experience < 5:
            tips.append("📈 Gain more experience to unlock higher salary bands.")
        if performance_rating < 4.0:
            tips.append("⭐ Improving your performance rating can significantly boost pay.")
        if education_level in ("High School", "Bachelor"):
            tips.append("🎓 Consider pursuing a higher degree or certifications.")
        if projects_completed < 5:
            tips.append("📂 Take on more projects to demonstrate productivity.")

        if tips:
            st.subheader("💡 Recommendations")
            for t in tips:
                st.write(t)

# ===================================================================
# PAGE 2 – Data Explorer
# ===================================================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")

    if data is None:
        st.error("⚠️ Dataset not found. Run `python src/create_dataset.py` first.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees", f"{len(data):,}")
    k2.metric("Avg Salary", f"${data['salary'].mean():,.0f}")
    k3.metric("Median Salary", f"${data['salary'].median():,.0f}")
    k4.metric("Departments", data["department"].nunique())

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Salary Distribution", "Department Analysis", "Experience vs Salary", "Dataset"]
    )

    # --- Tab 1: Distribution ---
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data["salary"], nbinsx=40, marker_color="#667eea", opacity=0.75,
        ))
        mean_sal = data["salary"].mean()
        fig.add_vline(x=mean_sal, line_dash="dash", line_color="red",
                      annotation_text=f"Mean ${mean_sal:,.0f}")
        fig.update_layout(
            title="Salary Distribution",
            xaxis_title="Annual Salary ($)",
            yaxis_title="Count",
            template="plotly_white", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Department ---
    with tab2:
        dept = data.groupby("department")["salary"].agg(["mean", "median", "count"]).reset_index()
        dept.columns = ["Department", "Mean Salary", "Median Salary", "Count"]
        dept = dept.sort_values("Mean Salary", ascending=True)

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(dept, y="Department", x="Mean Salary", orientation="h",
                          color="Mean Salary", color_continuous_scale="Viridis",
                          title="Average Salary by Department", text_auto="$,.0f")
            fig1.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(dept, y="Department", x="Count", orientation="h",
                          color="Count", color_continuous_scale="Blues",
                          title="Employee Count by Department", text_auto=True)
            fig2.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 3: Experience ---
    with tab3:
        sample = data.sample(min(2000, len(data)), random_state=42)
        fig3 = px.scatter(
            sample, x="years_experience", y="salary",
            color="department", size="performance_rating",
            title="Years of Experience vs Salary",
            labels={"years_experience": "Years of Experience", "salary": "Salary ($)"},
            opacity=0.6,
        )
        fig3.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 4: Raw data ---
    with tab4:
        st.dataframe(data, use_container_width=True, height=400)

# ===================================================================
# PAGE 3 – Model Performance
# ===================================================================
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance")

    if model_scores is None:
        st.error("⚠️ Model scores not found. Run the training pipeline first.")
        st.stop()

    scores_df = pd.DataFrame(model_scores).T.round(4)
    scores_df.index.name = "Model"

    # Best model highlight
    best_name = scores_df["R2"].idxmax()
    best_r2 = scores_df.loc[best_name, "R2"]
    best_rmse = scores_df.loc[best_name, "RMSE"]

    b1, b2, b3 = st.columns(3)
    b1.metric("Best Model", best_name.replace("_", " ").title())
    b2.metric("R² Score", f"{best_r2:.4f}")
    b3.metric("RMSE", f"${best_rmse:,.0f}")

    st.divider()

    tab_a, tab_b, tab_c = st.tabs(["R² Comparison", "Error Metrics", "Full Table"])

    with tab_a:
        sorted_df = scores_df.sort_values("R2", ascending=True)
        fig = px.bar(
            sorted_df, y=sorted_df.index, x="R2", orientation="h",
            color="R2", color_continuous_scale="Viridis",
            title="R² Score by Model (higher is better)",
            text_auto=".4f",
        )
        fig.update_layout(template="plotly_white", height=450, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=("RMSE", "MAE", "MAPE (%)"))
        sorted_df = scores_df.sort_values("RMSE", ascending=False)
        fig2.add_trace(go.Bar(y=sorted_df.index, x=sorted_df["RMSE"], orientation="h",
                              marker_color="#ef5350", name="RMSE"), row=1, col=1)
        fig2.add_trace(go.Bar(y=sorted_df.index, x=sorted_df["MAE"], orientation="h",
                              marker_color="#66bb6a", name="MAE"), row=1, col=2)
        fig2.add_trace(go.Bar(y=sorted_df.index, x=sorted_df["MAPE"], orientation="h",
                              marker_color="#ffa726", name="MAPE"), row=1, col=3)
        fig2.update_layout(template="plotly_white", height=450, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_c:
        st.dataframe(scores_df.style.highlight_max(subset=["R2"], color="#c8e6c9")
                     .highlight_min(subset=["RMSE", "MAE", "MAPE"], color="#c8e6c9"),
                     use_container_width=True)

    # Feature importance (if available)
    if model and hasattr(model, "feature_importances_") and preprocessor:
        st.divider()
        st.subheader("🔍 Feature Importance")
        feat_df = pd.DataFrame({
            "Feature": preprocessor["feature_columns"],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(feat_df, y="Feature", x="Importance", orientation="h",
                        color="Importance", color_continuous_scale="Tealgrn",
                        title="Feature Importance (Best Model)")
        fig_fi.update_layout(template="plotly_white", height=500, yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)
