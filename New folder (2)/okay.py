import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Health Premium Estimator",
    page_icon="🏥",
    layout="wide"
)

# Load trained files
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f7f3ff);
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #123c69, #2563eb);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(37,99,235,0.18);
}
.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0ecff;
    font-size: 16px;
}
.section {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e1e9f5;
    margin-bottom: 18px;
    box-shadow: 0 5px 18px rgba(15, 45, 90, 0.06);
}
.section h3 {
    color: #123c69;
    margin-top: 0;
}
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #6d5dfc);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
    transition: 0.2s;
}
.stButton > button:hover {
    color: white;
    border: 1px solid #2563eb;
    transform: translateY(-2px);
}
.result {
    background: linear-gradient(120deg, #0f766e, #10b981);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(16,185,129,0.18);
}
.result h2 {
    color: white;
    font-size: 22px;
}
.amount {
    font-size: 38px;
    font-weight: 800;
    color: white;
}
.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 13px;
}
div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <h1>🏥 Health Insurance Premium Estimator</h1>
    <p>Smart estimation • Simple inputs • Instant prediction</p>
    <p>Enter your details below to estimate your health insurance premium.</p>
</div>
""", unsafe_allow_html=True)

# Input form
with st.form("insurance_form"):
    st.markdown('<div class="section"><h3>👤 Personal Information</h3></div>',
                unsafe_allow_html=True)

    inputs = {}
    col1, col2, col3 = st.columns(3)

    for i, f in enumerate(features):
        with [col1, col2, col3][i % 3]:
            if f == "sex":
                choice = st.selectbox(
                    "Gender",
                    ["Female (0)", "Male (1)"],
                    key=f
                )
                inputs[f] = 1 if choice.endswith("(1)") else 0

            elif f == "smoker":
                choice = st.selectbox(
                    "Smoking Status",
                    ["No (0)", "Yes (1)"],
                    key=f
                )
                inputs[f] = 1 if choice.endswith("(1)") else 0

            elif f == "age":
                inputs[f] = st.number_input(
                    "Age (years)", min_value=18, max_value=100,
                    value=30, step=1, key=f
                )

            elif f == "bmi":
                inputs[f] = st.number_input(
                    "BMI", min_value=10.0, max_value=80.0,
                    value=25.0, step=0.5, key=f
                )

            elif f == "children":
                inputs[f] = st.number_input(
                    "Number of Children", min_value=0,
                    max_value=10, value=0, step=1, key=f
                )

            elif f == "region":
                choice = st.selectbox(
                    "Region",
                    ["Northeast", "Northwest", "Southeast", "Southwest"],
                    key=f
                )
                # This numeric mapping must match your training preprocessing.
                inputs[f] = choice

            else:
                inputs[f] = st.number_input(
                    f.replace("_", " ").title(),
                    value=0.0,
                    key=f
                )

    submitted = st.form_submit_button("✨ Calculate My Premium")

if submitted:
    try:
        X_new = pd.DataFrame([inputs])[features]
        X_new = pd.get_dummies(X_new)

        # Align encoded columns if the model was trained with feature names.
        if hasattr(scaler, "feature_names_in_"):
            X_new = X_new.reindex(
                columns=scaler.feature_names_in_,
                fill_value=0
            )

        X_scaled = scaler.transform(X_new)
        prediction = model.predict(X_scaled)[0]

        st.markdown(f"""
        <div class="result">
            <h2>🎉 Your Estimated Insurance Premium</h2>
            <div class="amount">{prediction:,.2f}</div>
            <p>Estimated amount based on the information provided</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Note: This is a model-based estimate, not an official insurance quote."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check that your input columns and preprocessing match the model's training setup."
        )

st.markdown("""
<div class="footer">
    Health Insurance Premium Estimator | Machine Learning Project
</div>
""", unsafe_allow_html=True)