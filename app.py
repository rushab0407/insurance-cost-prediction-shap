import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model and SHAP explainer
model = joblib.load("model.pkl")
explainer = joblib.load("shap_explainer.pkl")

st.title("🏥 Medical Cost Prediction with SHAP")

# User Inputs
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input data
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [1 if sex == "male" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0]
})

# Predict
prediction = model.predict(input_data)[0]
st.subheader(f"💰 Estimated Insurance Cost: ${prediction:,.2f}")

# SHAP explanation

shap_values = explainer.shap_values(input_data)

expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

fig, ax = plt.subplots()
shap.plots._waterfall.waterfall_legacy(
    expected_val,
    shap_values[0],
    input_data.iloc[0],
    show=False
)
st.pyplot(fig)
