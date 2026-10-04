import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor ❤️", layout="centered")

# Title
st.title("❤️ Heart Disease Prediction App")
st.markdown("### Enter patient details below:")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app predicts the likelihood of heart disease using Machine Learning.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical
sex = 1 if sex == "Male" else 0

# Prediction button
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")