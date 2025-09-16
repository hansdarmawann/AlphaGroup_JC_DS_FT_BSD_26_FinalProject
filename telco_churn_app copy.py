import streamlit as st
import cloudpickle
import pandas as pd
import numpy as np

# ✅ Load model with cloudpickle
with open("Model/Model_Logreg_Telco_Churn_cloud.pkl", "rb") as f:
    model_bundle = cloudpickle.load(f)

model = model_bundle["model"]

st.title("📉 Telco Customer Churn Prediction")
st.header("🔍 Enter Customer Information")

# --- INPUT FORM ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])

# Numeric inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)

# Services
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Billing
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# --- PREDICTION ---
if st.button("Predict Churn"):
    try:
        # Strongly-typed input row
        row = {
            'gender': str(gender),
            'SeniorCitizen': int(1 if senior == "Yes" else 0),
            'Partner': str(partner),
            'Dependents': str(dependents),
            'tenure': np.float64(tenure),
            'MonthlyCharges': np.float64(monthly_charges),
            'TotalCharges': np.float64(total_charges),
            'PhoneService': str(phone_service),
            'MultipleLines': str(multiple_lines),
            'InternetService': str(internet_service),
            'OnlineSecurity': str(online_security),
            'OnlineBackup': str(online_backup),
            'DeviceProtection': str(device_protection),
            'TechSupport': str(tech_support),
            'StreamingTV': str(streaming_tv),
            'StreamingMovies': str(streaming_movies),
            'Contract': str(contract),
            'PaperlessBilling': str(paperless_billing),
            'PaymentMethod': str(payment_method)
        }

        input_data = pd.DataFrame([row])

        # Debug
        st.subheader("📋 Debug Info")
        st.write("✅ FINAL input to model:")
        st.write(input_data)
        st.write("✅ FINAL dtypes:")
        st.write(input_data.dtypes)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer Likely to Churn (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Customer Likely to Stay (Probability: {probability:.2%})")

    except Exception as e:
        st.exception(f"❌ Prediction failed: {e}")
