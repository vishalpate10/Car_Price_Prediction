import streamlit as st
import pandas as pd
import numpy as np
from joblib import load  # ✅ safer than pickle for sklearn models

# ------------------------------
# Load trained model safely
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = load("Final_Linear_Model.joblib")  # use .joblib instead of .pkl
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return model

model = load_model()

# ------------------------------
# Streamlit page configuration
# ------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction App")
st.markdown("Predict car prices based on features using a trained Linear Regression model.")

# ------------------------------
# Input fields for user data
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Car Manufacturing Year", 2000, 2025, 2018)

with col2:
    present_price = st.number_input("Present Price (in lakhs)", 0.1, 50.0, 5.0)

with col3:
    kms_driven = st.number_input("Kms Driven", 500, 200000, 30000)

owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# ------------------------------
# Convert categorical inputs to numeric
# ------------------------------
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0
seller_individual = 1 if seller_type == "Individual" else 0
trans_manual = 1 if transmission == "Manual" else 0

# ------------------------------
# Prepare input data for prediction
# ------------------------------
input_data = pd.DataFrame({
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner],
    'Car_Age': [2025 - year],
    'Fuel_Type_Petrol': [fuel_petrol],
    'Fuel_Type_Diesel': [fuel_diesel],
    'Seller_Type_Individual': [seller_individual],
    'Transmission_Manual': [trans_manual]
})

# ------------------------------
# Predict price
# ------------------------------
if st.button("Predict Car Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
