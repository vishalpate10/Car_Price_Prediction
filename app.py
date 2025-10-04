import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# --- Background Image ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t3.ftcdn.net/jpg/12/96/43/92/360_F_1296439246_C65lAcWsM7L5qs2lNT6CCuZHZXEIIe4a.jpg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model Safely ---
@st.cache_resource
def load_model():
    try:
        return load("Final_Linear_Model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Page Config ---
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Predict car prices based on car features using a trained Linear Regression model.")

# --- Input Fields ---
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Manufacturing Year", 1990, 2025, 2018)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Previous Owners", ["First", "Second", "Third", "Fourth & Above"])

with col2:
    present_price = st.number_input("Present Price (in lakhs)", 0.1, 50.0, 5.0)
    kms_driven = st.number_input("Kilometers Driven", 500, 500000, 30000)
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
    seating_capacity = st.slider("Seating Capacity", 2, 10, 5)

with col3:
    engine = st.number_input("Engine (CC)", 600, 6000, 1500)
    max_power = st.number_input("Max Power (BHP)", 30, 600, 80)
    max_torque = st.number_input("Max Torque (Nm)", 50, 800, 120)

# --- Categorical Encoding ---
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0
trans_manual = 1 if transmission == "Manual" else 0
seller_individual = 1 if seller_type == "Individual" else 0
owner_map = {"First": 0, "Second": 1, "Third": 2, "Fourth & Above": 3}
owner_num = owner_map[owner]

# --- Prepare Data for Prediction ---
input_data = pd.DataFrame({
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner_num],
    'Car_Age': [2025 - year],
    'Fuel_Type_Petrol': [fuel_petrol],
    'Fuel_Type_Diesel': [fuel_diesel],
    'Seller_Type_Individual': [seller_individual],
    'Transmission_Manual': [trans_manual],
    'Engine': [engine],
    'Max_Power': [max_power],
    'Max_Torque': [max_torque],
    'Seating_Capacity': [seating_capacity]
})

st.write("### Input Summary")
st.dataframe(input_data)

# --- Prediction ---
if st.button("Predict Car Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
