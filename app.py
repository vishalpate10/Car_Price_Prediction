import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Set background image (after importing st) ---
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

# Load trained model
with open("Final_Linear_Model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# App title
st.title("🚗 Car Price Prediction App")
st.markdown("### Enter car details to predict the selling price")

# --- Sidebar for user inputs ---
st.sidebar.header("Car Details")

# Input fields
make = st.sidebar.text_input("Car Make (e.g. Maruti, Hyundai, Tata)")
model_name = st.sidebar.text_input("Car Model (e.g. Swift, i20, Nexon)")
year = st.sidebar.number_input("Year of Manufacture", 1990, 2025, 2018)
kilometer = st.sidebar.number_input("Kilometers Driven", 0, 500000, 50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
location = st.sidebar.text_input("Location (e.g. Mumbai, Pune, Delhi)")
color = st.sidebar.text_input("Car Color", "White")
owner = st.sidebar.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
engine = st.sidebar.number_input("Engine (CC)", 600, 6000, 1500)
max_power = st.sidebar.number_input("Max Power (BHP)", 30, 600, 80)
max_torque = st.sidebar.number_input("Max Torque (Nm)", 50, 800, 120)
drivetrain = st.sidebar.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])
length = st.sidebar.number_input("Length (mm)", 3000, 6000, 4000)
width = st.sidebar.number_input("Width (mm)", 1200, 2500, 1700)
height = st.sidebar.number_input("Height (mm)", 1200, 2500, 1500)
seating_capacity = st.sidebar.slider("Seating Capacity", 2, 10, 5)
fuel_tank = st.sidebar.number_input("Fuel Tank Capacity (Litres)", 20, 120, 45)

# --- Prepare Input Data ---
input_data = pd.DataFrame({
    "Make": [make],
    "Model": [model_name],
    "Year": [year],
    "Kilometer": [kilometer],
    "Fuel Type": [fuel_type],
    "Transmission": [transmission],
    "Location": [location],
    "Color": [color],
    "Owner": [owner],
    "Seller Type": [seller_type],
    "Engine": [engine],
    "Max Power": [max_power],
    "Max Torque": [max_torque],
    "Drivetrain": [drivetrain],
    "Length": [length],
    "Width": [width],
    "Height": [height],
    "Seating Capacity": [seating_capacity],
    "Fuel Tank Capacity": [fuel_tank]
})

# --- Encoding for categorical columns ---
cat_cols = ["Make", "Model", "Fuel Type", "Transmission", "Location", "Color", "Owner", "Seller Type", "Drivetrain"]
for col in cat_cols:
    input_data[col] = input_data[col].astype("category").cat.codes

# --- Prediction ---
st.write("### Input Summary")
st.dataframe(input_data)

if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted Car Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
