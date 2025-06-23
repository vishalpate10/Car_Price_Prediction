import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- 🔹 Set Background Image with CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=983&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- 🔹 App Title ---
st.title("🚗 Car Price Prediction App")

# --- 🔹 User Inputs ---
make = st.selectbox("Make", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW'])
model_car = st.text_input("Model (e.g., Swift, i20)")
year = st.number_input("Manufacturing Year", min_value=1995, max_value=2025, value=2015)
km = st.number_input("Kilometers Driven", value=50000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
location = st.selectbox("Location", ['Pune', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad'])
owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
seller = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
engine = st.number_input("Engine (cc)", value=1197)
power = st.number_input("Max Power (bhp)", value=80.0)
drive = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD', '4WD'])
length = st.number_input("Length (mm)", value=3995)
width = st.number_input("Width (mm)", value=1680)
height = st.number_input("Height (mm)", value=1500)
seating = st.number_input("Seating Capacity", value=5)
fuel_tank = st.number_input("Fuel Tank Capacity (L)", value=40)

# --- 🔹 Derived Feature ---
car_age = 2025 - year

# --- 🔹 Create Input DataFrame ---
input_data = pd.DataFrame({
    'Kilometer': [km],
    'Engine': [engine],
    'Max Power': [power],
    'Length': [length],
    'Width': [width],
    'Height': [height],
    'Seating Capacity': [seating],
    'Fuel Tank Capacity': [fuel_tank],
    'Car_Age': [car_age],
    f'Make_{make}': [1],
    f'Fuel Type_{fuel}': [1],
    f'Transmission_{transmission}': [1],
    f'Location_{location}': [1],
    f'Owner_{owner}': [1],
    f'Seller Type_{seller}': [1],
    f'Drivetrain_{drive}': [1],
})

# --- 🔹 Fill Missing Columns ---
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# --- 🔹 Reorder Columns to Match Model ---
input_data = input_data[model_columns]

# --- 🔹 Predict Price ---
if st.button("Predict Price"):
    price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹{price:,.0f}")
