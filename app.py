import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load trained model ---
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Set background image ---
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

# --- Title ---
st.title("🚗 Car Price Prediction App")

# --- Make to Model mapping ---
make_model_map = {
    'Maruti': ['Swift', 'Baleno', 'Dzire', 'Ertiga'],
    'Hyundai': ['i10', 'i20', 'Creta', 'Verna'],
    'Honda': ['City', 'Jazz', 'Amaze'],
    'Toyota': ['Fortuner', 'Innova', 'Glanza'],
    'Ford': ['EcoSport', 'Figo', 'Endeavour'],
    'BMW': ['X1', 'X3', '3 Series', '5 Series']
}

# --- User Inputs ---
make = st.selectbox("Make", list(make_model_map.keys()))
model_car = st.selectbox("Model", make_model_map[make])
year = st.selectbox("Year of Manufacture", list(range(2000, 2025)))

# ✅ Number input for Kilometer Driven
km = st.number_input("Kilometers Driven", min_value=0, max_value=300000, step=5000, value=50000)

fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
location = st.selectbox("Location", ['Pune', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
seller = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
engine = st.selectbox("Engine (cc)", [800, 1000, 1197, 1498, 1591, 1998, 2200])
power = st.selectbox("Max Power (bhp)", [60, 75, 90, 100, 110, 120, 150])
drive = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD', '4WD'])
length = st.selectbox("Length (mm)", [3500, 3700, 3900, 4100, 4300])
width = st.selectbox("Width (mm)", [1500, 1600, 1700, 1800])
height = st.selectbox("Height (mm)", [1450, 1500, 1550, 1600])
seating = st.selectbox("Seating Capacity", [2, 4, 5, 6, 7, 8])
fuel_tank = st.selectbox("Fuel Tank Capacity (L)", [30, 35, 40, 45, 50])

# --- Derived feature ---
car_age = 2025 - year

# --- Create input dataframe ---
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

# --- Match model columns ---
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_columns]

# --- Prediction ---
if st.button("Predict Price"):
    price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹{price:,.0f}")
