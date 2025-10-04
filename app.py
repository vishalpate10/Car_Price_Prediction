import streamlit as st
import pandas as pd
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

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return load("Final_Linear_Model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Page Config ---
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Predict car prices using a trained Linear Regression model.")

# --- Dropdown Options (replace with your dataset unique values) ---
makes = ["Maruti", "Hyundai", "Tata", "Honda", "Mahindra"]
models_dict = {
    "Maruti": ["Swift", "Baleno", "Dzire"],
    "Hyundai": ["i20", "Verna", "Creta"],
    "Tata": ["Nexon", "Altroz", "Tiago"],
    "Honda": ["City", "Amaze", "WR-V"],
    "Mahindra": ["Thar", "XUV500", "Scorpio"]
}
fuel_types = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
transmissions = ["Manual", "Automatic"]
locations = ["Mumbai", "Pune", "Delhi", "Bangalore", "Chennai"]
owners = ["First", "Second", "Third", "Fourth & Above"]
seller_types = ["Dealer", "Individual", "Trustmark Dealer"]
drivetrains = ["FWD", "RWD", "AWD", "4WD"]

# --- Sidebar Inputs ---
st.sidebar.header("Enter Car Details")

make = st.sidebar.selectbox("Car Make", makes)

# Model dropdown depends on selected Make
model_name = st.sidebar.selectbox("Car Model", models_dict[make])

fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types)
transmission = st.sidebar.selectbox("Transmission Type", transmissions)
location = st.sidebar.selectbox("Location", locations)
owner = st.sidebar.selectbox("Owner Type", owners)
seller_type = st.sidebar.selectbox("Seller Type", seller_types)
drivetrain = st.sidebar.selectbox("Drivetrain", drivetrains)

# Numeric Inputs
year = st.sidebar.number_input("Year of Manufacture", 1990, 2025, 2018)
kilometer = st.sidebar.number_input("Kilometers Driven", 0, 500000, 50000)
length = st.sidebar.number_input("Length (mm)", 3000, 6000, 4000)
width = st.sidebar.number_input("Width (mm)", 1200, 2500, 1700)
height = st.sidebar.number_input("Height (mm)", 1200, 2500, 1500)
seating_capacity = st.sidebar.slider("Seating Capacity", 2, 10, 5)
fuel_tank = st.sidebar.number_input("Fuel Tank Capacity (Litres)", 20, 120, 45)
engine = st.sidebar.number_input("Engine (CC)", 600, 6000, 1500)
max_power = st.sidebar.number_input("Max Power (BHP)", 30, 600, 80)
max_rpm = st.sidebar.number_input("Max Power (RPM)", 1000, 10000, 6000)
present_price = st.sidebar.number_input("Present Price (in lakhs)", 0.1, 50.0, 5.0)

# --- Preprocess Inputs ---
owner_map = {"First": 0, "Second": 1, "Third": 2, "Fourth & Above": 3}

input_data = pd.DataFrame({
    "Make": [make],
    "Model": [model_name],
    "Kilometer": [kilometer],
    "Fuel Type": [fuel_type],
    "Transmission": [transmission],
    "Location": [location],
    "Owner": [owner_map[owner]],
    "Seller Type": [seller_type],
    "Drivetrain": [drivetrain],
    "Length": [length],
    "Width": [width],
    "Height": [height],
    "Seating Capacity": [seating_capacity],
    "Fuel Tank Capacity": [fuel_tank],
    "Car_Age": [2025 - year],
    "Engine (cc)": [engine],
    "Max Power (BHP)": [max_power],
    "Max Power (RPM)": [max_rpm],
    "Present_Price": [present_price]
})

# Encode categorical columns
cat_cols = ["Make", "Model", "Fuel Type", "Transmission", "Location", "Seller Type", "Drivetrain"]
for col in cat_cols:
    input_data[col] = input_data[col].astype("category").cat.codes

st.write("### Input Summary")
st.dataframe(input_data)

# --- Predict ---
if st.button("Predict Car Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
