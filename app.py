import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

# App title and description
st.title("🚚 Delivery Delay Predictor")
st.write("Enter the delivery details below to predict whether a shipment will be delayed.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("delivery_delay_model.sav")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model file 'delivery_delay_model.sav'. Please ensure it is in the same directory. Details: {e}")
    st.stop()

# Form for input features
with st.form("prediction_form"):
    st.subheader("Delivery Characteristics")
    
    # Arrange inputs in columns for a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles/km)", min_value=0.0, max_value=1000.0, value=25.0)
        traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition Severity (1-5)", min_value=1, max_value=5, value=2)
        delivery_slot = st.selectbox("Delivery Slot (Time of Day)", options=[1, 2, 3], format_func=lambda x: f"Slot {x}")
        driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=5)
        num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=3)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=4)
        road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
        package_weight = st.number_input("Package Weight", min_value=0.0, max_value=500.0, value=15.0)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg/kml)", min_value=0.0, max_value=100.0, value=15.0)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (Minutes)", min_value=0, max_value=480, value=45)

    # Submit button
    submit_button = st.form_submit_button("Predict Delay")

if submit_button:
    # Create a DataFrame for model input using correct feature names
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }])
    
    # Generate prediction and probability scores
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display predictions
    st.markdown("--- ")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"🚨 **Prediction: Delayed!**")
        st.write(f"Confidence: **{probabilities[1]*100:.2f}%** risk of delay.")
    else:
        st.success(f"✅ **Prediction: On Time!**")
        st.write(f"Confidence: **{probabilities[0]*100:.2f}%** probability of arriving on time.")
