import streamlit as st
import pandas as pd
import joblib

# Loading trained model
model = joblib.load("Carbon Fotprints Prediction.pkl")


st.title("Carbon Footprint Prediction for Industries")
st.write("Predict the estimated CO₂ emissions based on industry inputs")

# Input fields
electricity_usage = st.number_input("Electricity Usage (kWh)")
fuel_type = st.number_input("Fuel Type", ["Coal : 3", "Diesel: 2", "Natural Gas: 1)", "Renewables: 0"])
production_volume = st.number_input("Production Volume (Tons)")
efficiency = st.number_input("Efficiency Rating")

# Prediction button
if st.button("Predict Carbon Emission"):
    # Prepare input data
    input_data = pd.DataFrame([{
        "Electricity_Usage_kWh": electricity_usage,
        "Fuel_Type": fuel_type,
        "Production_Volume_Tons": production_volume,
        "Efficiency_Rating": efficiency
    }])


    # Prediction
    prediction = model.predict(input_data)

    st.success(f" Estimated Carbon Emission: **{prediction:.2f} kg CO₂**")
