import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🏡 House Price Prediction App")
st.write("Predict the *median house value* based on the number of total rooms.")

# Input from user
total_rooms = st.number_input("Enter total number of rooms:", min_value=1)

# Predict on button click
if st.button("Predict"):
    # Scale the input
    scaled_input = scaler.transform([[total_rooms]])
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    # Display result
    st.success(f"Predicted Median House Value: ${prediction:,.2f}")