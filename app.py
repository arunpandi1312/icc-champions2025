import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading saved models
import lightgbm as lgb

# Load the trained model
model = joblib.load("lightgbm_model.pkl")  # Ensure the model file is in the same directory

# Define the Streamlit UI
st.title("ICC Champions 2025 Prediction")

# Input Fields
team = st.text_input("Enter Team Name")
top_scorer = st.text_input("Enter Top Scorer Name")
wins = st.number_input("Enter Number of Wins", min_value=0, max_value=100)
net_run_rate = st.number_input("Enter Net Run Rate", min_value=-10.0, max_value=10.0, step=0.01)

# Prediction Button
if st.button("Predict Outcome"):
    # Create a DataFrame for model input
    input_data = pd.DataFrame([[team, top_scorer, wins, net_run_rate]],
                              columns=["Team", "Top_Scorer", "Wins", "NRR"])

    # Apply the same feature engineering as in training
    input_data["Wins × NRR"] = input_data["Wins"] * input_data["NRR"]

    # Make Prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Outcome: {'Win' if prediction[0] == 1 else 'Loss'}")
