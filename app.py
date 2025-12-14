import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Project COOK – PDT Recommendation",
    layout="centered"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("pdt_recommendation_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍗 Project COOK – PDT Recommendation System")
st.caption("Human-in-the-loop assisted cooking forecast")

st.divider()

date_input = st.date_input("📅 Select Date")

weather = st.selectbox(
    "🌦 Weather",
    ["Cold", "Warm", "Hot", "Rainy"]
)

public_event = st.selectbox(
    "🎉 Public / Store Event",
    ["No", "Yes"]
)

human_traffic = st.selectbox(
    "👥 Expected Customer Traffic",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

# --------------------------------------------------
# Encoding (MUST match training)
# --------------------------------------------------
traffic_map = {
    "Much Lower": -2,
    "Neutral": 0,
    "Higher": 1,
    "Much Higher": 2
}

weather_map = {
    "Cold": 1,
    "Warm": 0,
    "Hot": -1,
    "Rainy": 2
}

# --------------------------------------------------
# Predict
# --------------------------------------------------
if st.button("🔮 Predict Chicken Requirement"):

    date = pd.to_datetime(date_input)

    # Build input row EXACTLY like training
    input_data = [
        0,                                   # OutOfStockBefore7pm
        traffic_map[human_traffic],          # Human_Traffic
        weather_map[weather],                # Weather
        1 if public_event == "Yes" else 0,   # Public_Event
        date.dayofweek,                      # day_of_week
        date.day,                            # day_of_month
        int(date.isocalendar().week)         # week_of_year
    ]

    # Convert to NumPy (THIS FIXES YOUR ERROR)
    input_array = np.array(input_data).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)[0]

    st.success(
        f"✅ **Recommended chickens to cook:** {int(round(prediction))}"
    )

    st.caption("Prediction combines calendar context + human input")

st.divider()
st.caption("Project COOK | Coles R&D | Human + AI Forecasting")
