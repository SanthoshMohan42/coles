=import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Project COOK – PDT Recommendation", layout="centered")

# --------------------------------------------------
# LOCKED FEATURE ORDER (MUST MATCH TRAINING)
# --------------------------------------------------
FEATURE_COLUMNS = [
    "Shredded_chicken",
    "OutOfStockBefore7pm",
    "Human_Traffic",
    "Weather",
    "Public_Event",
    "day_of_week",
    "day_of_month",
    "week_of_year"
]

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
st.title("🍗 Project COOK – Smart PDT Recommendation")
st.caption("Human-in-the-loop cooking recommendation system")

date_input = st.date_input("Date")

shredded = st.number_input("Shredded chicken (units)", min_value=0, value=12)
out_of_stock = st.selectbox("Out of stock before 7pm?", ["No", "Yes"])
human_traffic = st.selectbox("Expected customer traffic",
                             ["Much Lower", "Neutral", "Higher", "Much Higher"])
weather = st.selectbox("Weather", ["Cold", "Warm", "Hot", "Rainy"])
public_event = st.selectbox("Public / Store Event", ["No", "Yes"])

# --------------------------------------------------
# Encoding maps (MATCH TRAINING)
# --------------------------------------------------
traffic_map = {"Much Lower": -2, "Neutral": 0, "Higher": 1, "Much Higher": 2}
weather_map = {"Cold": 1, "Warm": 0, "Hot": -1, "Rainy": 2}

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict chicken requirement"):
    date = pd.to_datetime(date_input)

    row = [
        shredded,
        1 if out_of_stock == "Yes" else 0,
        traffic_map[human_traffic],
        weather_map[weather],
        1 if public_event == "Yes" else 0,
        date.dayofweek,
        date.day,
        int(date.isocalendar().week)
    ]

    # Convert to NumPy array (THIS FIXES YOUR ERROR)
    input_array = np.array(row).reshape(1, -1)

    # Debug safety (remove later)
    st.write("Input array shape:", input_array.shape)
    st.write("Input values:", input_array)

    prediction = model.predict(input_array)[0]

    st.success(f"✅ Recommended chickens to cook: **{int(round(prediction))}**")
