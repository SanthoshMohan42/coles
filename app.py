import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Project COOK – PDT Recommendation",
    layout="centered"
)

# --------------------------------------------------
# Feature order (MUST MATCH MODEL TRAINING)
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
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("pdt_recommendation_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍗 Project COOK – Smart PDT Recommendation")
st.caption("Human-in-the-loop cooking recommendation system")

date_input = st.date_input("Select date")

shredded = st.number_input(
    "Shredded chicken prepared (units)",
    min_value=0,
    value=12
)

out_of_stock = st.selectbox(
    "Out of stock before 7pm?",
    ["No", "Yes"]
)

human_traffic = st.selectbox(
    "Expected customer demand",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

weather = st.selectbox(
    "Weather condition",
    ["Cold", "Warm", "Hot", "Rainy"]
)

public_event = st.selectbox(
    "Public / Store Event",
    ["No", "Yes"]
)

# --------------------------------------------------
# Encoding maps (MUST MATCH TRAINING)
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
# Prediction logic
# --------------------------------------------------
if st.button("Predict chicken requirement"):
    date = pd.to_datetime(date_input)

    input_row = [
        shredded,
        1 if out_of_stock == "Yes" else 0,
        traffic_map[human_traffic],
        weather_map[weather],
        1 if public_event == "Yes" else 0,
        date.dayofweek,
        date.day,
        int(date.isocalendar().week)
    ]

    # Convert to NumPy array (shape = (1, 8))
    input_array = np.array(input_row, dtype=float).reshape(1, -1)

    # 🔍 Debug (can remove later)
    st.write("Input shape:", input_array.shape)
    st.write("Input values:", input_array)

    prediction = model.predict(input_array)[0]

    st.success(
        f"✅ Recommended chickens to cook: **{int(round(prediction))}**"
    )
