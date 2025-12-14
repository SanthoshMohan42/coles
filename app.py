import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model & date preprocessor
# -----------------------------
with open("pdt_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pdt_preprocessing.pkl", "rb") as f:
    date_preprocessor = pickle.load(f)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Coles PDT – Human in the Loop", layout="centered")

st.title("🍗 Coles PDT – Human-in-the-Loop Recommendation")
st.caption("Deli team insights + AI forecasting")

st.subheader("Weekly Deli Insight Survey")

# -----------------------------
# Survey Inputs
# -----------------------------
date = st.date_input("Date to plan for")

human_traffic_label = st.selectbox(
    "Expected customer traffic",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

weather_label = st.selectbox(
    "Expected weather",
    ["Cold", "Warm", "Hot", "Rainy"]
)

event_label = st.radio(
    "Public holiday / local event?",
    ["No", "Yes"]
)

# -----------------------------
# Encoding (same as training)
# -----------------------------
human_traffic_map = {
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

event_map = {
    "No": 0,
    "Yes": 1
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("Generate Recommendation"):
    # Date preprocessing
    date_df = pd.DataFrame({"Date": [pd.to_datetime(date)]})
    date_features = date_preprocessor.transform(date_df)

    # Combine all features
    input_df = pd.DataFrame({
        "Human_Traffic": [human_traffic_map[human_traffic_label]],
        "Weather": [weather_map[weather_label]],
        "Public_Event": [event_map[event_label]],
        "day_of_week": [date_features[0][0]],
        "day_of_month": [date_features[0][1]],
        "week_of_year": [date_features[0][2]]
    })

    prediction = model.predict(input_df)[0]

    st.success(f" Recommended chickens to cook: **{int(round(prediction))}**")
