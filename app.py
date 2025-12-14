import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Project COOK – PDT Recommendation",
    layout="centered"
)

st.title("🍗 Project COOK – PDT Recommendation System")
st.caption("Human-in-the-loop assisted forecasting")

# --------------------------------------------------
# Load model safely
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("pdt_recommendation_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# UI Inputs
# --------------------------------------------------
date_input = st.date_input("📅 Select Date")

weather = st.selectbox("🌦 Weather", ["Cold", "Warm", "Hot", "Rainy"])
public_event = st.selectbox("🎉 Public / Store Event", ["No", "Yes"])
human_traffic = st.selectbox(
    "👥 Expected Customer Traffic",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

# --------------------------------------------------
# Encoders (must match training)
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

    # Step 1: Create feature dictionary
    feature_dict = {
        "OutOfStockBefore7pm": 0,
        "Human_Traffic": traffic_map[human_traffic],
        "Weather": weather_map[weather],
        "Public_Event": 1 if public_event == "Yes" else 0,
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "week_of_year": int(date.isocalendar().week)
    }

    # Step 2: Handle model expectations safely
    try:
        # Case A: Model knows feature names
        if hasattr(model, "feature_names_in_"):
            ordered_cols = list(model.feature_names_in_)
            input_array = np.array([[feature_dict[col] for col in ordered_cols]])

        # Case B: Model does NOT know names (trained on numpy)
        else:
            input_array = np.array([list(feature_dict.values())])

        # Final safety check
        if input_array.ndim != 2:
            raise ValueError("Input shape incorrect")

        prediction = model.predict(input_array)[0]

        st.success(
            f"✅ Recommended chickens to cook: **{int(round(prediction))}**"
        )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))

st.divider()
st.caption("Coles R&D | Project COOK | Human + AI Forecasting")
