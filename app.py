import streamlit as st
import pickle
import pandas as pd

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
    try:
        with open("pdt_recommendation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ pdt_recommendation_model.pkl not found in root directory.")
        st.stop()

model = load_model()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("🍗 Project COOK – Smart PDT Recommendation")
st.caption("Human-in-the-Loop Assisted Chicken Cooking Forecast")

st.markdown("---")

# Inputs
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
# Encoding maps (MUST match training)
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
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Chicken Requirement"):

    date = pd.to_datetime(date_input)

    # Build feature row (EXACT training features)
    input_df = pd.DataFrame({
        "OutOfStockBefore7pm": [0],
        "Human_Traffic": [traffic_map[human_traffic]],
        "Weather": [weather_map[weather]],
        "Public_Event": [1 if public_event == "Yes" else 0],
        "day_of_week": [date.dayofweek],
        "day_of_month": [date.day],
        "week_of_year": [int(date.isocalendar().week)]
    })

    # FORCE correct column order
    expected_columns = [
        "OutOfStockBefore7pm",
        "Human_Traffic",
        "Weather",
        "Public_Event",
        "day_of_week",
        "day_of_month",
        "week_of_year"
    ]

    input_df = input_df[expected_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    st.success(
        f"✅ **Recommended chickens to cook:** {int(round(prediction))}"
    )

    st.caption(
        "Recommendation adjusted using human inputs + calendar context."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Project COOK | Coles R&D | Human + AI Forecasting")
