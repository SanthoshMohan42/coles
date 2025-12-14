import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Project COOK – PDT Recommendation",
    layout="centered"
)

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("pdt_recommendation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🍗 Project COOK – Smart PDT Recommendation")
st.write("Human-in-the-loop assisted cooking recommendation system")

date_input = st.date_input("Select date")

weather = st.selectbox("Weather", ["Cold", "Warm", "Hot", "Rainy"])
public_event = st.selectbox("Public / Store Event", ["No", "Yes"])
human_traffic = st.selectbox(
    "Expected Customer Traffic",
    ["Much Lower", "Neutral", "Higher", "Much Higher"]
)

# -----------------------------
# Encoding (same as training)
# -----------------------------
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

if st.button("Predict Chicken Requirement"):

    # Convert date
    date = pd.to_datetime(date_input)

    # Date feature engineering (INLINE – no pickle)
    input_df = pd.DataFrame({
        "day_of_week": [date.dayofweek],
        "day_of_month": [date.day],
        "week_of_year": [date.isocalendar().week],
        "Human_Traffic": [traffic_map[human_traffic]],
        "Weather": [weather_map[weather]],
        "Public_Event": [1 if public_event == "Yes" else 0],
        "OutOfStockBefore7pm": [0]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"✅ Recommended chickens to cook: **{int(round(prediction))}**"
    )
