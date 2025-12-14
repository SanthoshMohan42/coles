import streamlit as st
import pandas as pd
import pickle

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Coles PDT – Human in the Loop",
    layout="centered"
)

st.title("🍗 Coles PDT – Human-in-the-Loop Recommendation")
st.caption("Blending deli team insight with AI forecasting")

# =========================
# Load trained model
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================
# Date feature extraction
# (DO NOT pickle this)
# =========================
def extract_date_features(date):
    return {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "week_of_year": date.isocalendar()[1]
    }

# =========================
# Survey UI
# =========================
st.subheader("📝 Weekly Deli Insight Survey")

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
    "Public holiday or local event?",
    ["No", "Yes"]
)

# =========================
# Encoding (same as training)
# =========================
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

# =========================
# Prediction
# =========================
if st.button("🔮 Generate Recommendation"):
    date_features = extract_date_features(pd.to_datetime(date))

    input_df = pd.DataFrame({
        "Human_Traffic": [human_traffic_map[human_traffic_label]],
        "Weather": [weather_map[weather_label]],
        "Public_Event": [event_map[event_label]],
        "day_of_week": [date_features["day_of_week"]],
        "day_of_month": [date_features["day_of_month"]],
        "week_of_year": [date_features["week_of_year"]]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"✅ **Recommended chickens to cook:** {int(round(prediction))}"
    )

    st.caption(
        "Recommendation combines historical data with real deli team insight."
    )
