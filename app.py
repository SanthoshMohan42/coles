import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Project COOK – PDT Recommendation", layout="centered")

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("pdt_recommendation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Please upload it to the root directory.")
        st.stop()

@st.cache_resource
def load_date_preprocessor():
    try:
        with open("pdt_preprocessing.pkl", "rb") as f:
            dp = pickle.load(f)
        return dp
    except FileNotFoundError:
        st.error("❌ date_preprocessor.pkl not found.")
        st.stop()

model = load_model()
date_preprocessor = load_date_preprocessor()

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
    input_df = pd.DataFrame({
        "Date": [pd.to_datetime(date_input)],
        "Human_Traffic": [traffic_map[human_traffic]],
        "Weather": [weather_map[weather]],
        "Public_Event": [1 if public_event == "Yes" else 0],
        "OutOfStockBefore7pm": [0]
    })

    # Date features
    date_features = date_preprocessor.transform(input_df[["Date"]])
    date_features = pd.DataFrame(
        date_features,
        columns=["day_of_week", "day_of_month", "week_of_year"]
    )

    final_input = pd.concat(
        [date_features, input_df.drop(columns=["Date"])],
        axis=1
    )

    prediction = model.predict(final_input)[0]

    st.success(f"✅ Recommended chickens to cook: **{int(round(prediction))}**")
