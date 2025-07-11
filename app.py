import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score



with open('model.pkl', 'rb') as file:
    model =  pickle.load(file)

# with open('model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

st.set_page_config(page_title="Personality Predictor", layout="centered")
st.title("🧠 Extrovert vs. Introvert Personality Predictor")

st.markdown("""
Welcome to the **Extrovert vs. Introvert Personality Traits Analyzer**!  
Fill in your behavioral details below and click **Predict** to discover your likely personality type.

---

**About the Dataset**  
This model is built on a dataset of 18524 individuals analyzing behavioral and social traits such as:
- Time spent alone
- Stage fright
- Social event habits
- Online behavior

---
""")

# -------------------------------
# Sidebar Input
# -------------------------------
st.sidebar.header("Input Your Behavior Data")

time_alone = st.sidebar.slider("Time spent alone daily (hours)", 0, 11, 3)
stage_fear = st.sidebar.selectbox("Do you have stage fear?", ['Yes', 'No'])
social_events = st.sidebar.slider("Social event attendance (0–10)", 0, 10, 5)
going_out = st.sidebar.slider("Going outside frequency (0–7)", 0, 7, 4)
drained = st.sidebar.selectbox("Do you feel drained after socializing?", ['Yes', 'No'])
friends = st.sidebar.slider("Number of close friends (0–15)", 0, 15, 7)
posts = st.sidebar.slider("Social media post frequency (0–10)", 0, 10, 4)

# -------------------------------
# Data Preparation
# -------------------------------
input_dict = {
    'Time_spent_Alone': time_alone,
    'Stage_fear': 1 if stage_fear == 'Yes' else 0,
    'Social_event_attendance': social_events,
    'Going_outside': going_out,
    'Drained_after_socializing': 1 if drained == 'Yes' else 0,
    'Friends_circle_size': friends,
    'Post_frequency': posts
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🎯 Predict Personality"):
    prediction = model.predict(input_df)[0]
    label = "Extrovert" if prediction == 1 else "Introvert"
    st.success(f"🧬 Based on your input, you are likely an **{label}**!")
    st.markdown("---")
    st.markdown("Feel free to adjust the values and try again.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr style="border:0.5px solid #ccc">
👨‍🔬 Built using **Streamlit** & **Scikit-learn** | Dataset from behavioral research on personality types.
""", unsafe_allow_html=True)
