import streamlit as st
import pandas as pd
from train_model import train_model

st.set_page_config(page_title="Salary Prediction", layout="wide")

st.title("💰 Employee Salary Prediction")
st.write("Predict whether income is **>50K or <=50K**")

# Train model
@st.cache_resource
def load_model():
    model = train_model()
    return model

model = load_model()

st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox(
    "Education",
    ["Bachelors", "HS-grad", "Masters", "Some-college", "Assoc", "Doctorate"]
)
occupation = st.sidebar.selectbox(
    "Occupation",
    ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty"]
)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)

input_data = pd.DataFrame({
    "age": [age],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week]
})

if st.button("Predict Salary"):

    prediction = model.predict(input_data)[0]

    if prediction == ">50K":
        st.success("Predicted Income: >50K 💰")
    else:
        st.info("Predicted Income: <=50K")

st.write(" ")
st.write("---")
st.caption("Machine Learning Salary Predictor using Streamlit")
