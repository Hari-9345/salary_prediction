import streamlit as st
import pandas as pd
from train_model import train_model

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("Salary Prediction App")
st.write("Predict whether a person's income is **>50K or <=50K**")

# Load model (trained once and cached)
@st.cache_resource
def load_model():
    model = train_model()
    return model

model = load_model()

st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)

education = st.sidebar.selectbox(
    "Education",
    ["Bachelors", "HS-grad", "Masters", "Some-college", "Assoc-acdm", "Assoc-voc", "Doctorate"]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv"
    ]
)

hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# Create dataframe for prediction
input_data = pd.DataFrame({
    "age": [age],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week]
})

if st.button("Predict Salary"):

    prediction = model.predict(input_data)[0]

    if prediction.strip() == ">50K":
        st.success(" Predicted Income: >50K")
    else:
        st.info(" Predicted Income: <=50K")

st.write("---")
st.caption("Machine Learning Model deployed with Streamlit")
