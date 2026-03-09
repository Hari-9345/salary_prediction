import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from train_model import train_model

st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")

st.title(" Salary Prediction Dashboard")
data = pd.read_csv("adult_small.csv")
data.columns = data.columns.str.strip()
@st.cache_resource
def load_model():
    return train_model()

model = load_model()

tab1, tab2, tab3 = st.tabs(["Charts", "Prediction", "Dataset Preview"])

with tab1:

    st.header(" Data Analytics")

    col1, col2 = st.columns(2)

    with col1:

        income_counts = data["income"].value_counts().reset_index()
        income_counts.columns = ["Income", "Count"]

        fig = px.bar(
            income_counts,
            x="Income",
            y="Count",
            color="Income",
            title="Income Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        edu_counts = data["education"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Count"]

        fig2 = px.bar(
            edu_counts.head(10),
            x="Education",
            y="Count",
            color="Education",
            title="Top Education Levels"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Model Performance")

    features = ["age", "education", "occupation", "hours.per.week"]

    X = data[features]
    y = data["income"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    cm = confusion_matrix(y, y_pred)

    fig3, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig3)

with tab2:

    st.header(" Salary Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 65, 30)

        education = st.selectbox(
            "Education",
            sorted(data["education"].unique())
        )

        occupation = st.selectbox(
            "Occupation",
            sorted(data["occupation"].unique())
        )

        hours_per_week = st.slider("Hours per Week", 1, 80, 40)

    input_data = pd.DataFrame({
        "age": [age],
        "education": [education],
        "occupation": [occupation],
        "hours.per.week": [hours_per_week]
    })

    if st.button("Predict Salary"):

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)

        prob = max(probability[0]) * 100

        if ">50K" in str(prediction):

            st.success(" Predicted Income: >50K")

        else:

            st.info(" Predicted Income: <=50K")

        st.subheader("Prediction Confidence")

        st.progress(int(prob))

        st.write(f"Confidence: **{prob:.2f}%**")
    st.subheader("Feature Importance")

    try:

        feature_importance = model.named_steps["model"].feature_importances_

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        })

        importance_df = importance_df.sort_values("Importance", ascending=False)

        fig4 = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(fig4, use_container_width=True)

    except:
        st.info("Feature importance not available.")
with tab3:

    st.header(" Dataset Preview")

    st.write("First 10 rows of dataset")

    st.dataframe(data.head(10))

    st.subheader("Dataset Shape")

    st.write(data.shape)

    st.subheader("Column Names")

    st.write(list(data.columns))

st.write("---")
st.caption("Professional Machine Learning Dashboard • Streamlit")