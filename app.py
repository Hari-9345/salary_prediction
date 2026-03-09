import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
st.set_page_config(page_title="Employee Income Intelligence", layout="wide")
st.title(" Employee Income Intelligence Dashboard")
st.markdown("Predict employee income and explore workforce analytics.")
model = joblib.load("salary_pipeline.pkl")
data = pd.read_csv("adult.csv")
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(".", "-", regex=False)
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)
st.subheader(" Workforce Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", len(data))
col2.metric("High Income (>50K)", len(data[data["income"] == ">50K"]))
col3.metric("Low Income (<=50K)", len(data[data["income"] == "<=50K"]))
col4.metric("Average Age", round(data["age"].mean(),1))
st.divider()
tab1, tab2, tab3 = st.tabs(["Prediction", " Analytics", " Dataset"])
with tab1:
    st.subheader("Employee Salary Prediction")
    col1, col2, col3 = st.columns(3)
    age = col1.slider("Age", 18, 65, 30)
    workclass = col2.selectbox("Workclass", sorted(data["workclass"].unique()))
    fnlwgt = col3.number_input("Final Weight", value=200000)
    education = col1.selectbox("Education", sorted(data["education"].unique()))
    education_num = col2.number_input("Education Number", value=10)
    marital_status = col3.selectbox("Marital Status", sorted(data["marital-status"].unique()))
    occupation = col1.selectbox("Occupation", sorted(data["occupation"].unique()))
    relationship = col2.selectbox("Relationship", sorted(data["relationship"].unique()))
    race = col3.selectbox("Race", sorted(data["race"].unique()))
    sex = col1.selectbox("Gender", sorted(data["sex"].unique()))
    capital_gain = col2.number_input("Capital Gain", value=0)
    capital_loss = col3.number_input("Capital Loss", value=0)
    hours = col1.slider("Hours per Week", 1, 80, 40)
    native_country = col2.selectbox("Country", sorted(data["native-country"].unique()))
    input_data = pd.DataFrame({
        "age":[age],
        "workclass":[workclass],
        "fnlwgt":[fnlwgt],
        "education":[education],
        "education-num":[education_num],
        "marital-status":[marital_status],
        "occupation":[occupation],
        "relationship":[relationship],
        "race":[race],
        "sex":[sex],
        "capital-gain":[capital_gain],
        "capital-loss":[capital_loss],
        "hours-per-week":[hours],
        "native-country":[native_country]
    })
    if st.button(" Predict Income"):
        prediction = model.predict(input_data)
        if prediction[0] == ">50K":
            st.success("Predicted Income: HIGH (>50K)")
        else:
            st.warning("Income Level: LOW (<=50K)")
with tab2:
    st.subheader(" Workforce Analytics")
    col1, col2 = st.columns(2)
    fig1 = px.histogram(
        data,
        x="age",
        color="income",
        title="Income Distribution by Age"
    )

    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(
        data,
        x="income",
        y="hours-per-week",
        title="Working Hours vs Income"
    )

    col2.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    education_chart = data["education"].value_counts().reset_index()
    education_chart.columns = ["education", "count"]

    fig3 = px.bar(
        education_chart,
        x="education",
        y="count",
        title="Education Distribution"
    )

    col3.plotly_chart(fig3, use_container_width=True)

    country_chart = data["native-country"].value_counts().head(10).reset_index()
    country_chart.columns = ["country", "count"]

    fig4 = px.pie(
        country_chart,
        values="count",
        names="country",
        title="Top Countries"
    )

    col4.plotly_chart(fig4, use_container_width=True)


with tab3:

    st.subheader("Dataset Explorer")

    st.dataframe(data)

    st.subheader("Dataset Summary")

    st.write(data.describe())