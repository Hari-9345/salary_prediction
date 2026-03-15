
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from train_model import train_model
st.set_page_config(
    page_title="AI Salary Prediction",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71");
background-size: cover;
background-attachment: fixed;
}

h1{
text-align:center;
font-size:55px;
font-weight:bold;
background: linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
color:transparent;
}

section[data-testid="stSidebar"]{
background: linear-gradient(180deg,#0f172a,#1e293b);
}

button{
background: linear-gradient(90deg,#2563eb,#1d4ed8) !important;
color:white !important;
border-radius:10px !important;
}

button:hover{
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:rgba(0,0,0,0.6);
padding:25px;border-radius:15px;
text-align:center;color:white;font-size:40px'>
 AI Salary Prediction Dashboard
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Dashboard Controls")

mode = st.sidebar.selectbox(
    "Theme Mode",
    ["Dark Mode","Light Mode"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)"
)



if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("adult_small.csv")

data.columns = data.columns.str.strip()

@st.cache_resource
def load_model():
    return train_model()

model = load_model()

tab1, tab2, tab3 = st.tabs(
    ["Data Analytics"," Prediction"," Dataset"]
)

with tab1:

    st.header("Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:

        income_counts = data["income"].value_counts().reset_index()
        income_counts.columns=["Income","Count"]

        fig = px.bar(
            income_counts,
            x="Income",
            y="Count",
            color="Income",
            title="Income Distribution"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        edu_counts = data["education"].value_counts().reset_index()
        edu_counts.columns=["Education","Count"]

        fig2 = px.bar(
            edu_counts.head(10),
            x="Education",
            y="Count",
            color="Education",
            title="Top Education Levels"
        )

        st.plotly_chart(fig2,use_container_width=True)

  

    st.subheader("Model Performance")

    features = ["age","education","occupation","hours.per.week"]

    X = data[features]
    y = data["income"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(y,y_pred)

    st.metric("Model Accuracy",f"{accuracy*100:.2f}%")

    cm = confusion_matrix(y,y_pred)

    fig3,ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j,i,cm[i,j],ha="center",va="center")

    st.pyplot(fig3)

with tab2:

    st.header("Predict Salary")

    col1,col2 = st.columns(2)

    with col1:

        age = st.slider("Age",18,65,30)

        education = st.selectbox(
            "Education",
            sorted(data["education"].unique())
        )

        occupation = st.selectbox(
            "Occupation",
            sorted(data["occupation"].unique())
        )

        hours = st.slider("Hours per Week",1,80,40)

    input_data = pd.DataFrame({
        "age":[age],
        "education":[education],
        "occupation":[occupation],
        "hours.per.week":[hours]
    })

    if st.button("Predict Salary"):

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)

        prob = max(probability[0])*100

        if ">50K" in str(prediction):

            st.success(" Predicted Income: >50K")

        else:

            st.info(" Predicted Income: <=50K")

        st.subheader("Confidence Level")

        st.progress(int(prob))

        st.write(f"Confidence: **{prob:.2f}%**")

with tab3:

    st.header("Dataset Preview")

    st.dataframe(data.head(10))

    st.subheader("Dataset Shape")

    st.write(data.shape)

    st.subheader("Columns")

    st.write(list(data.columns))

st.markdown("""
<hr>
<center style='color:white'>
 Professional Data Science Dashboard
<br>
Built with Streamlit • Plotly • Scikit-Learn
</center>
""",unsafe_allow_html=True)
```
