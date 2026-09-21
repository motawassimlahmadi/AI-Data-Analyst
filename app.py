import streamlit as st
import pandas as pd

from src.profiler import *


st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("AI Data Analyst")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    profile = profile_dataset(df)

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["columns"])
    col3.metric("Duplicates", profile["duplicates"])

    st.subheader("Column Information")

    st.write(profile["dtypes"])

    st.subheader("Missing Values")

    st.write(profile["missing_values"])

    st.subheader("Metrics")

    metrics = ["mean","median","max","min","std"]

    for metric in metrics:
        st.write(metric_dataset(df,metric))

    categorical_columns = profile["categorical_columns"]

    st.subheader("Categorical Vizualization")

    colonne_choisie = st.selectbox(
        "Choose a column to visualise :",
        categorical_columns
    )

    if colonne_choisie:
        fig = categorical_viz(df,colonne_choisie)
        st.pyplot(fig , use_container_width=False)

    numerical_columns = profile["numeric_columns"]

    st.subheader("Numerical Vizualization")

    col_num1 , col_num2 = st.columns(2)

    with col_num1:
        x_choix = st.selectbox("Choose the x-coordinate" , numerical_columns)
    with col_num2:
        y_choix = st.selectbox("Choose the y-coordinate", numerical_columns)

    if x_choix and y_choix:
        fig = numerical_viz(df,x_choix,y_choix)
        st.pyplot(fig , use_container_width=False)

    st.subheader("Correlation HeatMap")

    fig = corr_matrix(df,numerical_columns)
    st.pyplot(fig , use_container_width=False)


    
    