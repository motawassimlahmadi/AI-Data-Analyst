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

    colonne_choisie = st.selectbox(
        "Choose a column to visualise :",
        categorical_columns
    )

    if colonne_choisie:
        fig = categorical_viz(df,colonne_choisie)
        st.pyplot(fig)
    
    