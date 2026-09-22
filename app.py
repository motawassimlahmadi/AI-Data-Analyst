import streamlit as st
import pandas as pd

from src.profiler import *
from src.agent import ask_agent



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
    st.subheader("Dataset Description")
    st.dataframe(df.describe())

    profile = profile_dataset(df)
    risk_of_HC = high_cardinality(df) # Detect columns that are lilkely to be ID's that are not useful for predictions.
    useless_col = useless_cols(df)
    numerical_columns = profile["numeric_columns"]

    st.subheader("Dataset Overview")

    col1, col2, col3 , col4 , col5 = st.columns(5)

    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["columns"])
    col3.metric("Duplicates", profile["duplicates"])
    col4.metric("Risk of High Cardinality", len(risk_of_HC))
    col5.metric("Useless Columns", len(useless_col))

    if risk_of_HC:
        col4.caption(f"Columns : {', '.join(risk_of_HC)}")

    if useless_col:
        col5.caption(f"Columns : {', '.join(useless_col)}")

    question = st.text_input("Enter your question here :")

    if st.button("Send your question"):
        if question:
            result = ask_agent(df,question)

            st.info(result)
            print(result)
        else:
            st.warning("Please write a question first !")

    st.subheader("Column Information")

    st.write(profile["dtypes"])

    st.subheader("Missing Values")

    st.write(profile["missing_values"])


    
    