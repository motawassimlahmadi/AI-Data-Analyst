import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.profiler import *
from src.agent import *
from google import genai
from google.genai import types
load_dotenv()



if "client" not in st.session_state:
    st.session_state.client = genai.Client()

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
    columns = df.columns
    mes_outils = my_tools()

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

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = st.session_state.client.chats.create(
            model="gemini-3.6-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an AI Data Analyst. "
                    "Use the available tools to analyze the dataset. "
                    "Never invent numerical results. "
                    f"The available columns are: {columns}. "
                    "Whenever the user asks for statistics about a specific numeric "
                    "column, you MUST use metric_dataset."
                ),
                tools=mes_outils,
                temperature=0.0
            )
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez une question sur vos données..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                reponse_ia = ask_agent(st.session_state.chat_session, df, prompt)
                st.markdown(reponse_ia)

        st.session_state.messages.append({"role": "assistant", "content": reponse_ia})

    


    
    