import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from src.profiler import *
import streamlit as st

load_dotenv()
client = genai.Client()


def ask_agent(df: pd.DataFrame, question: str):

    mes_outils = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="profile_dataset",
                    description=(
                        "Analyze the overall structure of the dataset. "
                        "Use this tool to get the number of rows, columns, "
                        "column names, data types, missing values, duplicates, "
                        "numeric columns and categorical columns."
                    ),
                ),
                types.FunctionDeclaration(
                    name="metric_dataset",
                    description=(
                            "Analyze a specific numeric column of the dataset. "
                            "Use this tool whenever the user asks for the average, mean, "
                            "median, minimum, maximum, standard deviation, or general "
                            "statistics of a specific column. "
                            "The column argument must be the exact column name from the dataset."
                        ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "col": types.Schema(
                                type=types.Type.STRING,
                                description="Name of the column"
                            )
                        },
                        required=["col"]
                    )
                ),
                types.FunctionDeclaration(
                    name="categorical_viz",
                    description=(
                        "Create a bar chart showing the frequency of each category "
                        "in a categorical column. Use this tool when the user asks "
                        "to visualize the distribution of a categorical variable."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "col": types.Schema(
                                type=types.Type.STRING,
                                description=(
                                    "Exact name of the categorical column to visualize."
                                )
                            )
                        },
                        required=["col"]
                    )
                ),
                types.FunctionDeclaration(
                    name="numerical_viz",
                    description=(
                        "Create a scatter plot showing the relationship between "
                        "two numerical columns. Use this tool when the user asks "
                        "to visualize the relationship or correlation between "
                        "two numerical variables."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "x_col": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the numerical column for the X axis."
                            ),
                            "y_col": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the numerical column for the Y axis."
                            )
                        },
                        required=["x_col", "y_col"]
                    )
                ),
                types.FunctionDeclaration(
                    name="corr_matrix",
                    description=(
                        "Create a correlation heatmap for numerical columns. "
                        "Use this tool when the user asks to visualize correlations / create the correlation matrix "
                        "between numerical variables."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "numerical_columns": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.STRING
                                ),
                                description=(
                                    "List of exact names of the numerical columns "
                                    "to include in the correlation matrix."
                                )
                            )
                        },
                        required=["numerical_columns"]
                    )
                ),
                types.FunctionDeclaration(
                    name="high_cardinality",
                    description=(
                        "Detect columns with very high cardinality, meaning columns "
                        "where most rows contain unique values. Use this tool to "
                        "identify potential identifiers such as IDs or indexes that "
                        "may not be useful for analysis."
                    )
                ),
                types.FunctionDeclaration(
                    name="useless_cols",
                    description=(
                        "Detect columns containing only one unique value. "
                        "Use this tool to identify constant columns that provide "
                        "no useful information for analysis or machine learning."
                    )
                )
            ]
        )
    ]

    columns = list(df.columns)

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
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

    response = chat.send_message(question)

    if response.function_calls:
        reponses_outils = []
        
        for tool_call in response.function_calls:
            
            if tool_call.name == "profile_dataset":
                print("1")
                resultat_outil = profile_dataset(df)

            elif tool_call.name == "metric_dataset":
                print("2")
                col_name = tool_call.args["col"]
                resultat_outil = metric_dataset(df, col_name)

            elif tool_call.name == "categorical_viz":
                print("3")
                col_name = tool_call.args["col"]
                resultat_outil = categorical_viz(df, col_name)

            elif tool_call.name == "numerical_viz":
                print("4")
                x_col = tool_call.args["x_col"]
                y_col = tool_call.args["y_col"]
                resultat_outil = numerical_viz(df, x_col, y_col)

            elif tool_call.name == "corr_matrix":
                print("5")
                numerical_columns = tool_call.args["numerical_columns"]
                print("Colonnes numériques" , numerical_columns)
                fig = corr_matrix(df, numerical_columns)
                st.pyplot(fig , width="content")
                resultat_outil = {
                    "status": "success",
                    "message": (
                        "The correlation matrix was generated "
                        "and displayed successfully."
                    )
                }

            elif tool_call.name == "high_cardinality":
                print("6")
                resultat_outil = high_cardinality(df)

            elif tool_call.name == "useless_cols":
                print("7")
                resultat_outil = useless_cols(df)

            # Formatage
            if isinstance(resultat_outil, (pd.DataFrame, pd.Series)):
                resultat_clean = resultat_outil.to_json()
            else:
                resultat_clean = str(resultat_outil)

            reponses_outils.append(
                types.Part.from_function_response(
                    name=tool_call.name,
                    response={"result": resultat_clean}
                )
            )
            
        response = chat.send_message(reponses_outils)
            
    return response.text
