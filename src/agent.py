import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from src.profiler import *

load_dotenv()
client = genai.Client()


def ask_agent(df: pd.DataFrame, question: str):

    mes_outils = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="profile_dataset",
                    description="Get descriptive statistics for the dataset.",
                ),
                types.FunctionDeclaration(
                    name="metric_dataset",
                    description="Analyze the metrics of a specific column.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "column": types.Schema(
                                type=types.Type.STRING,
                                description="Name of the column"
                            )
                        },
                        required=["column"]
                    )
                )
            ]
        )
    ]

    chat = client.chats.create(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an AI Data Analyst. "
                "Use the available tools to analyze the dataset. "
                "Never invent numerical results."
            ),
            tools=mes_outils,
            temperature=0.0
        )
    )

    response = chat.send_message(question)

    if response.function_calls:
        for tool_call in response.function_calls:
            
            if tool_call.name == "profile_dataset":
                print("PROFILE")
                resultat_outil = profile_dataset(df)
                
            elif tool_call.name == "metric_dataset":
                print("METRIC")
                col_name = tool_call.args["column"]
                resultat_outil = metric_dataset(df,col_name)

            if isinstance(resultat_outil,(pd.DataFrame , pd.Series)):
                resultat_outil = resultat_outil.to_json()
            else:
                resultat_outil = str(resultat_outil)

            response = chat.send_message(
                types.Part.from_function_response(
                    name=tool_call.name,
                    response={"result": resultat_outil}
                )
            )
            
    return response.text