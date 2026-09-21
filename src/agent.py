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
                print("Appel : PROFILE")
                resultat_outil = profile_dataset(df)
                
            elif tool_call.name == "metric_dataset":
                col_name = tool_call.args["col"]
                print(f"Appel : METRIC pour '{col_name}'")
                resultat_outil = metric_dataset(df, col_name)

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
