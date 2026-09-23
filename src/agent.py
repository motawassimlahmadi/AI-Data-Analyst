import os
import time
from google import genai
from google.genai import types
import pandas as pd
from src.profiler import *
import streamlit as st

def my_tools():

    my_tools = [
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
                                "Use to analyse a numeric column from the dataset"
                                "Only Use this tool whenever the user asks for the average, mean, "
                                "median, minimum, maximum, standard deviation, or general "
                                "statistics of a specific column. "
                                "The column argument must be the exact numeric column name from the dataset."
                                "Note that this function can alos be used to check possible outliers"
                            ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "col": types.Schema(
                                    type=types.Type.STRING,
                                    description="Name of the numeric column"
                                )
                            },
                            required=["col"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="categorical_count",
                        description="Use this tool to calculate the count of a categorical column",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "categorical_column": types.Schema(
                                    type=types.Type.STRING,
                                    description="Name of the categorical column"
                                )
                            },
                            required=["categorical_column"]
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
                    ),
                    types.FunctionDeclaration(
                            name="outlier_check",
                            description=(
                                "Use this tool to inspect the outliers in the dataset "
                            )
                        ),
                    types.FunctionDeclaration(
                        name="groupby_aggregation",
                        description=(
                            "Groups the dataset by a categorical column and applies an aggregation function "
                            "to a numerical column. Use this tool for questions asking for metrics per category, "
                            "like 'average salary by department' or 'total sales per city'."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "groupby_col": types.Schema(
                                    type=types.Type.STRING,
                                    description="The exact name of the categorical column to group by."
                                ),
                                "agg_col": types.Schema(
                                    type=types.Type.STRING,
                                    description="The exact name of the numerical column to aggregate."
                                ),
                                "agg_func": types.Schema(
                                    type=types.Type.STRING,
                                    description="The aggregation function to apply. Must be one of: 'mean', 'sum', 'max', 'min', 'count', or 'median'."
                                )
                            },
                            required=["groupby_col", "agg_col", "agg_func"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="distribution_viz",
                        description=(
                            "Creates a histogram to visualize the distribution of a single numerical column. "
                            "Use this tool when the user asks to see the distribution, spread, or shape of a numerical variable."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "numerical_column": types.Schema(
                                    type=types.Type.STRING,
                                    description="Exact name of the numerical column to visualize."
                                )
                            },
                            required=["numerical_column"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="missing_values_report",
                        description=(
                            "Calculates the exact percentage of missing values for all columns in the dataset. "
                            "Use this tool to evaluate data quality or when asked about null/missing values."
                        )
                        # Pas de parameters car la fonction ne prend que 'df' en argument
                    ),
                    types.FunctionDeclaration(
                        name="time_series_viz",
                        description=(
                            "Creates a line chart to visualize the evolution of a numerical variable over time. "
                            "Use this tool when the user asks to plot trends over dates or time."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "date_col": types.Schema(
                                    type=types.Type.STRING,
                                    description="Exact name of the date or time column for the X-axis."
                                ),
                                "numerical_col": types.Schema(
                                    type=types.Type.STRING,
                                    description="Exact name of the numerical column for the Y-axis."
                                )
                            },
                            required=["date_col", "numerical_col"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="top_n_records",
                        description=(
                            "Returns the top N rows of the dataset sorted by a specific column. "
                            "Use this tool for queries asking for the 'top 5 highest' or 'bottom 3 lowest' records."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "sort_col": types.Schema(
                                    type=types.Type.STRING,
                                    description="The exact name of the column to sort by."
                                ),
                                "n": types.Schema(
                                    type=types.Type.INTEGER,
                                    description="The number of records to return (e.g., 5 or 10)."
                                ),
                                "ascending": types.Schema(
                                    type=types.Type.BOOLEAN,
                                    description="Set to False for the highest values (Top). Set to True for the lowest values (Bottom)."
                                )
                            },
                            required=["sort_col", "n", "ascending"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="skewness_kurtosis",
                        description=(
                            "Calculates the Skewness (asymmetry) and Kurtosis of a numerical distribution. "
                            "Use this tool to check if a variable follows a normal distribution."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "numerical_column": types.Schema(
                                    type=types.Type.STRING,
                                    description="Exact name of the numerical column."
                                )
                            },
                            required=["numerical_column"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="IQR",
                        description=(
                            "Use this tool to inspect outliers in a numerical column using the "
                            "Interquartile Range (IQR) method. The method identifies values "
                            "below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. "
                            "Use this tool when the user asks to detect or identify "
                            "outliers in a specific numerical column."
                        ),
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "numerical_column": types.Schema(
                                    type=types.Type.STRING,
                                    description=(
                                        "Exact name of the numerical column "
                                        "to analyze for outliers."
                                    )
                                )
                            },
                            required=["numerical_column"]
                        ),
                    
                    ),
                    types.FunctionDeclaration(
                    name="boxplot_viz",
                    description=(
                        "Creates a boxplot to compare the distribution of a numerical variable "
                        "across different categories. Use this tool when the user wants to compare "
                        "a numeric metric across different groups (e.g., 'salary by department' or 'age by gender')."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "categorical_col": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the categorical column (the groups/X-axis)."
                            ),
                            "numerical_col": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the numerical column (the metric/Y-axis)."
                            )
                        },
                        required=["categorical_col", "numerical_col"]
                    )
                ),
                types.FunctionDeclaration(
                    name="crosstab_analysis",
                    description=(
                        "Computes a cross-tabulation (pivot table) of two categorical columns to see their "
                        "frequency distribution. Use this tool when the user asks for the count or relationship "
                        "between two categories (e.g., 'number of men and women in each department')."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "col1": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the first categorical column."
                            ),
                            "col2": types.Schema(
                                type=types.Type.STRING,
                                description="Exact name of the second categorical column."
                            )
                        },
                        required=["col1", "col2"]
                    )
                )
                ]
            )
        ]

    return my_tools



def ask_agent(chat_session, df: pd.DataFrame, question: str):

    response = chat_session.send_message(question)

    while response.function_calls:
        reponses_outils = []
        
        for tool_call in response.function_calls:
            
            if tool_call.name == "profile_dataset":
                resultat_outil = profile_dataset(df)
                if resultat_outil:
                    print("1")
                    print(resultat_outil)
                

            elif tool_call.name == "metric_dataset":
                print("2")
                col_name = tool_call.args["col"]
                resultat_outil = metric_dataset(df, col_name)

            elif tool_call.name == "categorical_viz":
                print("3")
                col_name = tool_call.args["col"]
                fig = categorical_viz(df, col_name)
                st.pyplot(fig,width="content")
                resultat_outil = {
                        "status": "success",
                        "message": (
                            "The graphic was generated "
                            "and displayed successfully."
                    )
                }

            elif tool_call.name == "numerical_viz":
                print("4")
                x_col = tool_call.args["x_col"]
                y_col = tool_call.args["y_col"]
                fig = numerical_viz(df, x_col, y_col)
                st.pyplot(fig,width="content")
                resultat_outil = {
                                    "status": "success",
                                    "message": (
                                        "The graphic was generated "
                                        "and displayed successfully."
                    )
                }

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
            elif tool_call.name == "outlier_check":
                print("OC")
                resultat_outil = outlier_check(df)
            elif tool_call.name == "IQR":
                print("IQR")
                numerical_column = tool_call.args["numerical_column"]

                resultat_outil = IQR(
                    df,
                    numerical_column
                )
            elif tool_call.name == "categorical_count":
                print("YES")
                categorical_column = tool_call.args["categorical_column"]
                resultat_outil = categorical_count(df,categorical_column)


            elif tool_call.name == "groupby_aggregation":
                print("8 - Aggrégation (Group By)")
                groupby_col = tool_call.args["groupby_col"]
                agg_col = tool_call.args["agg_col"]
                agg_func = tool_call.args["agg_func"]
                resultat_outil = groupby_aggregation(df, groupby_col, agg_col, agg_func)

            elif tool_call.name == "distribution_viz":
                print("9 - Graphique de distribution")
                num_col = tool_call.args["numerical_column"]
                fig = distribution_viz(df, num_col)
                st.pyplot(fig, width="content")
                resultat_outil = {
                    "status": "success", 
                    "message": f"L'histogramme de {num_col} a été généré et affiché."
                }

            elif tool_call.name == "missing_values_report":
                print("10 - Rapport des valeurs manquantes")
                resultat_outil = missing_values_report(df)

            elif tool_call.name == "time_series_viz":
                print("11 - Graphique temporel")
                date_col = tool_call.args["date_col"]
                num_col = tool_call.args["numerical_col"]
                fig = time_series_viz(df, date_col, num_col)
                st.pyplot(fig, width="content")
                resultat_outil = {
                    "status": "success", 
                    "message": f"Le graphique d'évolution de {num_col} dans le temps a été généré et affiché."
                }

            elif tool_call.name == "top_n_records":
                print("12 - Extraction Top N")
                sort_col = tool_call.args["sort_col"]
                n = int(tool_call.args["n"]) 
                ascending = bool(tool_call.args["ascending"])
                resultat_outil = top_n_records(df, sort_col, n, ascending)

            elif tool_call.name == "boxplot_viz":
                categorical_column = tool_call.args["categorical_col"]
                numerical_column = tool_call.args["numerical_col"]
                fig = boxplot_viz(df,categorical_column,numerical_column)
                st.pyplot(fig,width='content')

            elif tool_call.name == "crosstab_analysis":
                col1 = tool_call.args["col1"]
                col2 = tool_call.args["col2"]
                resultat_outil = crosstab_analysis(df,col1,col2)

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
            
        response = chat_session.send_message(reponses_outils)
            
    return response.text
