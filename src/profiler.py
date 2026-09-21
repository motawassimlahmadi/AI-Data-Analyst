import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": list(
            df.select_dtypes(include="number").columns
        ),
        "categorical_columns": list(
            df.select_dtypes(include="object").columns
        ),
    }

    return profile

def metric_dataset(df : pd.DataFrame , metric : str) -> dict:
    numerical_columns = list(df.select_dtypes(include="number").columns)
    metrics = ["mean","median","max","min","std"]

    metric = metric.rstrip().lower()

    metric_dict = {"Metric" : metric}

    for col in numerical_columns:
        if metric in metrics:
            metric_dict[col] = float(df[col].agg(metric))
        
    return metric_dict

def categorical_viz(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(6,6))

    
    counts = df[col].value_counts()
    X_cat = counts.index.astype(str).tolist()
    Y_cat = counts.values.tolist()


    ax.bar(X_cat, Y_cat)
    ax.set_title(col)
    ax.grid(True)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.grid(axis='y') 
    
    return fig

def numerical_viz(df: pd.DataFrame, x_col: str, y_col: str):
    fig, ax = plt.subplots(figsize=(6,6))

    
    X_val = df[x_col]
    Y_val = df[y_col]

    ax.scatter(X_val,Y_val)
    ax.set_title(f"{x_col} vs {y_col}")
    ax.grid(True)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    
    return fig



def corr_matrix(df: pd.DataFrame , numerical_columns: list[str]):
    df_corr = df.filter(items=numerical_columns)
    matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(matrix,annot=True, cmap="coolwarm",fmt=".2f" , linewidths=0.5)
    ax.set_title("Correlation HeatMap")

    return fig


def high_cardinality(df: pd.DataFrame):
    risk_of_HC = df.shape[0] * 0.75

    columns = df.columns
    cols_HC = []

    for col in columns:
        if df[col].nunique() >= risk_of_HC:
            cols_HC.append(col)

    return cols_HC

def useless_cols(df: pd.DataFrame):
    useless_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            useless_cols.append(col)

    return useless_cols





