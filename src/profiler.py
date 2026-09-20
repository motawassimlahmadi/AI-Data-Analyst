import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.grid(axis='y') 
    
    return fig
