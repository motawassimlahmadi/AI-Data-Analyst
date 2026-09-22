import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def profile_dataset(df: pd.DataFrame) -> dict:

    """

    df : The dataframe

    Analysis of the dataset profile : Rows, columns , columns names , dtypes , missing values , duplicates , numeric columns , categorical columns

    Returns a dict of the global profile
    
    """
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

def metric_dataset(df : pd.DataFrame,col) -> dict:

    """
    df : The dataframe 
    col : The column we want to calculate its metrics

    Calculates differents metrics like mean , max etc for the column in the df 

    Returns a dict of the different metrics for the column
    
    """
    metrics = ["mean","median","max","min","std"]


    metric_dict = {"Column":col}
    for metric in metrics:
        metric_dict[f"{metric}"] = df[col].agg(metric)

    return metric_dict


def categorical_count(df:pd.DataFrame,categorical_column:str):
    return df.groupby(categorical_column)[categorical_column].value_counts()



def categorical_viz(df: pd.DataFrame, col: str):

    """
    df : The Dataframe 
    col : Column we want to vizualize

    Returns the vizualation of the categorical column 
    
    """


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

    """
        df : The Dataframe 
        col : Column we want to vizualize
    
        Returns the vizualation of the numerical column 
        
    """


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

    """
    Returns the correlation matrix of the dataframe
    
    """
    df_corr = df.filter(items=numerical_columns)
    matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(matrix,annot=True, cmap="coolwarm",fmt=".2f" , linewidths=0.5)
    ax.set_title("Correlation HeatMap")

    return fig


def high_cardinality(df: pd.DataFrame):

    """
    
    Checks if there are useless high cardinality repitition columns likes IDs , index that can be dropped for analysis
    
    """


    risk_of_HC = df.shape[0] * 0.75

    columns = df.columns
    cols_HC = []

    for col in columns:
        if df[col].nunique() >= risk_of_HC:
            cols_HC.append(col)

    return cols_HC

def useless_cols(df: pd.DataFrame):

    """
    
    Columns that have the same value are useless
    
    
    """
    useless_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            useless_cols.append(col)

    return useless_cols


def outlier_check(df : pd.DataFrame):
    different_vals = {}
    for col in df.columns:
        different_vals[col] = df[col].unique()

    return different_vals

def IQR(df: pd.DataFrame , numerical_column : str):
    Q1 = np.percentile(df[numerical_column],25,method="midpoint")
    Q2 = np.percentile(df[numerical_column],50,method="midpoint")
    Q3 = np.percentile(df[numerical_column],75,method="midpoint")

    IQR = Q3 - Q1

    low_lim = Q1 - 1.5*IQR
    up_lim = Q3+1.5*IQR

    outliers = []
    for elem in df[numerical_column]:
        if (elem >up_lim) or (elem<low_lim):
            outliers.append(elem)
    
    return {
        "column": numerical_column,
        "Q1": float(Q1),
        "median": float(Q2),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "lower_bound": float(low_lim),
        "upper_bound": float(up_lim),
        "number_of_outliers": int(len(outliers)),
        "outlier_values": outliers
    }









