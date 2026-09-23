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



def groupby_aggregation(df: pd.DataFrame, groupby_col: str, agg_col: str, agg_func: str = "mean") -> dict:
    """
    Groups the dataframe by a categorical column and applies an aggregation function to a numerical column.
    
    groupby_col : The categorical column to group by.
    agg_col : The numerical column to aggregate.
    agg_func : The aggregation function ('mean', 'sum', 'max', 'min', 'count', 'median').
    
    Returns a dictionary of the aggregated results.
    """
    try:
        grouped_data = df.groupby(groupby_col)[agg_col].agg(agg_func).to_dict()
        return {
            "groupby_column": groupby_col,
            "aggregated_column": agg_col,
            "applied_function": agg_func,
            "results": grouped_data
        }
    except Exception as e:
        return {"error": str(e)}


def distribution_viz(df: pd.DataFrame, numerical_column: str):
    """
    Creates a histogram with a Kernel Density Estimate (KDE) to visualize 
    the distribution of a single numerical column.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(df[numerical_column].dropna(), kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Distribution of {numerical_column}")
    ax.set_xlabel(numerical_column)
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.7)
    
    return fig

def missing_values_report(df: pd.DataFrame) -> dict:
    """
    Calculates the exact percentage of missing values for columns that have at least one missing value.
    Returns a dictionary mapping column names to their missing value percentage.
    """
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_filtered = missing_percentages[missing_percentages > 0].round(2)
    
    if missing_filtered.empty:
        return {"message": "No missing values found in the dataset."}
        
    return missing_filtered.to_dict()

def time_series_viz(df: pd.DataFrame, date_col: str, numerical_col: str):
    """
    Creates a line chart to visualize the evolution of a numerical variable over time.
    Note: date_col should ideally be converted to datetime before, or be sortable.
    """
    # Trier les données par date temporairement pour le graphique
    df_sorted = df.sort_values(by=date_col)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df_sorted[date_col], df_sorted[numerical_col], marker='o', linestyle='-', markersize=4)
    ax.set_title(f"Evolution of {numerical_col} over {date_col}")
    ax.set_xlabel(date_col)
    ax.set_ylabel(numerical_col)
    ax.grid(True)
    
    # Rotation des labels X si ce sont des dates longues
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def top_n_records(df: pd.DataFrame, sort_col: str, n: int = 5, ascending: bool = False) -> list:
    """
    Returns the top N rows of the dataset sorted by a specific column.
    
    sort_col : The column to sort by.
    n : Number of rows to return (default 5).
    ascending : False for highest values (Top), True for lowest values (Bottom).
    
    Returns a list of dictionaries (records).
    """
    try:
        top_df = df.sort_values(by=sort_col, ascending=ascending).head(n)
        return top_df.to_dict(orient="records")
    except Exception as e:
        return [{"error": str(e)}]


def boxplot_viz(df: pd.DataFrame, categorical_col: str, numerical_col: str):
    """
    Creates a boxplot to compare the distribution of a numerical variable 
    across different categories.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x=categorical_col, y=numerical_col, ax=ax, palette="Set2")
    ax.set_title(f"Distribution of {numerical_col} by {categorical_col}")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def crosstab_analysis(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Computes a cross-tabulation of two categorical factors to see their frequency distribution.
    """
    try:
        ct = pd.crosstab(df[col1], df[col2])
        return {
            "row_variable": col1,
            "column_variable": col2,
            "crosstab_data": ct.to_dict()
        }
    except Exception as e:
        return {"error": str(e)}


