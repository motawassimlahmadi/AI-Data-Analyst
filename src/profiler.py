import pandas as pd 

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
