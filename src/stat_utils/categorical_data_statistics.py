import pandas as pd

def categorical_data_statistics(dataset: pd.DataFrame, columns):
    csv_table = {
                "columns": ["column", "unique_classes", "null_values", "proportion"],
                "rows": [],
        }
    
    for column in columns:
                csv_table["rows"].append([
                        column,
                        dataset[column].nunique(),
                        dataset[column].isnull().sum(),
                        ";".join([f"{k}:{v}" for k, v in (dataset[column].value_counts(normalize=True) * 100).items()])
                ])

    out_df = pd.DataFrame(csv_table["rows"], columns=csv_table["columns"])
    out_df.to_csv("statistics/categorical_statistics.csv")