import pandas as pd

def numeric_data_statistics(dataset: pd.DataFrame, columns):
        csv_table = {
                "columns": ["column", "mean", "median", "min_value", "max_value", "std_deviation", "5th_percentile", "95th_percentile", "null_values"],
                "rows": [],
        }

        for column in columns:
                csv_table["rows"].append([
                        column,
                        dataset[column].mean(),
                        dataset[column].median(),
                        dataset[column].min(),
                        dataset[column].max(),
                        dataset[column].std(),
                        dataset[column].quantile(0.05),
                        dataset[column].quantile(0.95),
                        dataset[column].isnull().sum()
                ])

        out_df = pd.DataFrame(csv_table["rows"], columns=csv_table["columns"])
        out_df.to_csv("statistics/numerical_statistics.csv")