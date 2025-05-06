import pandas as pd
import numpy as np

import src.stat_utils.numeric_data_statistics as nds
import src.stat_utils.categorical_data_statistics as cds
import src.working_utils.create_dir as cd

def data_statistics(dataset: pd.DataFrame):
    cd.create_dir('statistics')

    numeric_columns = []
    categorical_columns = []

    for column in dataset:
        if np.issubdtype(dataset[column].dtype, np.number):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    
    nds.numeric_data_statistics(dataset, numeric_columns)
    cds.categorical_data_statistics(dataset, categorical_columns)