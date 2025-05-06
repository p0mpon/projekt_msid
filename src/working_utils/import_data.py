import pandas as pd

def import_data():
    dataset = pd.read_csv("data\StudentPerformanceFactors.csv")
    return dataset