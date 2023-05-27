
import pandas as pd
import pyarrow as pa
import re
from sklearn.impute import SimpleImputer

def divide_dataset_by_collumn(dataset, collumn):
    unique_values = dataset[collumn].unique()
    divided_datasets = {}

    for value in unique_values:
        subset = pd.DataFrame(dataset[dataset[collumn] == value])
        divided_datasets[value] = subset

    return divided_datasets

def fill_null_values(dataset, strategy):
    # select only the columns with float data type
    float_cols = dataset.select_dtypes(include=['float'])

    # create a SimpleImputer object with the mean strategy
    imp = SimpleImputer(strategy='median')

    # fit the imputer to the float columns
    imp.fit(float_cols)

    # transform the float columns by replacing missing values with the strategy
    not_null = imp.transform(float_cols)

    # replace the original float columns with the transformed values
    dataset[float_cols.columns] = not_null

    return dataset

def fill_null_by_income(dataset, strategy):
    income_datasets = divide_dataset_by_collumn(dataset, 'incomeLevel')
    filled_datasets = []

    for value, df in income_datasets.items():
        filled_df = fill_null_values(df, strategy)
        filled_datasets.append(filled_df)
    
    filled_dataset = pd.concat(filled_datasets)

    return filled_dataset
