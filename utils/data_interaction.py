from io import StringIO
import requests
import pandas as pd

def takeKeywords():
    keywords = input("Enter keywords separated by space: ")
    return keywords.split(" ")

def fetch_csv_from_url(url, column_indices, column_names):

    if isinstance(column_names, str):
        column_names = [column_names]
    if isinstance(column_indices, int):
        column_indices = [column_indices]

    assert len(column_indices) == len(column_names), \
            'columns_indices and column_names should be the same length'
    assert 'text' in column_names, \
            'column_names should contain "text"'

    content=requests.get(url).content
    df=pd.read_csv(StringIO(content.decode('utf-8')), header=None, usecols=column_indices, names=column_names)
    return df
