from io import StringIO
import requests
import pandas as pd

def takeKeywords():
    keywords = input("Enter keywords separated by space: ")
    return keywords.split(" ")

def fetchCSVFromGit(url, index_col):
    content=requests.get(url).content
    df=pd.read_csv(StringIO(content.decode('utf-8')), header=None, index_col=index_col, names=['text'])
    return df