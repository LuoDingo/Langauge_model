from io import StringIO
import requests
import pandas as pd

def takeKeywords():
    keywords = input("Enter keywords separated by space: ")
    return keywords.split(" ")

def fetch_csv_from_git(url):
    content=requests.get(url).content
    df=pd.read_csv(StringIO(content.decode('utf-8')), header=None, names=['id', 'text'])
    return df
