# @Author: Kei Nemoto <box-key>
# @Date:   2020-03-05T18:53:55-05:00
# @Last modified by:   box-key
# @Last modified time: 2020-03-05T20:10:58-05:00

from itertools import combinations
import pandas as pd
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from io import StringIO
import requests

def takeKeywords():
    keywords = input("Enter keywords separated by space: ")
    return keywords.split(" ")

def fetchCSVFromGit(url, index_col):
    content=requests.get(url).content
    df=pd.read_csv(StringIO(content.decode('utf-8')), header=None, index_col=index_col, names=['text'])
    return df

"""
ratio = compares the entire string in order
partial_ration = compares subsections of the string
toke_sort_ratio = ignores word order
token_set_ratio = ignores duplicate words
Default is ratio if you don't specify scoring method.

* Reference: http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/fuzzing-matching-in-pandas-with-fuzzywuzzy/
"""

def fuzzyScore(query, candidate_sentences, limit, scoring_method='ratio'):
    if scoring_method == 'ratio':
        output = process.extract(query, candidate_sentences, limit=limit, scorer=fuzz.ratio)
    elif scoring_method == 'partial_ratio':
        output = process.extract(query, candidate_sentences, limit=limit, scorer=fuzz.partial_ratio)
    elif scoring_method == 'token_sort_ratio':
        output = process.extract(query, candidate_sentences, limit=limit, scorer=fuzz.token_sort_ratio)
    elif scoring_method == 'token_set_ratio':
        output = process.extract(query, candidate_sentences, limit=limit, scorer=fuzz.token_set_ratio)
    else:
        output = []
    return output

def selectKBestMatches(keywords, target_space, max_candidate, k, scoring_method):
    candidates = getCandidates(keywords=keywords, df=target_space, threshold=max_candidate)
    # Convert keywords into one string
    if isinstance(keywords, list):
        keywords = ' '.join(keywords)
    suggestions = fuzzyScore(query=keywords, candidate_sentences=candidates['text'], scoring_method=scoring_method, limit=k)
    return suggestions

def getCandidates(keywords, df, threshold):
    size = len(keywords)
    # Store sentences that contain keywords
    df_return = pd.DataFrame([])
    # Repeat the process until df_return contains more examples than threshold
    # Or finish searching the sentence
    while (df_return.shape[0] <= threshold) and (size>0):
        for query in getQueryCombinations(keywords, size):
            # Get all the sentences contains all words in a combination, but not exist in output candidates
            df_return = df[df['text'].str.contains(query) & ~df.index.isin(df_return.index)]
        size -= 1

    return df_return

def getQueryCombinations(keywords, r):
    queries = []
    # Generate all comibnations of size r in keywords
    for i in combinations(keywords, r):
        queries.append(getQuery(i))
    return queries

def getQuery(combinations):
    query = ""
    # (?=.*word1)(?=.*word2) is equivalent to word1&word2 in regex
    for word in combinations:
        query += '(?=.*' + word + ')'
    return query
