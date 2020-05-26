# @Author: Kei Nemoto <box-key>
# @Date:   2020-03-05T18:53:55-05:00
# @Last modified by:   box-key
# @Last modified time: 2020-03-05T20:10:58-05:00

from itertools import combinations
import pandas as pd
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from ..utils.data import fetch_csv_from_url

# Helper methods
def _getCandidates(keywords, df, threshold):
    size = len(keywords)
    # Store sentences that contain keywords
    df_return = pd.DataFrame([])
    # Repeat the process until df_return contains more examples than threshold
    # Or finish searching the sentence
    while (df_return.shape[0] <= threshold) and (size>0):
        for query in _getQueryCombinations(keywords, size):
            # Get all the sentences contains all words in a combination, but not exist in output candidates
            df_return = df[df['text'].str.contains(query) & ~df.index.isin(df_return.index)]
        size -= 1

    return df_return

def _getQueryCombinations(keywords, r):
    queries = []
    # Generate all comibnations of size r in keywords
    for i in combinations(keywords, r):
        queries.append(_getQuery(i))
    return queries

def _getQuery(combinations):
    query = ""
    # (?=.*word1)(?=.*word2) is equivalent to word1&word2 in regex
    for word in combinations:
        query += '(?=.*' + word + ')'
    return query

def _fuzzyScore(query, candidate_sentences, limit, scoring_method='ratio'):
    """
    ratio = compares the entire string in order
    partial_ration = compares subsections of the string
    toke_sort_ratio = ignores word order
    token_set_ratio = ignores duplicate words
    Default is ratio if you don't specify scoring method.

    * Reference: http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/fuzzing-matching-in-pandas-with-fuzzywuzzy/
    """
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

class KeywordMatcher():
    """
    target space must be a dataframe with only one column that contains target sentences

    (e.g.)
    --------------
    |    text    |
    --------------
    | sentence 1 |
    | sentence 2 |
    |    ....    |
    | sentence N |
    --------------
    """

    def __init__(self, url, col_index):
        self.url = url
        self.target_space = fetch_csv_from_url(url, column_indices=col_index, column_names=['text'])

    def selectKBestMatches(self, keywords, max_candidate, k, scoring_method):

        candidates = _getCandidates(keywords=keywords, df=self.target_space, threshold=max_candidate)
        # Convert keywords into one string
        if isinstance(keywords, list):
            keywords = ' '.join(keywords)
        suggestions = _fuzzyScore(query=keywords, candidate_sentences=candidates['text'], scoring_method=scoring_method, limit=k)
        return suggestions

    def update_target_space(self, url, col_index):
        self.target_space = fetch_csv_from_url(url, column_indices=col_index, column_names=['text'])
