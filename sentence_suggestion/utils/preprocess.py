import re
import spacy
import nltk
from nltk.corpus import stopwords
import string, re

KEEP = ['my','her','him','them',
        'me','hers','his','their',
        'do','does','did','doing',
        'i','we','she','he','you','they','it',
        'which','when','where','who','when','how','what',
        'no','not',
        'both','all','more','few','very','now','too',
        'can','will']

spacy_en = spacy.load('en_core_web_sm')
def tokenize(sentence):
    return [token.text for token in spacy_en.tokenizer(sentence)]

def _remove_char(value, target):
    question_idx = value.find(target)
    # if target doesn't exist in value
    if question_idx == -1:
        return value
    value = list(value)
    value[question_idx]=''
    return ''.join(value)

def _remove_digit(sentence):
    return re.sub(r'\d+', '', sentence)

def _remove_punctuation(sentence, keep_puncs):
    punc = string.punctuation
    for char in keep_puncs:
        punc = _remove_char(punc, char)
    return sentence.translate(str.maketrans('', '',punc))

nltk.download('stopwords')
def _remove_stopwords(sentence, reduce):
    stop_words = stopwords.words('english')
    if reduce:
        [stop_words.remove(word) if word in stop_words else None for word in KEEP]
    return ' '.join(list(filter(lambda x: x not in stop_words, sentence.split())))

def clean_corpus(sentence, keep_puncs, reduce=True):
    sentence = sentence.lower()
    sentence = _remove_digit(sentence)
    sentence = _remove_stopwords(sentence, reduce)
    sentence = _remove_punctuation(sentence, keep_puncs)
    return sentence
