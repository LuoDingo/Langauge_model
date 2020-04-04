import pandas as pd
import random
import spacy_udpipe

def _default_taggeer():
    spacy_udpipe.download('en')
    return spacy_udpipe.load('en')

def _is_mask(prob):
    return random.random()<prob

def mask_corpus(corpus, prob_mask, keep_tags, keep_words, tagger=None):
    if tagger==None:
        tagger = _default_taggeer()

    masked_corpus = []
    for id, sentence in enumerate(corpus):
        _sentence = tagger(sentence.lower())
        masked_sentence = []
        for token in _sentence:
            if (token.text not in keep_words) and (token.pos_ not in keep_tags):
                if _is_mask(prob_mask):
                    masked_sentence.append('[MASK]')
                else:
                    masked_sentence.append(token.text)
            else:
                masked_sentence.append(token.text)
        masked_corpus.append((masked_sentence, id))
    return masked_corpus
