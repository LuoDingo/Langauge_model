import random
import numpy as np

def _filter_space(df, min):
    # keep all texts with more than min words
    df_filtered = df[df['text'].apply(len) > min]
    return df_filtered

def _splits(data, train, val, test):
    random.shuffle(data)
    length = len(data)
    train_data = data[:int(train*length)]
    val_data = data[int(train*length):int((train+val)*length)]
    test_data = data[int((1-test)*length):]
    return train_data, val_data, test_data

def _generate_random_examples(text, min, max, num_data, label):
    assert len(text) > min, 'text should contain more than min words'

    # randomly decide the number of words for each example
    # bc we don't know how many keywords user would input
    if len(text) > max:
        n = [random.randint(min, max) for x in range(num_data)]
    else:
        n = [random.randint(min-1, min) for x in range(num_data)]
    # randomly pick up n words from text
    return [(' '.join(random.sample(text,x)), label) for x in n]

def generate_dataset(df, min_word, max_word, num_data, train_ratio, val_ratio, test_ratio):
    assert min_word >= 1, 'min_word should be more than at least one'
    assert train_ratio+test_ratio+val_ratio == 1, 'train_ratio, val_ratio, and test_ratio should be summed up to 1'

    # remove sentences whose length is less than than min_word
    df_filtered = _filter_space(df, min_word)
    train, val, test = [], [], []
    for idx, row in df_filtered.iterrows():
        label, text = row[0], row[1]
        # randomly generate samples from one text
        examples = _generate_random_examples(text, min_word, max_word, num_data, label)
        # split data into train, validation, and test
        train_, val_, test_  = _splits(examples, train_ratio, test_ratio, val_ratio)
        # add examples into data
        train = train + train_
        val = val + val_
        test = test + test_
    return train, val, test
