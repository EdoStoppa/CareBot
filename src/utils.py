import pickle as pkl
import numpy as np
import string

# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector

# inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()

# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)

    tokens = get_tokens(preprocessing(user_input))
    for tkn in tokens:
        embedding += w2v(word2vec, tkn)
    embedding = embedding / len(tokens)
    return embedding

import pandas as pd


# fname: A string indicating a filename
# Returns: Two lists: one a list of strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (lexicon) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    lexicon = df['Lexicon'].values.tolist()
    label = df['Label'].values.tolist()
    return lexicon, label

# user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    # Initialize modified_input to be the same as the original user input
    modified_input = user_input

    # Write your code here:
    tokens = get_tokens(user_input)
    no_punctuation = []
    for t in tokens:
        if t not in string.punctuation:
            no_punctuation.append(t.lower())
    modified_input = ' '.join(no_punctuation)
    return modified_input