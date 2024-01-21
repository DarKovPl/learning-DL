import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer


datasets = tf.keras.datasets
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
sns.set()

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data()

print(f'train_data shape: {x_train.shape}')
print(f'test_data shape: {x_test.shape}')

INDEX_FROM = 3
word_index = datasets.imdb.get_word_index()
word_index = {k: (v + INDEX_FROM) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
index_word = {v: k for k, v in word_index.items()}
print(' '.join(index_word[idx] for idx in x_train[0]))


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(x_train, np.max(x_train.max() + x_test.max()))
x_test = vectorize_sequences(x_test, np.max(x_train.max() + x_test.max()))
print(f'\n{x_train.shape}')
print(f'\n{x_test.shape}')

