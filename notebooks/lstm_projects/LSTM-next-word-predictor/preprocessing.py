"""Text vectorization and n-gram sequence construction."""

import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import VOCAB_SIZE


def build_vectorizer(text_data, vocab_size=VOCAB_SIZE):
    vectorize_layer = TextVectorization(max_tokens=vocab_size, output_mode="int")
    vectorize_layer.adapt(text_data)
    return vectorize_layer


def build_ngram_sequences(text_data, vectorize_layer):
    input_sequences = []

    for line in text_data:
        token_list = vectorize_layer([line]).numpy()[0]
        token_list = token_list[token_list != 0]

        for i in range(1, len(token_list)):
            input_sequences.append(token_list[: i + 1])

    return input_sequences


def pad_and_split(input_sequences):
    max_sequence_len = max(len(seq) for seq in input_sequences)
    padded = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

    X = padded[:, :-1]
    y = padded[:, -1]
    return X, y, max_sequence_len
