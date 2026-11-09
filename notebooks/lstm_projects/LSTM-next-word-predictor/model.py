"""LSTM model definition."""

from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential

from config import EMBEDDING_DIM, LSTM_UNITS, VOCAB_SIZE


def build_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(lstm_units),
        Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
