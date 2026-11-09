"""Train the LSTM next-word predictor and save model + vectorizer artifacts."""

import json

from config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    HISTORY_PLOT_PATH,
    META_PATH,
    MODEL_PATH,
    VALIDATION_SPLIT,
    VOCAB_PATH,
    VOCAB_SIZE,
)
from data import load_text_data
from evaluation import plot_history, print_summary
from model import build_model
from preprocessing import build_ngram_sequences, build_vectorizer, pad_and_split


def main():
    text_data = load_text_data()
    print(f"Number of lines in the text data: {len(text_data)}")

    vectorize_layer = build_vectorizer(text_data)
    input_sequences = build_ngram_sequences(text_data, vectorize_layer)
    print(f"Total training sequences: {len(input_sequences)}")

    X, y, max_sequence_len = pad_and_split(input_sequences)
    print(f"max sequence length: {max_sequence_len}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    model = build_model()
    model.summary()
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )
    print_summary(history)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    VOCAB_PATH.write_text(json.dumps(vectorize_layer.get_vocabulary()))
    META_PATH.write_text(
        json.dumps({"vocab_size": VOCAB_SIZE, "max_sequence_len": max_sequence_len})
    )
    plot_history(history, HISTORY_PLOT_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")
    print(f"Saved training curves to {HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
