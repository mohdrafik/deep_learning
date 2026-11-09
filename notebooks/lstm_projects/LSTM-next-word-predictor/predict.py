"""Load a trained model + vectorizer and generate next-word predictions."""

import argparse
import json

import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import META_PATH, MODEL_PATH, VOCAB_PATH


def load_artifacts():
    model = load_model(MODEL_PATH)
    vocabulary = json.loads(VOCAB_PATH.read_text())
    meta = json.loads(META_PATH.read_text())

    vectorize_layer = TextVectorization(max_tokens=meta["vocab_size"], output_mode="int")
    vectorize_layer.set_vocabulary(vocabulary[2:])  # drop '' and '[UNK]'; set_vocabulary re-adds them

    return model, vectorize_layer, meta["max_sequence_len"]


def predict_next_words(seed_text, model, vectorize_layer, max_sequence_len, num_words=5):
    for _ in range(num_words):
        token_list = vectorize_layer([seed_text]).numpy()[0]
        token_list = token_list[token_list != 0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        predicted_word = vectorize_layer.get_vocabulary()[predicted_index]

        seed_text += " " + predicted_word

    return seed_text


def main():
    parser = argparse.ArgumentParser(description="Generate next words from a seed phrase.")
    parser.add_argument("seed_text", nargs="?", default="Take a leading role")
    parser.add_argument("--num-words", type=int, default=8)
    args = parser.parse_args()

    model, vectorize_layer, max_sequence_len = load_artifacts()
    result = predict_next_words(args.seed_text, model, vectorize_layer, max_sequence_len, args.num_words)
    print(result)


if __name__ == "__main__":
    main()
