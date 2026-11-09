"""Loading the raw text corpus."""

from config import DATA_FILE


def load_text_data(path=DATA_FILE):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return [line for line in raw_text.split("\n") if line.strip() != ""]
