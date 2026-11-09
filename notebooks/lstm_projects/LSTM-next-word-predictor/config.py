"""Paths and hyperparameters for the LSTM next-word predictor."""

from pathlib import Path

DATA_FILE = Path("/home/mrafiku/AI_learning/deep_learning/data/lstm_text_data/cms_vac_data.txt")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "next_word_model.keras"
VOCAB_PATH = ARTIFACTS_DIR / "vocabulary.json"
META_PATH = ARTIFACTS_DIR / "meta.json"
HISTORY_PLOT_PATH = ARTIFACTS_DIR / "training_history.png"

VOCAB_SIZE = 200
EMBEDDING_DIM = 60
LSTM_UNITS = 150
EPOCHS = 50
BATCH_SIZE = 24
VALIDATION_SPLIT = 0.2
