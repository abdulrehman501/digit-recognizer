"""Dataset loading for the Kaggle Digit Recognizer competition."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 10
PIXEL_MAX = 255


def load_train() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "sample_submission.csv")
