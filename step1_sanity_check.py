"""Step 1: sanity-check the Kaggle Digit Recognizer data before any modelling.

Expects data/train.csv and data/test.csv relative to this file.

Run:  python step1_sanity_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 10
PIXEL_MAX = 255
EXPECTED_TRAIN_ROWS = 42000
EXPECTED_TEST_ROWS = 28000

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append((name, passed, detail))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    section("Files")
    for name in ("train.csv", "test.csv"):
        path = DATA_DIR / name
        exists = path.exists()
        size = path.stat().st_size / 1024**2 if exists else 0
        print(f"  {name:<16} {'found' if exists else 'MISSING':<8} {size:>7.1f} MB")
        check(f"{name} exists", exists, str(path))
    if not all(c[1] for c in checks):
        print("\nData files missing - stopping.")
        raise SystemExit(1)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    section("Shapes")
    print(f"  train  {train.shape[0]:>6} rows x {train.shape[1]:>3} cols")
    print(f"  test   {test.shape[0]:>6} rows x {test.shape[1]:>3} cols")
    check(
        "train is 42000 x 785 (label + 784 pixels)",
        train.shape == (EXPECTED_TRAIN_ROWS, N_PIXELS + 1),
        f"got {train.shape}",
    )
    check(
        "test is 28000 x 784 (pixels only)",
        test.shape == (EXPECTED_TEST_ROWS, N_PIXELS),
        f"got {test.shape}",
    )

    section("Columns")
    expected_pixels = [f"pixel{i}" for i in range(N_PIXELS)]
    print(f"  train first 5 : {list(train.columns[:5])}")
    print(f"  test  first 5 : {list(test.columns[:5])}")
    check(
        "train cols are label + pixel0..pixel783",
        list(train.columns) == ["label"] + expected_pixels,
        "names and order match",
    )
    check(
        "test cols are pixel0..pixel783",
        list(test.columns) == expected_pixels,
        "names and order match",
    )

    section("Missing values")
    tr_na, te_na = int(train.isna().sum().sum()), int(test.isna().sum().sum())
    print(f"  train NaN : {tr_na}")
    print(f"  test  NaN : {te_na}")
    check("no missing values", tr_na == 0 and te_na == 0, f"train={tr_na}, test={te_na}")

    section("Pixel values")
    x_train = train.drop(columns="label").to_numpy(dtype=np.int32)
    x_test = test.to_numpy(dtype=np.int32)
    print(f"  train : min={x_train.min()}  max={x_train.max()}  mean={x_train.mean():.2f}")
    print(f"  test  : min={x_test.min()}  max={x_test.max()}  mean={x_test.mean():.2f}")
    print(f"  background (zero) pixels in train: {(x_train == 0).mean():.1%}")
    check("train pixels in 0..255", x_train.min() >= 0 and x_train.max() <= PIXEL_MAX, f"[{x_train.min()}, {x_train.max()}]")
    check("test pixels in 0..255", x_test.min() >= 0 and x_test.max() <= PIXEL_MAX, f"[{x_test.min()}, {x_test.max()}]")

    section("Labels")
    counts = train["label"].value_counts().sort_index()
    total = len(train)
    for digit, n in counts.items():
        print(f"  {digit}  {n:>6}  {n / total:>6.2%}  {'#' * round(n / total * 200)}")
    ratio = counts.max() / counts.min()
    print(f"\n  imbalance ratio (max/min): {ratio:.3f}x")
    check("all 10 digits present", set(counts.index) == set(range(N_CLASSES)), f"{len(counts)} classes")
    check("classes roughly balanced (<1.25x)", ratio < 1.25, f"{ratio:.3f}x")

    section("Summary")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<44} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
