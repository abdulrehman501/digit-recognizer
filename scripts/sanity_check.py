"""Step 1: verify the raw competition data matches what we expect before modelling.

Run from the repo root:  python -m scripts.sanity_check
"""

import numpy as np

from src.data import (
    N_CLASSES,
    N_PIXELS,
    PIXEL_MAX,
    load_sample_submission,
    load_test,
    load_train,
)

EXPECTED_TRAIN_ROWS = 42000
EXPECTED_TEST_ROWS = 28000


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    print("Loading CSVs from data/raw ...")
    train = load_train()
    test = load_test()
    sample = load_sample_submission()

    section("Shapes")
    print(f"train.csv              {train.shape[0]:>6} rows x {train.shape[1]:>3} cols")
    print(f"test.csv               {test.shape[0]:>6} rows x {test.shape[1]:>3} cols")
    print(f"sample_submission.csv  {sample.shape[0]:>6} rows x {sample.shape[1]:>3} cols")

    check(
        "train shape is 42000 x 785 (label + 784 pixels)",
        train.shape == (EXPECTED_TRAIN_ROWS, N_PIXELS + 1),
        f"got {train.shape}",
    )
    check(
        "test shape is 28000 x 784 (pixels only, no label)",
        test.shape == (EXPECTED_TEST_ROWS, N_PIXELS),
        f"got {test.shape}",
    )

    section("Columns")
    print(f"train first 5 : {list(train.columns[:5])}")
    print(f"train last 3  : {list(train.columns[-3:])}")
    print(f"test  first 5 : {list(test.columns[:5])}")
    print(f"submission    : {list(sample.columns)}")

    expected_pixel_cols = [f"pixel{i}" for i in range(N_PIXELS)]
    check(
        "train columns are 'label' + pixel0..pixel783",
        list(train.columns) == ["label"] + expected_pixel_cols,
        "column names/order match",
    )
    check(
        "test columns are pixel0..pixel783",
        list(test.columns) == expected_pixel_cols,
        "column names/order match",
    )

    section("Missing values")
    train_na = int(train.isna().sum().sum())
    test_na = int(test.isna().sum().sum())
    print(f"train NaN count : {train_na}")
    print(f"test  NaN count : {test_na}")
    check("no missing values anywhere", train_na == 0 and test_na == 0, f"train={train_na}, test={test_na}")

    section("Dtypes")
    print(f"train dtypes    : {sorted({str(d) for d in train.dtypes})}")
    print(f"test dtypes     : {sorted({str(d) for d in test.dtypes})}")
    print(f"train memory    : {train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"test memory     : {test.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    section("Pixel value range")
    x_train = train.drop(columns="label").to_numpy()
    x_test = test.to_numpy()
    tr_min, tr_max = int(x_train.min()), int(x_train.max())
    te_min, te_max = int(x_test.min()), int(x_test.max())
    print(f"train pixels    : min={tr_min}  max={tr_max}  mean={x_train.mean():.2f}")
    print(f"test  pixels    : min={te_min}  max={te_max}  mean={x_test.mean():.2f}")
    print(f"fraction of train pixels that are 0 (background): {(x_train == 0).mean():.1%}")

    check(
        "train pixels within 0..255",
        tr_min >= 0 and tr_max <= PIXEL_MAX,
        f"[{tr_min}, {tr_max}]",
    )
    check(
        "test pixels within 0..255",
        te_min >= 0 and te_max <= PIXEL_MAX,
        f"[{te_min}, {te_max}]",
    )

    section("Class balance (train labels)")
    counts = train["label"].value_counts().sort_index()
    total = len(train)
    for digit, n in counts.items():
        bar = "#" * round(n / total * 200)
        print(f"  {digit}  {n:>6}  {n / total:>6.2%}  {bar}")
    print(f"\n  mean per class : {counts.mean():.0f}")
    print(f"  min / max      : {counts.min()} (digit {counts.idxmin()}) / {counts.max()} (digit {counts.idxmax()})")
    print(f"  imbalance ratio: {counts.max() / counts.min():.3f}x")

    check(
        "all 10 digits present",
        len(counts) == N_CLASSES and set(counts.index) == set(range(N_CLASSES)),
        f"{len(counts)} classes found",
    )
    check(
        "classes roughly balanced (max/min < 1.25x)",
        counts.max() / counts.min() < 1.25,
        f"{counts.max() / counts.min():.3f}x",
    )

    section("Sample submission format")
    print(sample.head(3).to_string(index=False))
    print(f"ImageId range   : {sample['ImageId'].min()}..{sample['ImageId'].max()}")
    check(
        "submission ImageId is 1..28000 contiguous",
        np.array_equal(sample["ImageId"].to_numpy(), np.arange(1, EXPECTED_TEST_ROWS + 1)),
        f"{sample['ImageId'].min()}..{sample['ImageId'].max()}",
    )
    check(
        "submission columns are ['ImageId', 'Label']",
        list(sample.columns) == ["ImageId", "Label"],
        f"got {list(sample.columns)}",
    )

    section("Summary")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<52} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
