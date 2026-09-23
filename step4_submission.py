"""Step 4: train the baseline, evaluate it properly, and generate submission.csv.

Adds the statistics the raw accuracy number hides: a binomial confidence interval on
the validation estimate, a confusion matrix, per-class precision/recall/F1, and a
prediction-distribution check against the training prior.

Run:  python step4_submission.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from step2_baseline_model import (
    BATCH_SIZE,
    DATA_DIR,
    N_CLASSES,
    PIXEL_MAX,
    SEED,
    BaselineModel,
    build_dataloaders,
    section,
    set_seed,
)
from step3_train_loop import EPOCHS, LEARNING_RATE, run_epoch

ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submissions"
CHECKPOINT = ROOT / "baseline_model.pt"
Z_95 = 1.959964  # two-sided 95% normal quantile


def wilson_interval(correct: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval - better than the normal approximation near 0 or 1."""
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - margin, centre + margin


def train_model() -> BaselineModel:
    train_loader, val_loader = build_dataloaders()

    set_seed(SEED)
    model = BaselineModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    section("Training")
    print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}")
    print("  " + "-" * 40)
    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\n  checkpoint saved: {CHECKPOINT.name} "
          f"({CHECKPOINT.stat().st_size / 1024:.1f} KB)")
    return model, val_loader


def evaluate(model: BaselineModel, val_loader) -> np.ndarray:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds.append(model(images).argmax(dim=1))
            targets.append(labels)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    n = len(y_true)
    correct = int((y_pred == y_true).sum())
    acc = correct / n

    section("How reliable is that accuracy?")
    # Accuracy is a proportion from n Bernoulli trials, so it carries sampling error.
    se = math.sqrt(acc * (1 - acc) / n)
    lo, hi = wilson_interval(correct, n)
    print(f"  validation accuracy  {acc:.2%}  ({correct} of {n})")
    print(f"  standard error       {se:.4%}   = sqrt(p(1-p)/n)")
    print(f"  95% CI (Wilson)      [{lo:.2%}, {hi:.2%}]   width {(hi - lo) * 100:.2f} points")
    print(f"\n  Any future model within +/-{(hi - lo) / 2 * 100:.2f} points of this is NOT")
    print("  provably better - the difference is inside the noise of an 8,400-row sample.")

    # The test set is larger, so its estimate is tighter.
    se_test = math.sqrt(acc * (1 - acc) / 28000)
    print(f"\n  on 28,000 test rows the same accuracy would carry SE {se_test:.4%}")
    print(f"  so expect a leaderboard score near {acc:.2%} +/- {Z_95 * se_test * 100:.2f} points")

    section("Confusion matrix (rows = true, cols = predicted)")
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print("       " + "".join(f"{c:>6}" for c in range(N_CLASSES)) + "   recall")
    for c in range(N_CLASSES):
        row = "".join(f"{v:>6}" for v in cm[c])
        recall = cm[c, c] / cm[c].sum()
        print(f"   {c}  {row}   {recall:>6.2%}")

    section("Per-class precision / recall / F1")
    print(f"  {'digit':>5}  {'precision':>9}  {'recall':>7}  {'F1':>7}  {'support':>7}  {'errors':>7}")
    print("  " + "-" * 52)
    f1s = []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        precision = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        recall = tp / cm[c].sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        print(f"  {c:>5}  {precision:>9.2%}  {recall:>7.2%}  {f1:>7.2%}  "
              f"{cm[c].sum():>7}  {cm[c].sum() - tp:>7}")
    print(f"\n  macro F1: {np.mean(f1s):.2%}   (unweighted mean - treats every digit equally)")

    section("Worst confusions")
    off = [(cm[i, j], i, j) for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j]
    for count, i, j in sorted(off, reverse=True)[:6]:
        print(f"  true {i} predicted as {j}:  {count:>3} times  "
              f"({count / cm[i].sum():.1%} of all {i}s)")

    return y_true, y_pred


def predict_test(model: BaselineModel) -> pd.DataFrame:
    section("Test set inference")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  test.csv  {test.shape[0]} rows x {test.shape[1]} pixels")

    x_test = torch.tensor(test.to_numpy(), dtype=torch.float32) / PIXEL_MAX
    print(f"  tensor    {tuple(x_test.shape)}  range [{x_test.min():.1f}, {x_test.max():.1f}]")

    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(x_test), BATCH_SIZE):
            all_logits.append(model(x_test[start:start + BATCH_SIZE]))
    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=1)
    labels = logits.argmax(dim=1).numpy()
    confidence = probs.max(dim=1).values.numpy()

    print(f"  predicted {len(labels)} labels")
    print(f"  mean confidence {confidence.mean():.2%}, median {np.median(confidence):.2%}")
    low = (confidence < 0.5).sum()
    print(f"  {low} predictions ({low / len(labels):.1%}) below 50% confidence")

    section("Does the prediction distribution look sane?")
    # Predicted class shares should roughly track the training prior. A wild
    # deviation means the model collapsed onto a few classes.
    train_prior = pd.read_csv(DATA_DIR / "train.csv")["label"].value_counts(normalize=True).sort_index()
    print(f"  {'digit':>5}  {'train %':>8}  {'pred %':>8}  {'diff':>7}")
    print("  " + "-" * 34)
    worst = 0.0
    for c in range(N_CLASSES):
        pred_pct = (labels == c).mean() * 100
        tr_pct = train_prior[c] * 100
        diff = pred_pct - tr_pct
        worst = max(worst, abs(diff))
        print(f"  {c:>5}  {tr_pct:>7.2f}%  {pred_pct:>7.2f}%  {diff:>+6.2f}pp")
    print(f"\n  largest deviation {worst:.2f}pp - "
          f"{'sane' if worst < 3 else 'SUSPICIOUS, model may be collapsing'}")

    return pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels})


def write_and_validate(submission: pd.DataFrame) -> Path:
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(path, index=False)

    section("Submission validation")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    written = pd.read_csv(path)

    checks = [
        ("file written", path.exists(), str(path.relative_to(ROOT))),
        ("row count is 28000", len(written) == 28000, f"{len(written)}"),
        ("columns are ImageId, Label", list(written.columns) == ["ImageId", "Label"],
         f"{list(written.columns)}"),
        ("matches sample_submission shape", written.shape == sample.shape,
         f"{written.shape} vs {sample.shape}"),
        ("ImageId is 1..28000 contiguous",
         np.array_equal(written["ImageId"].to_numpy(), np.arange(1, 28001)), "contiguous"),
        ("labels all within 0..9",
         written["Label"].between(0, 9).all(), f"[{written['Label'].min()}, {written['Label'].max()}]"),
        ("no missing values", written.isna().sum().sum() == 0, f"{int(written.isna().sum().sum())}"),
        ("all 10 digits predicted", written["Label"].nunique() == 10,
         f"{written['Label'].nunique()} distinct"),
    ]
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<34} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed.")

    print("\n  first 5 rows:")
    for line in written.head(5).to_string(index=False).splitlines():
        print(f"    {line}")

    if failed:
        raise SystemExit(1)
    return path


def main() -> None:
    model, val_loader = train_model()
    evaluate(model, val_loader)
    submission = predict_test(model)
    path = write_and_validate(submission)

    section("Done")
    print(f"  submission ready: {path}")
    print(f"  size {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
