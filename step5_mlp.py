"""Step 5: rung one of the ladder — add a hidden layer.

Controlled experiment: identical seed, split, optimiser, learning rate, batch size
and epoch count as the linear baseline. The ONLY change is the architecture, so any
difference in score is attributable to the hidden layer.

Compares against the saved baseline with McNemar's test, the correct paired test for
two classifiers evaluated on the same data.

Run:  python step5_mlp.py
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2

from step2_baseline_model import (
    BATCH_SIZE,
    DATA_DIR,
    N_CLASSES,
    N_PIXELS,
    PIXEL_MAX,
    SEED,
    BaselineModel,
    build_dataloaders,
    section,
    set_seed,
)
from step3_train_loop import EPOCHS, LEARNING_RATE, run_epoch
from step4_submission import CHECKPOINT as BASELINE_CHECKPOINT
from step4_submission import SUBMISSION_DIR, wilson_interval

ROOT = Path(__file__).resolve().parent
MLP_CHECKPOINT = ROOT / "mlp_model.pt"
HIDDEN_UNITS = 128

# Measured in step 4, for reference.
BASELINE_VAL_ACC = 0.8926
BASELINE_LB = 0.89367


class MLP(nn.Module):
    """784 -> 128 -> 10 with a ReLU between. One hidden layer, nothing else."""

    def __init__(self, hidden: int = HIDDEN_UNITS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_PIXELS, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))  # logits, no softmax


def predictions(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images).argmax(dim=1))
            targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def compare_architectures() -> None:
    section("What changed")
    linear_p = sum(p.numel() for p in BaselineModel().parameters())
    mlp = MLP()
    mlp_p = sum(p.numel() for p in mlp.parameters())

    print(f"  baseline  Linear(784 -> 10)")
    print(f"            {N_PIXELS} x {N_CLASSES} + {N_CLASSES} = {linear_p:,} parameters")
    print(f"\n  MLP       Linear(784 -> {HIDDEN_UNITS}) -> ReLU -> Linear({HIDDEN_UNITS} -> 10)")
    print(f"            fc1: {N_PIXELS} x {HIDDEN_UNITS} + {HIDDEN_UNITS} = {N_PIXELS * HIDDEN_UNITS + HIDDEN_UNITS:,}")
    print(f"            fc2: {HIDDEN_UNITS} x {N_CLASSES} + {N_CLASSES} = {HIDDEN_UNITS * N_CLASSES + N_CLASSES:,}")
    print(f"            total {mlp_p:,} parameters ({mlp_p / linear_p:.1f}x the baseline)")
    print(f"\n  The ReLU is the entire point. Without it, Linear->Linear collapses")
    print(f"  algebraically into a single Linear layer and buys nothing.")


def train_mlp(train_loader, val_loader) -> MLP:
    set_seed(SEED)
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Same sanity check we run on every classifier.
    pre_loss, pre_acc = run_epoch(model, val_loader, criterion, optimizer=None)
    section("Initial-loss check")
    print(f"  val loss {pre_loss:.4f} vs ln({N_CLASSES}) = {math.log(N_CLASSES):.4f} "
          f"(diff {abs(pre_loss - math.log(N_CLASSES)):.4f})")
    print(f"  val accuracy {pre_acc:.2%} (expect ~{1 / N_CLASSES:.0%})")

    section(f"Training (identical settings: SGD lr={LEARNING_RATE}, {EPOCHS} epochs, batch {BATCH_SIZE})")
    print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}  {'vs base':>8}  {'time':>6}")
    print("  " + "-" * 56)
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        elapsed = time.perf_counter() - started
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}  "
              f"{val_acc - BASELINE_VAL_ACC:>+7.2%}  {elapsed:>5.1f}s")

    torch.save(model.state_dict(), MLP_CHECKPOINT)
    print(f"\n  checkpoint saved: {MLP_CHECKPOINT.name}")
    return model


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> None:
    """Paired test for two classifiers on the same samples."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    both = int((a_ok & b_ok).sum())
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    neither = int((~a_ok & ~b_ok).sum())

    section("Is the improvement real? (McNemar's test)")
    print("  Comparing the two models on the SAME 8,400 images, pair by pair.")
    print("  Only the disagreements carry information.\n")
    print(f"  {'':22}{'MLP right':>12}{'MLP wrong':>12}")
    print(f"  {'baseline right':22}{both:>12}{only_a:>12}")
    print(f"  {'baseline wrong':22}{only_b:>12}{neither:>12}")

    n_disagree = only_a + only_b
    print(f"\n  baseline right / MLP wrong : {only_a:>5}   (MLP broke these)")
    print(f"  baseline wrong / MLP right : {only_b:>5}   (MLP fixed these)")
    print(f"  net improvement            : {only_b - only_a:>+5} images")

    # Continuity-corrected McNemar statistic, chi-square with 1 df.
    stat = (abs(only_b - only_a) - 1) ** 2 / n_disagree
    p = 1 - chi2.cdf(stat, df=1)
    print(f"\n  chi-square statistic : {stat:.2f}  (1 degree of freedom)")
    print(f"  p-value              : {p:.2e}")
    if p < 0.001:
        print("  -> p < 0.001. The improvement is real, not sampling noise.")
    elif p < 0.05:
        print("  -> p < 0.05. Statistically significant.")
    else:
        print("  -> not significant. The difference could be chance.")
    print("\n  Why this beats comparing confidence intervals: the models saw identical")
    print("  images, so the comparison is paired. That removes the between-sample")
    print("  variance and detects smaller real differences than overlapping CIs would.")


def error_analysis(y_true: np.ndarray, pred_base: np.ndarray, pred_mlp: np.ndarray) -> None:
    section("Did it fix the confusions we identified?")
    pairs = [(4, 9), (7, 9), (5, 3), (5, 8), (9, 4), (3, 5)]
    print(f"  {'confusion':>12}  {'baseline':>9}  {'MLP':>6}  {'change':>7}")
    print("  " + "-" * 40)
    for t, p in pairs:
        base_n = int(((y_true == t) & (pred_base == p)).sum())
        mlp_n = int(((y_true == t) & (pred_mlp == p)).sum())
        print(f"  {f'{t} -> {p}':>12}  {base_n:>9}  {mlp_n:>6}  {mlp_n - base_n:>+7}")

    section("Per-class recall")
    print(f"  {'digit':>5}  {'baseline':>9}  {'MLP':>8}  {'change':>8}")
    print("  " + "-" * 36)
    for c in range(N_CLASSES):
        mask = y_true == c
        base_r = (pred_base[mask] == c).mean()
        mlp_r = (pred_mlp[mask] == c).mean()
        print(f"  {c:>5}  {base_r:>9.2%}  {mlp_r:>8.2%}  {mlp_r - base_r:>+8.2%}")


def write_submission(model: MLP) -> Path:
    section("Test inference and submission")
    test = pd.read_csv(DATA_DIR / "test.csv")
    x_test = torch.tensor(test.to_numpy(), dtype=torch.float32) / PIXEL_MAX

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_test), BATCH_SIZE):
            chunks.append(model(x_test[start:start + BATCH_SIZE]))
    logits = torch.cat(chunks)
    labels = logits.argmax(dim=1).numpy()
    confidence = torch.softmax(logits, dim=1).max(dim=1).values.numpy()

    low = int((confidence < 0.5).sum())
    print(f"  predicted {len(labels)} labels")
    print(f"  mean confidence {confidence.mean():.2%} (baseline was 81.88%)")
    print(f"  below 50% confident: {low} ({low / len(labels):.1%})  [baseline: 2808, 10.0%]")

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / "submission_mlp.csv"
    pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels}).to_csv(path, index=False)

    written = pd.read_csv(path)
    ok = (
        len(written) == 28000
        and list(written.columns) == ["ImageId", "Label"]
        and np.array_equal(written["ImageId"].to_numpy(), np.arange(1, 28001))
        and written["Label"].between(0, 9).all()
        and written.isna().sum().sum() == 0
    )
    print(f"\n  written: {path.relative_to(ROOT)}  ({'valid' if ok else 'INVALID'})")
    return path


def main() -> None:
    train_loader, val_loader = build_dataloaders()
    compare_architectures()

    model = train_mlp(train_loader, val_loader)

    pred_mlp, y_true = predictions(model, val_loader)
    baseline = BaselineModel()
    baseline.load_state_dict(torch.load(BASELINE_CHECKPOINT))
    pred_base, _ = predictions(baseline, val_loader)

    n = len(y_true)
    acc_mlp = (pred_mlp == y_true).mean()
    acc_base = (pred_base == y_true).mean()
    lo, hi = wilson_interval(int((pred_mlp == y_true).sum()), n)

    section("Result")
    print(f"  baseline (linear)  {acc_base:.2%}")
    print(f"  MLP (1 hidden)     {acc_mlp:.2%}   95% CI [{lo:.2%}, {hi:.2%}]")
    print(f"  gain               {acc_mlp - acc_base:+.2%} points")
    print(f"\n  noise floor was +/-0.66 points. This gain is "
          f"{'WELL ABOVE' if (acc_mlp - acc_base) > 0.0066 else 'inside'} it.")
    print(f"\n  predicted ladder said linear -> MLP recovers ~8 points.")
    print(f"  actually recovered: {(acc_mlp - acc_base) * 100:.2f} points.")

    mcnemar(y_true, pred_base, pred_mlp)
    error_analysis(y_true, pred_base, pred_mlp)
    write_submission(model)


if __name__ == "__main__":
    main()
