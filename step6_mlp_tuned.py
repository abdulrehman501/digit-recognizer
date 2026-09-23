"""Step 6: find the MLP's real ceiling — and disentangle optimiser from budget.

The obvious move is to switch SGD -> Adam AND 10 -> 30 epochs at once. That changes
two variables simultaneously, so the resulting score cannot be attributed to either.

This runs the full 2x2 instead:

                10 epochs   30 epochs
    SGD            ?            ?
    Adam           ?            ?

so the optimiser effect and the budget effect are separable.

Run:  python step6_mlp_tuned.py
"""

import math
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
    PIXEL_MAX,
    SEED,
    build_dataloaders,
    section,
    set_seed,
)
from step3_train_loop import run_epoch
from step4_submission import SUBMISSION_DIR, wilson_interval
from step5_mlp import MLP, predictions

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "mlp_tuned.pt"

SGD_LR = 0.01
ADAM_LR = 0.001
SHORT, LONG = 10, 30

BASELINE_ACC = 0.8926   # linear, step 4
MLP_SGD10_ACC = 0.9108  # MLP + SGD + 10 epochs, step 5


def make_optimizer(name: str, model: nn.Module) -> optim.Optimizer:
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=SGD_LR)
    return optim.Adam(model.parameters(), lr=ADAM_LR)


def train(name: str, epochs: int, train_loader, val_loader, verbose: bool = False):
    """Train a fresh MLP from the same seed. Returns per-epoch history."""
    set_seed(SEED)
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(name, model)

    history = []
    if verbose:
        print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}  {'gap':>8}")
        print("  " + "-" * 48)
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        history.append((epoch, train_loss, val_loss, val_acc))
        if verbose and (epoch <= 5 or epoch % 5 == 0 or epoch == epochs):
            print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}  "
                  f"{val_loss - train_loss:>+8.4f}")
    return model, history


def ablation(train_loader, val_loader) -> dict:
    section("2x2 ablation: optimiser x epoch budget")
    print("  Same seed, same split, same architecture, same batch size.")
    print("  Only the optimiser and the number of epochs vary.\n")

    results = {}
    for opt_name in ("SGD", "Adam"):
        for epochs in (SHORT, LONG):
            _, history = train(opt_name, epochs, train_loader, val_loader)
            final_acc = history[-1][3]
            best_epoch, _, best_vloss, best_acc = min(history, key=lambda r: r[2])
            results[(opt_name, epochs)] = {
                "final_acc": final_acc,
                "best_acc": max(h[3] for h in history),
                "best_vloss_epoch": best_epoch,
                "history": history,
            }
            lr = SGD_LR if opt_name == "SGD" else ADAM_LR
            print(f"  {opt_name:>4} lr={lr:<6} {epochs:>2} epochs -> {final_acc:.2%} "
                  f"(best {max(h[3] for h in history):.2%})")

    section("Attributing the gain")
    sgd10 = results[("SGD", SHORT)]["final_acc"]
    sgd30 = results[("SGD", LONG)]["final_acc"]
    adam10 = results[("Adam", SHORT)]["final_acc"]
    adam30 = results[("Adam", LONG)]["final_acc"]

    print(f"  {'':16}{'10 epochs':>12}{'30 epochs':>12}{'budget effect':>15}")
    print("  " + "-" * 55)
    print(f"  {'SGD':16}{sgd10:>11.2%}{sgd30:>12.2%}{sgd30 - sgd10:>+14.2%}")
    print(f"  {'Adam':16}{adam10:>11.2%}{adam30:>12.2%}{adam30 - adam10:>+14.2%}")
    print(f"  {'optimiser effect':16}{adam10 - sgd10:>+11.2%}{adam30 - sgd30:>+12.2%}")

    print(f"\n  total gain from doing both: {adam30 - sgd10:+.2%} points")
    print(f"    attributable to optimiser (at 30 epochs): {adam30 - sgd30:+.2%}")
    print(f"    attributable to budget (with Adam):       {adam30 - adam10:+.2%}")
    interaction = (adam30 - sgd30) - (adam10 - sgd10)
    print(f"    interaction between the two:              {interaction:+.2%}")
    print("\n  A non-zero interaction means the two changes are not independent -")
    print("  the value of extra epochs depends on which optimiser you use.")
    return results


def detailed_run(train_loader, val_loader):
    section(f"Adam lr={ADAM_LR}, {LONG} epochs — full curve")
    model, history = train("Adam", LONG, train_loader, val_loader, verbose=True)

    section("Convergence analysis")
    best_epoch, _, best_vloss, _ = min(history, key=lambda r: r[2])
    best_acc_epoch, _, _, best_acc = max(history, key=lambda r: r[3])
    final = history[-1]

    print(f"  lowest val loss      {best_vloss:.4f} at epoch {best_epoch}")
    print(f"  highest val accuracy {best_acc:.2%} at epoch {best_acc_epoch}")
    print(f"  final (epoch {LONG})       {final[3]:.2%}, val loss {final[2]:.4f}")

    if best_epoch < LONG:
        print(f"\n  Val loss bottomed at epoch {best_epoch} and rose after -> OVERFITTING.")
        print(f"  The model kept fitting training noise for {LONG - best_epoch} more epochs.")
        print(f"  Early stopping at epoch {best_epoch} is the correct call.")
    else:
        print(f"\n  Val loss still falling at epoch {LONG} - not yet converged.")

    gap = final[2] - final[1]
    print(f"\n  final train loss {final[1]:.4f} vs val loss {final[2]:.4f} (gap {gap:+.4f})")
    if gap > 0.05:
        print("  Positive gap: the model now fits training data better than validation.")
        print("  This is the overfitting regime the linear baseline never reached.")
    return model, history


def mcnemar(y_true, pred_a, pred_b, label_a: str, label_b: str) -> None:
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    n_dis = only_a + only_b

    section(f"{label_b} vs {label_a} (McNemar)")
    print(f"  {label_a} right / {label_b} wrong : {only_a:>5}")
    print(f"  {label_a} wrong / {label_b} right : {only_b:>5}")
    print(f"  net                              : {only_b - only_a:>+5} images")
    if n_dis == 0:
        print("  identical predictions")
        return
    stat = (abs(only_b - only_a) - 1) ** 2 / n_dis
    p = 1 - chi2.cdf(stat, df=1)
    print(f"  chi-square {stat:.2f}, p = {p:.2e}"
          f"  -> {'significant' if p < 0.05 else 'NOT significant'}")


def write_submission(model: MLP) -> Path:
    section("Submission")
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
    print(f"  mean confidence {confidence.mean():.2%}  "
          f"(linear 81.88%, MLP+SGD 87.45%)")
    print(f"  below 50% confident: {low} ({low / len(labels):.1%})")

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / "submission_mlp_tuned.csv"
    pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels}).to_csv(path, index=False)
    w = pd.read_csv(path)
    ok = (len(w) == 28000 and list(w.columns) == ["ImageId", "Label"]
          and np.array_equal(w["ImageId"].to_numpy(), np.arange(1, 28001))
          and w["Label"].between(0, 9).all() and w.isna().sum().sum() == 0)
    print(f"  written {path.relative_to(ROOT)}  ({'valid' if ok else 'INVALID'})")
    return path


def main() -> None:
    train_loader, val_loader = build_dataloaders()
    ablation(train_loader, val_loader)
    model, history = detailed_run(train_loader, val_loader)
    torch.save(model.state_dict(), CHECKPOINT)

    pred_new, y_true = predictions(model, val_loader)
    n = len(y_true)
    acc = (pred_new == y_true).mean()
    lo, hi = wilson_interval(int((pred_new == y_true).sum()), n)

    section("Result")
    print(f"  linear baseline        {BASELINE_ACC:.2%}")
    print(f"  MLP + SGD, 10 epochs   {MLP_SGD10_ACC:.2%}")
    print(f"  MLP + Adam, 30 epochs  {acc:.2%}   95% CI [{lo:.2%}, {hi:.2%}]")
    print(f"\n  gain over linear   {acc - BASELINE_ACC:+.2%} points")
    print(f"  gain over MLP+SGD  {acc - MLP_SGD10_ACC:+.2%} points")
    print(f"\n  the ladder predicted linear -> MLP recovers ~8 points")
    print(f"  properly trained, it recovered {(acc - BASELINE_ACC) * 100:.2f}")

    prev = MLP()
    prev.load_state_dict(torch.load(ROOT / "mlp_model.pt"))
    pred_prev, _ = predictions(prev, val_loader)
    mcnemar(y_true, pred_prev, pred_new, "MLP+SGD", "MLP+Adam")

    write_submission(model)


if __name__ == "__main__":
    main()
