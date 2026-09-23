"""Step 9: the same CNN, trained on augmented data.

Controlled comparison against step 7: identical architecture, optimiser, learning
rate, batch size and seed. The only change is that training images are randomly
transformed on every access.

Runs longer than step 7 because augmentation makes each epoch harder - the model
never sees the same image twice, so it converges more slowly but to a better point.
Reports the epoch-15 value for a like-for-like comparison, plus the best overall.

Run:  python step9_cnn_augmented.py
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

from step2_baseline_model import DATA_DIR, N_CLASSES, PIXEL_MAX, SEED, section, set_seed
from step4_submission import SUBMISSION_DIR, wilson_interval
from step7_cnn import ADAM_LR, CNN, IMAGE_SIZE, predictions, run_epoch
from step8_augmented_dataset import build_loaders

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "cnn_augmented.pt"
BATCH = 64
EPOCHS = 25
COMPARE_EPOCH = 15   # step 7 ran this many

CNN_PLAIN_ACC = 0.9802
MLP_ACC = 0.9755
LINEAR_ACC = 0.8926


def train(train_loader, val_loader):
    set_seed(SEED)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)

    pre_loss, pre_acc = run_epoch(model, val_loader, criterion, None)
    section("Initial-loss check")
    print(f"  val loss {pre_loss:.4f} vs ln({N_CLASSES}) = {math.log(N_CLASSES):.4f} "
          f"(diff {abs(pre_loss - math.log(N_CLASSES)):.4f})   acc {pre_acc:.2%}")

    section(f"Training with augmentation (Adam lr={ADAM_LR}, {EPOCHS} epochs)")
    print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}  "
          f"{'gap':>8}  {'vs plain':>9}  {'time':>7}")
    print("  " + "-" * 68)

    history, best_state, best_acc = [], None, 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None)
        dt = time.perf_counter() - t0
        history.append((epoch, tr_loss, va_loss, va_acc))
        if va_acc > best_acc:
            best_acc, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {va_loss:>9.4f}  {va_acc:>7.2%}  "
              f"{va_loss - tr_loss:>+8.4f}  {va_acc - CNN_PLAIN_ACC:>+8.2%}  {dt:>6.1f}s")

    model.load_state_dict(best_state)
    return model, history


def analyse(history) -> None:
    section("Augmentation changed the overfitting picture")
    at15 = next(h for h in history if h[0] == COMPARE_EPOCH)
    best_loss_epoch, _, best_vloss, _ = min(history, key=lambda r: r[2])
    best_acc_epoch, _, _, best_acc = max(history, key=lambda r: r[3])
    final = history[-1]

    print(f"  epoch {COMPARE_EPOCH} (matches step 7's budget): {at15[3]:.2%}"
          f"   vs plain CNN {CNN_PLAIN_ACC:.2%}  ({at15[3] - CNN_PLAIN_ACC:+.2%})")
    print(f"  best val accuracy  {best_acc:.2%} at epoch {best_acc_epoch}")
    print(f"  lowest val loss    {best_vloss:.4f} at epoch {best_loss_epoch}")
    print(f"  final train/val gap {final[2] - final[1]:+.4f}")

    print(f"\n  plain CNN (step 7):     val loss bottomed epoch 9, gap +0.0510")
    print(f"  augmented CNN:          val loss bottomed epoch {best_loss_epoch}, "
          f"gap {final[2] - final[1]:+.4f}")
    if final[2] - final[1] < 0.0510:
        print("  -> the gap shrank. Augmentation is doing its job as a regulariser.")
    if best_loss_epoch > 9:
        print(f"  -> overfitting onset moved later ({best_loss_epoch} vs 9), so the")
        print("     model can train longer before it starts memorising.")


def mcnemar(y_true, pred_a, pred_b, label_a, label_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    section(f"{label_b} vs {label_a} (McNemar)")
    print(f"  {label_a} right / {label_b} wrong : {only_a:>5}")
    print(f"  {label_a} wrong / {label_b} right : {only_b:>5}")
    print(f"  net                            : {only_b - only_a:>+5}")
    n = only_a + only_b
    if n:
        stat = (abs(only_b - only_a) - 1) ** 2 / n
        p = 1 - chi2.cdf(stat, df=1)
        print(f"  chi-square {stat:.2f}, p = {p:.2e} -> "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")


def error_analysis(y_true, pred_plain, pred_aug) -> None:
    section("Confusion trail across every rung")
    tracked = {(4, 9): (51, 10, 4), (7, 9): (41, 4, 2), (5, 3): (36, 7, 3),
               (3, 5): (33, 15, 7), (9, 4): (33, 10, 13), (5, 8): (35, 5, 7)}
    print(f"  {'pair':>10}  {'linear':>7}  {'MLP':>6}  {'CNN':>6}  {'CNN+aug':>8}")
    print("  " + "-" * 44)
    for (t, p), (lin, mlp, cnn) in tracked.items():
        aug = int(((y_true == t) & (pred_aug == p)).sum())
        print(f"  {f'{t} -> {p}':>10}  {lin:>7}  {mlp:>6}  {cnn:>6}  {aug:>8}")

    section("Per-class recall")
    print(f"  {'digit':>5}  {'CNN':>8}  {'CNN+aug':>9}  {'change':>8}")
    print("  " + "-" * 36)
    for c in range(N_CLASSES):
        m = y_true == c
        a, b = (pred_plain[m] == c).mean(), (pred_aug[m] == c).mean()
        print(f"  {c:>5}  {a:>8.2%}  {b:>9.2%}  {b - a:>+8.2%}")


def write_submission(model) -> Path:
    section("Submission")
    test = pd.read_csv(DATA_DIR / "test.csv")
    x = torch.tensor(test.to_numpy(), dtype=torch.float32).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE) / PIXEL_MAX
    model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(x), BATCH):
            chunks.append(model(x[s:s + BATCH]))
    logits = torch.cat(chunks)
    labels = logits.argmax(dim=1).numpy()
    conf = torch.softmax(logits, dim=1).max(dim=1).values.numpy()
    low = int((conf < 0.5).sum())
    print(f"  mean confidence {conf.mean():.2%}   below 50%: {low} ({low / len(labels):.1%})")

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / "submission_cnn_augmented.csv"
    pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels}).to_csv(path, index=False)
    w = pd.read_csv(path)
    ok = (len(w) == 28000 and list(w.columns) == ["ImageId", "Label"]
          and np.array_equal(w["ImageId"].to_numpy(), np.arange(1, 28001))
          and w["Label"].between(0, 9).all() and w.isna().sum().sum() == 0)
    print(f"  written {path.relative_to(ROOT)}  ({'valid' if ok else 'INVALID'})")
    return path


def main() -> None:
    train_loader, val_loader = build_loaders(augment=True)
    model, history = train(train_loader, val_loader)
    torch.save(model.state_dict(), CHECKPOINT)
    analyse(history)

    pred_aug, y_true = predictions(model, val_loader)
    plain = CNN()
    plain.load_state_dict(torch.load(ROOT / "cnn_model.pt"))
    pred_plain, _ = predictions(plain, val_loader)

    acc = (pred_aug == y_true).mean()
    lo, hi = wilson_interval(int((pred_aug == y_true).sum()), len(y_true))

    section("Final ladder")
    print(f"  {'rung':<32}{'params':>10}{'val acc':>10}")
    print("  " + "-" * 52)
    print(f"  {'1. linear':<32}{'7,850':>10}{LINEAR_ACC:>10.2%}")
    print(f"  {'2. MLP tuned':<32}{'101,770':>10}{MLP_ACC:>10.2%}")
    print(f"  {'3. CNN':<32}{'31,530':>10}{CNN_PLAIN_ACC:>10.2%}")
    print(f"  {'4. CNN + augmentation':<32}{'31,530':>10}{acc:>10.2%}")
    print(f"\n  95% CI [{lo:.2%}, {hi:.2%}]")
    print(f"  gain over plain CNN {acc - CNN_PLAIN_ACC:+.2%} points")
    print(f"  total gain over linear baseline {acc - LINEAR_ACC:+.2%} points")

    mcnemar(y_true, pred_plain, pred_aug, "CNN", "CNN+aug")
    error_analysis(y_true, pred_plain, pred_aug)
    write_submission(model)


if __name__ == "__main__":
    main()
