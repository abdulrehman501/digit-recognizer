"""Step 7: rung three — convolution.

Conv2d(1->16, k=3, pad=1) -> ReLU -> MaxPool2d(2,2) -> Flatten -> Linear(3136->10)

Traces tensor shapes through every layer, since shape mismatches are where CNNs
actually break. Trains with Adam (step 6 established it beats SGD decisively here).

Run:  python step7_cnn.py
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
    PIXEL_MAX,
    SEED,
    build_dataloaders,
    section,
    set_seed,
)
from step4_submission import SUBMISSION_DIR, wilson_interval
from step5_mlp import MLP

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "cnn_model.pt"

IMAGE_SIZE = 28
ADAM_LR = 0.001
EPOCHS = 15
CONV_CHANNELS = 16
KERNEL = 3
PADDING = 1
POOL = 2

MLP_ACC = 0.9755       # step 6, Adam 30 epochs
LINEAR_ACC = 0.8926    # step 4


def conv_out(size: int, kernel: int, padding: int, stride: int = 1) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class CNN(nn.Module):
    """One conv block, then a linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS, kernel_size=KERNEL, padding=PADDING)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=POOL, stride=POOL)
        self.flatten = nn.Flatten()
        after_conv = conv_out(IMAGE_SIZE, KERNEL, PADDING)   # 28
        after_pool = after_conv // POOL                       # 14
        self.n_features = CONV_CHANNELS * after_pool * after_pool  # 3136
        self.fc = nn.Linear(self.n_features, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Incoming batches are flat [B, 784]; conv needs [B, C, H, W].
        if x.dim() == 2:
            x = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = self.pool(self.relu(self.conv1(x)))
        return self.fc(self.flatten(x))


def trace_shapes() -> None:
    section("Shape trace (the part that actually breaks)")
    model = CNN()
    x = torch.zeros(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
    print(f"  {'layer':<34}{'output shape':>22}{'elements/img':>14}")
    print("  " + "-" * 70)

    def show(label: str, t: torch.Tensor) -> None:
        per_img = int(np.prod(t.shape[1:]))
        print(f"  {label:<34}{str(tuple(t.shape)):>22}{per_img:>14,}")

    show("input", x)
    x = model.conv1(x)
    show(f"Conv2d(1->{CONV_CHANNELS}, k={KERNEL}, pad={PADDING})", x)
    x = model.relu(x)
    show("ReLU", x)
    x = model.pool(x)
    show(f"MaxPool2d({POOL}, {POOL})", x)
    x = model.flatten(x)
    show("Flatten", x)
    x = model.fc(x)
    show(f"Linear({model.n_features} -> {N_CLASSES})", x)

    section("The spatial maths")
    after_conv = conv_out(IMAGE_SIZE, KERNEL, PADDING)
    after_pool = after_conv // POOL
    print(f"  conv:  ({IMAGE_SIZE} + 2*{PADDING} - {KERNEL}) / 1 + 1 = {after_conv}"
          f"   padding={PADDING} preserves the grid")
    print(f"  pool:  ({after_conv} - {POOL}) / {POOL} + 1 = {after_pool}"
          f"   halves each spatial dimension")
    print(f"  flat:  {CONV_CHANNELS} channels x {after_pool} x {after_pool} = "
          f"{model.n_features:,}  <- in_features for the Linear layer")

    section("Dummy tensor test")
    out = model(torch.randn(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))
    ok = tuple(out.shape) == (BATCH_SIZE, N_CLASSES)
    print(f"  input  [{BATCH_SIZE}, 1, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"  output {tuple(out.shape)}  -> {'PASS' if ok else 'FAIL'}")

    section("Parameters: where convolution wins")
    conv_p = sum(p.numel() for p in model.conv1.parameters())
    fc_p = sum(p.numel() for p in model.fc.parameters())
    total = conv_p + fc_p
    mlp_total = sum(p.numel() for p in MLP().parameters())
    print(f"  conv1   {CONV_CHANNELS} filters x (1 x {KERNEL} x {KERNEL}) + {CONV_CHANNELS} bias "
          f"= {conv_p:>7,}")
    print(f"  fc      {model.n_features:,} x {N_CLASSES} + {N_CLASSES}"
          f"{'':<18}= {fc_p:>7,}")
    print(f"  total                                          {total:>7,}")
    print(f"\n  MLP (784->128->10) for comparison:             {mlp_total:>7,}")
    print(f"  the CNN uses {total / mlp_total:.0%} of the MLP's parameters.")
    print(f"\n  Only {conv_p} weights do the feature extraction. One 3x3 filter is")
    print(f"  reused across all 784 positions - that reuse is the whole advantage.")
    print(f"  An MLP needs a separate weight per pixel per unit and cannot share.")


def run_epoch(model, loader, criterion, optimizer):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_correct = total_seen = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += n
    return total_loss / total_seen, total_correct / total_seen


def predictions(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images).argmax(dim=1))
            targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def train(train_loader, val_loader):
    set_seed(SEED)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)

    pre_loss, pre_acc = run_epoch(model, val_loader, criterion, None)
    section("Initial-loss check")
    print(f"  val loss {pre_loss:.4f} vs ln({N_CLASSES}) = {math.log(N_CLASSES):.4f} "
          f"(diff {abs(pre_loss - math.log(N_CLASSES)):.4f})")
    print(f"  val accuracy {pre_acc:.2%}")

    section(f"Training (Adam lr={ADAM_LR}, {EPOCHS} epochs)")
    print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}  {'vs MLP':>8}  {'time':>7}")
    print("  " + "-" * 58)
    history, best_state, best_acc = [], None, 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None)
        dt = time.perf_counter() - t0
        history.append((epoch, tr_loss, va_loss, va_acc))
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {va_loss:>9.4f}  {va_acc:>7.2%}  "
              f"{va_acc - MLP_ACC:>+7.2%}  {dt:>6.1f}s")

    best_epoch, _, best_vloss, _ = min(history, key=lambda r: r[2])
    section("Convergence")
    print(f"  lowest val loss  {best_vloss:.4f} at epoch {best_epoch}")
    print(f"  best val acc     {best_acc:.2%}")
    print(f"  final gap        {history[-1][2] - history[-1][1]:+.4f} (val - train loss)")
    if best_epoch < EPOCHS:
        print(f"  val loss rose after epoch {best_epoch} - overfitting; "
              f"restoring best weights")
    model.load_state_dict(best_state)
    return model, history


def mcnemar(y_true, pred_a, pred_b, label_a, label_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    section(f"{label_b} vs {label_a} (McNemar)")
    print(f"  {label_a} right / {label_b} wrong : {only_a:>5}")
    print(f"  {label_a} wrong / {label_b} right : {only_b:>5}")
    print(f"  net                             : {only_b - only_a:>+5}")
    n = only_a + only_b
    if n:
        stat = (abs(only_b - only_a) - 1) ** 2 / n
        p = 1 - chi2.cdf(stat, df=1)
        print(f"  chi-square {stat:.2f}, p = {p:.2e} -> "
              f"{'significant' if p < 0.05 else 'NOT significant'}")


def error_analysis(y_true, pred_mlp, pred_cnn):
    section("The confusions we have tracked since step 4")
    print(f"  {'pair':>10}  {'linear':>7}  {'MLP':>6}  {'CNN':>6}")
    print("  " + "-" * 34)
    baseline = {(4, 9): 51, (7, 9): 41, (5, 3): 36, (5, 8): 35, (9, 4): 33, (3, 5): 33}
    for (t, p), base_n in baseline.items():
        mlp_n = int(((y_true == t) & (pred_mlp == p)).sum())
        cnn_n = int(((y_true == t) & (pred_cnn == p)).sum())
        print(f"  {f'{t} -> {p}':>10}  {base_n:>7}  {mlp_n:>6}  {cnn_n:>6}")

    section("Per-class recall")
    print(f"  {'digit':>5}  {'MLP':>8}  {'CNN':>8}  {'change':>8}")
    print("  " + "-" * 34)
    for c in range(N_CLASSES):
        mask = y_true == c
        print(f"  {c:>5}  {(pred_mlp[mask] == c).mean():>8.2%}  "
              f"{(pred_cnn[mask] == c).mean():>8.2%}  "
              f"{(pred_cnn[mask] == c).mean() - (pred_mlp[mask] == c).mean():>+8.2%}")


def write_submission(model):
    section("Submission")
    test = pd.read_csv(DATA_DIR / "test.csv")
    x = torch.tensor(test.to_numpy(), dtype=torch.float32) / PIXEL_MAX
    model.eval()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(x), BATCH_SIZE):
            chunks.append(model(x[s:s + BATCH_SIZE]))
    logits = torch.cat(chunks)
    labels = logits.argmax(dim=1).numpy()
    conf = torch.softmax(logits, dim=1).max(dim=1).values.numpy()
    low = int((conf < 0.5).sum())
    print(f"  mean confidence {conf.mean():.2%}   below 50%: {low} ({low / len(labels):.1%})")

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / "submission_cnn.csv"
    pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels}).to_csv(path, index=False)
    w = pd.read_csv(path)
    ok = (len(w) == 28000 and list(w.columns) == ["ImageId", "Label"]
          and np.array_equal(w["ImageId"].to_numpy(), np.arange(1, 28001))
          and w["Label"].between(0, 9).all() and w.isna().sum().sum() == 0)
    print(f"  written {path.relative_to(ROOT)}  ({'valid' if ok else 'INVALID'})")


def main() -> None:
    trace_shapes()
    train_loader, val_loader = build_dataloaders()
    model, _ = train(train_loader, val_loader)
    torch.save(model.state_dict(), CHECKPOINT)

    pred_cnn, y_true = predictions(model, val_loader)
    mlp = MLP()
    mlp.load_state_dict(torch.load(ROOT / "mlp_tuned.pt"))
    pred_mlp, _ = predictions(mlp, val_loader)

    acc = (pred_cnn == y_true).mean()
    lo, hi = wilson_interval(int((pred_cnn == y_true).sum()), len(y_true))

    section("Ladder so far")
    print(f"  {'rung':<28}{'val acc':>9}")
    print("  " + "-" * 37)
    print(f"  {'1. linear (7,850)':<28}{LINEAR_ACC:>9.2%}")
    print(f"  {'2. MLP tuned (101,770)':<28}{MLP_ACC:>9.2%}")
    print(f"  {'3. CNN (31,530)':<28}{acc:>9.2%}")
    print(f"\n  CNN 95% CI [{lo:.2%}, {hi:.2%}]")
    print(f"  gain over MLP {acc - MLP_ACC:+.2%} points "
          f"({'above' if abs(acc - MLP_ACC) > 0.0066 else 'INSIDE'} the +/-0.66 noise floor)")

    mcnemar(y_true, pred_mlp, pred_cnn, "MLP", "CNN")
    error_analysis(y_true, pred_mlp, pred_cnn)
    write_submission(model)


if __name__ == "__main__":
    main()
