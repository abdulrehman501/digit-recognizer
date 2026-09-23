"""Step 3: the universal training loop, run on the baseline linear model.

Trains BaselineModel (784 -> 10) with SGD for 10 epochs and reports per-epoch
train loss, validation loss and validation accuracy.

Run:  python step3_train_loop.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from step2_baseline_model import (
    BATCH_SIZE,
    N_CLASSES,
    SEED,
    BaselineModel,
    build_dataloaders,
    section,
    set_seed,
)

EPOCHS = 10
LEARNING_RATE = 0.01


def run_epoch(
    model: BaselineModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains when an optimizer is given, else evaluates.

    Returns (mean loss per sample, accuracy).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    # Weight by batch size rather than averaging per-batch: the final validation
    # batch holds 16 images, not 64, so a plain mean over batches would overweight it.
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # no_grad saves memory and compute during evaluation. Note it is NOT what stops
    # the model learning from validation data - never calling backward()/step() is.
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            if training:
                optimizer.zero_grad()        # 1. clear accumulated gradients

            logits = model(images)           # 2. forward pass
            loss = criterion(logits, labels)  # 3. compute loss

            if training:
                loss.backward()              # 4. backpropagate
                optimizer.step()             # 5. update weights

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_n

    return total_loss / total_seen, total_correct / total_seen


def main() -> None:
    train_loader, val_loader = build_dataloaders()

    set_seed(SEED)
    model = BaselineModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    section("Setup")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model       BaselineModel (Linear 784 -> {N_CLASSES}), {n_params:,} parameters")
    print(f"  loss        CrossEntropyLoss")
    print(f"  optimizer   SGD(lr={LEARNING_RATE})")
    print(f"  epochs      {EPOCHS}")
    print(f"  batch size  {BATCH_SIZE}  ({len(train_loader)} train / {len(val_loader)} val batches)")

    # Epoch 0: measure before any training, to confirm we start near ln(C).
    pre_loss, pre_acc = run_epoch(model, val_loader, criterion, optimizer=None)
    section("Before training (epoch 0)")
    print(f"  val loss {pre_loss:.4f}  vs  ln({N_CLASSES}) = {math.log(N_CLASSES):.4f}"
          f"   (diff {abs(pre_loss - math.log(N_CLASSES)):.4f})")
    print(f"  val accuracy {pre_acc:.2%}  (expect ~{1 / N_CLASSES:.0%})")

    section("Training")
    header = f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}  {'d(acc)':>7}  {'time':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    history = []
    prev_acc = pre_acc
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        elapsed = time.perf_counter() - started

        delta = val_acc - prev_acc
        prev_acc = val_acc
        history.append((epoch, train_loss, val_loss, val_acc))
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}  "
              f"{delta:>+7.2%}  {elapsed:>5.1f}s")

    section("Summary")
    best_epoch, _, best_vloss, best_acc = max(history, key=lambda r: r[3])
    final = history[-1]
    print(f"  best val accuracy   {best_acc:.2%}  (epoch {best_epoch}, val loss {best_vloss:.4f})")
    print(f"  final val accuracy  {final[3]:.2%}  (epoch {final[0]})")
    print(f"  loss fell           {pre_loss:.4f} -> {final[2]:.4f}"
          f"  ({(1 - final[2] / pre_loss):.1%} reduction)")
    print(f"  accuracy rose       {pre_acc:.2%} -> {final[3]:.2%}"
          f"  (+{(final[3] - pre_acc) * 100:.1f} points over random)")

    gap = final[1] - final[2]
    print(f"\n  train loss {final[1]:.4f} vs val loss {final[2]:.4f}  (gap {gap:+.4f})")
    if abs(gap) < 0.05:
        print("  Gap is small - the model is underfitting, not overfitting.")
        print("  Expected: a linear model lacks the capacity to memorise 33,600 images.")


if __name__ == "__main__":
    main()
