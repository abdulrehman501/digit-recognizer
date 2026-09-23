"""Step 2: baseline model — data ingestion and preparation.

Loads train.csv, makes a stratified 80/20 train/validation split, normalises pixels
to [0,1], converts to tensors, and wraps them in DataLoaders.

Run:  python step2_baseline_model.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent / "data"

SEED = 42
VAL_FRACTION = 0.2
BATCH_SIZE = 64
PIXEL_MAX = 255.0
N_CLASSES = 10
N_PIXELS = 784


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def set_seed(seed: int) -> None:
    """Without this the split changes every run and scores are not comparable."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    set_seed(SEED)

    section("Load")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y = train_df["label"].to_numpy()
    x = train_df.drop(columns="label").to_numpy()
    print(f"  train.csv      {x.shape[0]} rows x {x.shape[1]} pixels")

    section("Stratified split")
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=VAL_FRACTION,
        stratify=y,
        random_state=SEED,
    )
    print(f"  train  {len(x_train):>6} rows  ({1 - VAL_FRACTION:.0%})")
    print(f"  val    {len(x_val):>6} rows  ({VAL_FRACTION:.0%})")

    # Verify the stratification actually held, rather than trusting that it did.
    print("\n  digit   train%    val%     diff")
    worst = 0.0
    for digit in range(N_CLASSES):
        tr_pct = (y_train == digit).mean() * 100
        va_pct = (y_val == digit).mean() * 100
        diff = abs(tr_pct - va_pct)
        worst = max(worst, diff)
        print(f"    {digit}    {tr_pct:6.2f}%  {va_pct:6.2f}%   {diff:.3f}pp")
    print(f"\n  largest train/val gap: {worst:.3f} percentage points")

    section("Normalise and tensorise")
    # float32 not float64: half the memory, and what the network layers expect.
    xt_train = torch.tensor(x_train, dtype=torch.float32) / PIXEL_MAX
    xt_val = torch.tensor(x_val, dtype=torch.float32) / PIXEL_MAX
    # long (int64) because CrossEntropyLoss expects integer class indices.
    yt_train = torch.tensor(y_train, dtype=torch.long)
    yt_val = torch.tensor(y_val, dtype=torch.long)

    print(f"  x_train  {tuple(xt_train.shape)}  {xt_train.dtype}  range [{xt_train.min():.1f}, {xt_train.max():.1f}]")
    print(f"  x_val    {tuple(xt_val.shape)}  {xt_val.dtype}  range [{xt_val.min():.1f}, {xt_val.max():.1f}]")
    print(f"  y_train  {tuple(yt_train.shape)}  {yt_train.dtype}  labels {yt_train.min()}..{yt_train.max()}")
    print(f"  y_val    {tuple(yt_val.shape)}  {yt_val.dtype}  labels {yt_val.min()}..{yt_val.max()}")
    # Input data does not need gradients - only model parameters do.
    print(f"  requires_grad on inputs: {xt_train.requires_grad}  (correct: data is not learned)")

    section("DataLoaders")
    train_loader = DataLoader(
        TensorDataset(xt_train, yt_train),
        batch_size=BATCH_SIZE,
        shuffle=True,  # break any ordering the model could memorise
    )
    val_loader = DataLoader(
        TensorDataset(xt_val, yt_val),
        batch_size=BATCH_SIZE,
        shuffle=False,  # grading only, order is irrelevant
    )

    train_last = len(xt_train) % BATCH_SIZE or BATCH_SIZE
    val_last = len(xt_val) % BATCH_SIZE or BATCH_SIZE
    print(f"  train  {len(train_loader):>4} batches of {BATCH_SIZE}  (last batch: {train_last})")
    print(f"  val    {len(val_loader):>4} batches of {BATCH_SIZE}  (last batch: {val_last})")

    xb, yb = next(iter(train_loader))
    print(f"\n  first batch: x {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)} {yb.dtype}")
    print(f"  labels in that batch: {yb[:16].tolist()} ...")

    return train_loader, val_loader


class BaselineModel(nn.Module):
    """Multinomial logistic regression: one linear layer, 784 -> 10, no hidden layer."""

    def __init__(self, n_inputs: int = N_PIXELS, n_outputs: int = N_CLASSES) -> None:
        super().__init__()
        self.layer = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits. No softmax here - nn.CrossEntropyLoss applies
        # log_softmax internally, and doing it twice breaks the gradients.
        return self.layer(x)


def inspect_model(model: BaselineModel, xb: torch.Tensor, yb: torch.Tensor) -> None:
    section("Architecture")
    print(f"  {model}")

    section("Parameter count")
    w, b = model.layer.weight, model.layer.bias
    n_w, n_b = w.numel(), b.numel()
    print(f"  weight  {tuple(w.shape)}  = {N_CLASSES} x {N_PIXELS} = {n_w:>5}")
    print(f"  bias    {tuple(b.shape)}       = {N_CLASSES:<12}   {n_b:>5}")
    print(f"  total                          {n_w + n_b:>5}")
    counted = sum(p.numel() for p in model.parameters())
    print(f"  torch agrees: {counted} ({'match' if counted == n_w + n_b else 'MISMATCH'})")

    section("Initialisation")
    # nn.Linear inits both weight and bias from U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
    bound = 1.0 / math.sqrt(N_PIXELS)
    print(f"  theoretical bound  +/-1/sqrt({N_PIXELS}) = +/-{bound:.6f}")
    print(f"  actual weight range  [{w.min():.6f}, {w.max():.6f}]")
    print(f"  actual bias   range  [{b.min():.6f}, {b.max():.6f}]")
    within = w.abs().max().item() <= bound + 1e-6 and b.abs().max().item() <= bound + 1e-6
    print(f"  all params within bound: {within}")
    print(f"  weight mean {w.mean():+.6f} (expect ~0), std {w.std():.6f} "
          f"(expect ~{bound / math.sqrt(3):.6f} for a uniform dist)")

    section("Forward pass")
    print(f"  input   {tuple(xb.shape)}")
    print(f"  matmul  [{xb.shape[0]}, {N_PIXELS}] @ [{N_PIXELS}, {N_CLASSES}] "
          f"-> [{xb.shape[0]}, {N_CLASSES}]  (+ bias broadcast)")
    model.eval()
    with torch.no_grad():
        logits = model(xb)
    print(f"  output  {tuple(logits.shape)}")
    print(f"  logit range [{logits.min():.4f}, {logits.max():.4f}] - raw scores, not probabilities")

    section("Logits -> probabilities (what CrossEntropyLoss does internally)")
    probs = torch.softmax(logits, dim=1)
    print(f"  softmax row sums: min {probs.sum(dim=1).min():.6f}, max {probs.sum(dim=1).max():.6f} (must be 1.0)")
    print(f"  sample 0 probabilities: {[f'{p:.3f}' for p in probs[0].tolist()]}")
    print(f"  most confident prediction in batch: {probs.max():.4f} "
          f"(untrained, so should sit near {1 / N_CLASSES:.2f})")

    section("Sanity check: initial loss")
    # An untrained model is uniform over 10 classes, so loss = -ln(1/10) = ln(10).
    expected = math.log(N_CLASSES)
    with torch.no_grad():
        actual = nn.CrossEntropyLoss()(logits, yb).item()
    print(f"  theoretical  -ln(1/{N_CLASSES}) = ln({N_CLASSES}) = {expected:.4f}")
    print(f"  actual                          = {actual:.4f}")
    print(f"  difference                      = {abs(actual - expected):.4f}")
    print("  If this were far from ln(10), the wiring or init would be wrong.")

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == yb).float().mean().item()
    print(f"\n  untrained accuracy on this batch: {acc:.1%} (expect ~{1 / N_CLASSES:.0%}, random guessing)")


def main() -> None:
    train_loader, val_loader = build_dataloaders()

    section("Answer")
    print(f"  The training DataLoader reports {len(train_loader)} batches.")
    print(f"  {len(train_loader.dataset)} images / {BATCH_SIZE} per batch = "
          f"{len(train_loader.dataset) / BATCH_SIZE:.1f} -> {len(train_loader)} batches.")

    set_seed(SEED)
    xb, yb = next(iter(train_loader))
    inspect_model(BaselineModel(), xb, yb)


if __name__ == "__main__":
    main()
