"""Step 8: a custom Dataset so augmentations can be applied on the fly.

TensorDataset hands back a fixed tensor. To transform an image differently on every
epoch, we need our own Dataset subclass with a transform hook in __getitem__.

Critical rule: augment the TRAINING set only. The validation set must stay fixed, or
the score becomes a moving target and is not comparable between epochs or models.

Run:  python step8_augmented_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from step2_baseline_model import (
    BATCH_SIZE,
    DATA_DIR,
    N_CLASSES,
    PIXEL_MAX,
    SEED,
    VAL_FRACTION,
    section,
    set_seed,
)

IMAGE_SIZE = 28

# Ranges chosen to match the real variation seen in the EDA (step 1): handwriting
# tilts and drifts, but never mirrors. Keep them conservative - augmentation that
# changes the label is worse than no augmentation.
ROTATION_DEGREES = 10
TRANSLATE = 0.10
SCALE = (0.90, 1.10)
SHEAR = 5


class DigitDataset(Dataset):
    """MNIST digits with an optional transform applied per access.

    A Dataset must implement __len__ and __getitem__. The DataLoader calls
    __getitem__ once per image per epoch, so a random transform placed here
    produces a different variant every epoch - that is the whole point.
    """

    def __init__(self, pixels: np.ndarray, labels: np.ndarray, transform=None) -> None:
        # Keep as uint8 [N, 28, 28]: that is what torchvision transforms expect,
        # and it uses a quarter of the memory of float32.
        self.images = torch.tensor(pixels, dtype=torch.uint8).view(-1, IMAGE_SIZE, IMAGE_SIZE)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].unsqueeze(0)  # [1, 28, 28], channel dim for conv
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.float() / PIXEL_MAX
        return image, self.labels[idx]


def build_transform() -> transforms.Compose:
    """Label-preserving geometric jitter only. No flips - a mirrored 2 is not a 2,
    and a flipped 6 is a 9."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=ROTATION_DEGREES,
            translate=(TRANSLATE, TRANSLATE),
            scale=SCALE,
            shear=SHEAR,
            fill=0,  # background is 0; filling with anything else invents ink
        ),
        transforms.ConvertImageDtype(torch.float32),  # also scales uint8 -> [0,1]
    ])


def build_loaders(augment: bool = True) -> tuple[DataLoader, DataLoader]:
    set_seed(SEED)
    df = pd.read_csv(DATA_DIR / "train.csv")
    y = df["label"].to_numpy()
    x = df.drop(columns="label").to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=VAL_FRACTION, stratify=y, random_state=SEED
    )

    train_ds = DigitDataset(x_train, y_train, transform=build_transform() if augment else None)
    val_ds = DigitDataset(x_val, y_val, transform=None)  # never augment validation

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def verify() -> None:
    section("Dataset contract")
    train_loader, val_loader = build_loaders(augment=True)
    train_ds, val_ds = train_loader.dataset, val_loader.dataset
    print(f"  train  __len__ = {len(train_ds):,}   transform: {'yes' if train_ds.transform else 'no'}")
    print(f"  val    __len__ = {len(val_ds):,}   transform: "
          f"{'yes' if val_ds.transform else 'no'}  <- must be 'no'")

    img, label = train_ds[0]
    print(f"\n  __getitem__(0) -> image {tuple(img.shape)} {img.dtype} "
          f"range [{img.min():.2f}, {img.max():.2f}], label {label.item()}")
    print(f"  shape is [C, H, W] - conv-ready, no reshape needed in forward()")

    section("Augmentation is live (same index, different output)")
    # The key behaviour: calling __getitem__ twice on the SAME index gives
    # different pixels, because the transform re-samples each call.
    set_seed(SEED)
    variants = [train_ds[0][0] for _ in range(5)]
    base = variants[0]
    print(f"  index 0, accessed 5 times:")
    for i, v in enumerate(variants):
        diff = (v - base).abs().mean().item()
        ink = (v > 0).float().mean().item()
        print(f"    access {i + 1}:  mean abs diff vs first {diff:.4f}   ink coverage {ink:.1%}")
    identical = all(torch.equal(v, base) for v in variants[1:])
    print(f"\n  all identical: {identical}  <- must be False, or the transform is not firing")

    print("\n  Validation set, same test:")
    v_variants = [val_ds[0][0] for _ in range(3)]
    v_identical = all(torch.equal(v, v_variants[0]) for v in v_variants[1:])
    print(f"    all identical: {v_identical}  <- must be True, or scores are not comparable")

    section("Effective dataset size")
    print(f"  stored images        {len(train_ds):,}")
    print(f"  distinct variants    effectively unbounded (continuous transform space)")
    print(f"  over 15 epochs the model sees {len(train_ds) * 15:,} image presentations,")
    print(f"  almost all of them geometrically distinct.")
    print("\n  This is why augmentation fights overfitting: the model cannot memorise")
    print("  33,600 fixed images when it never sees the same image twice.")

    section("Label preservation")
    print(f"  rotation  +/-{ROTATION_DEGREES} degrees")
    print(f"  translate +/-{TRANSLATE:.0%}")
    print(f"  scale     {SCALE[0]:.2f} - {SCALE[1]:.2f}")
    print(f"  shear     +/-{SHEAR} degrees")
    print(f"  flips     NONE")
    print("\n  Horizontal flip turns 2 into a mirrored shape that is not a digit.")
    print("  Vertical flip turns 6 into 9 - the label becomes actively wrong.")
    print("  Large rotation does the same: 6 at 180 degrees is a 9.")
    print("  Augmentation must preserve the label, or you are training on wrong answers.")

    section("Batch check")
    xb, yb = next(iter(train_loader))
    print(f"  batch x {tuple(xb.shape)} {xb.dtype}  range [{xb.min():.2f}, {xb.max():.2f}]")
    print(f"  batch y {tuple(yb.shape)} {yb.dtype}  labels {yb.min().item()}..{yb.max().item()}")
    print(f"  train batches {len(train_loader)}, val batches {len(val_loader)}")

    section("Class balance survived")
    counts = torch.bincount(train_ds.labels, minlength=N_CLASSES)
    total = counts.sum().item()
    print("  " + "  ".join(f"{c}:{counts[c].item() / total:.2%}" for c in range(N_CLASSES)))
    print("\n  Augmentation transforms images, never labels - balance is unchanged.")


if __name__ == "__main__":
    verify()
