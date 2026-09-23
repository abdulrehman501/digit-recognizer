"""The initial-loss check: verifying that an untrained classifier starts at ln(C).

Proves the law empirically across class counts, shows how batch size affects it,
and reproduces the failure modes it is designed to catch.

Run:  python initial_loss_check.py
"""

import math

import torch
import torch.nn as nn

SEED = 42
N_FEATURES = 784


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def initial_loss(n_classes: int, batch_size: int, seed: int = SEED) -> float:
    """Loss of a freshly initialised Linear classifier on random data."""
    torch.manual_seed(seed)
    model = nn.Linear(N_FEATURES, n_classes)
    x = torch.rand(batch_size, N_FEATURES)              # already in [0,1]
    y = torch.randint(0, n_classes, (batch_size,))
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(x), y).item()


def law_across_class_counts() -> None:
    section("1. The law: initial loss = ln(C)")
    print(f"  {'classes':>8}  {'ln(C)':>8}  {'measured':>9}  {'diff':>7}  {'example'}")
    print("  " + "-" * 60)
    examples = {
        2: "binary (spam/not-spam)",
        10: "MNIST digits",
        26: "letters A-Z",
        100: "CIFAR-100",
        1000: "ImageNet",
    }
    for c, label in examples.items():
        expected = math.log(c)
        actual = initial_loss(c, batch_size=4096)
        print(f"  {c:>8}  {expected:>8.4f}  {actual:>9.4f}  {abs(actual - expected):>7.4f}  {label}")
    print("\n  Why: an untrained model spreads probability evenly, so each class gets 1/C.")
    print("  Cross-entropy of that prediction is -ln(1/C) = ln(C).")


def batch_size_effect() -> None:
    section("2. Larger batches converge tighter to the theoretical value")
    c = 10
    expected = math.log(c)
    print(f"  target: ln({c}) = {expected:.4f}\n")
    print(f"  {'batch':>7}  {'measured':>9}  {'diff':>7}")
    print("  " + "-" * 27)
    for batch in (8, 32, 64, 256, 1024, 8192):
        actual = initial_loss(c, batch_size=batch)
        print(f"  {batch:>7}  {actual:>9.4f}  {abs(actual - expected):>7.4f}")
    print("\n  Small batches fluctuate; the law is about the expectation, not one sample.")
    print("  A batch of 64 landing at 2.29 instead of 2.30 is normal, not a bug.")


def failure_modes() -> None:
    section("3. The failure modes this check catches")
    c = 10
    expected = math.log(c)
    batch = 4096
    results = []

    # Baseline: everything correct.
    results.append(("correct setup", initial_loss(c, batch), "-"))

    # Unnormalised inputs: pixels left at 0-255 instead of 0-1.
    torch.manual_seed(SEED)
    model = nn.Linear(N_FEATURES, c)
    x_raw = torch.rand(batch, N_FEATURES) * 255.0
    y = torch.randint(0, c, (batch,))
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(x_raw), y).item()
    results.append(("inputs not divided by 255", loss, "forgot normalisation"))

    # Oversized initialisation.
    torch.manual_seed(SEED)
    model_big = nn.Linear(N_FEATURES, c)
    with torch.no_grad():
        model_big.weight *= 100
    x = torch.rand(batch, N_FEATURES)
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model_big(x), y).item()
    results.append(("weights initialised 100x too large", loss, "bad init"))

    # Double softmax: softmax in forward AND inside CrossEntropyLoss.
    torch.manual_seed(SEED)
    model = nn.Linear(N_FEATURES, c)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        loss = nn.CrossEntropyLoss()(probs, y).item()
    results.append(("softmax applied before the loss", loss, "double softmax"))

    # Label leakage: the label is handed to the model as a feature.
    torch.manual_seed(SEED)
    leaky = nn.Linear(c, c)
    with torch.no_grad():
        leaky.weight.copy_(torch.eye(c) * 10)
        leaky.bias.zero_()
        one_hot = torch.nn.functional.one_hot(y, c).float()
        loss = nn.CrossEntropyLoss()(leaky(one_hot), y).item()
    results.append(("label leaked in as a feature", loss, "leakage"))

    print(f"  target: ln({c}) = {expected:.4f}\n")
    print(f"  {'scenario':<36}  {'loss':>9}  {'verdict':>9}  {'diagnosis'}")
    print("  " + "-" * 82)
    for name, loss, diagnosis in results:
        if math.isnan(loss):
            verdict = "NaN"
        elif abs(loss - expected) < 0.10:
            verdict = "ok"
        elif loss > expected:
            verdict = "TOO HIGH"
        else:
            verdict = "TOO LOW"
        print(f"  {name:<36}  {loss:>9.4f}  {verdict:>9}  {diagnosis}")

    print("\n  Note on double softmax: the initial loss stays near ln(C), so this check")
    print("  does NOT catch it. Squashing logits into [0,1] leaves them nearly equal,")
    print("  which still looks uniform. It shows up later as training that barely moves.")


def what_good_looks_like() -> None:
    section("4. Reference table")
    print(f"  {'classes':>8}  {'ln(C)':>8}  {'random acc':>11}")
    print("  " + "-" * 31)
    for c in (2, 3, 5, 10, 26, 100, 1000):
        print(f"  {c:>8}  {math.log(c):>8.4f}  {1 / c:>10.2%}")
    print("\n  Both columns are what an untrained classifier should produce.")
    print("  Check them together: loss near ln(C) AND accuracy near 1/C.")


def main() -> None:
    print("Initial-loss check: does an untrained classifier start at ln(C)?")
    law_across_class_counts()
    batch_size_effect()
    failure_modes()
    what_good_looks_like()


if __name__ == "__main__":
    main()
