# Digit Recognizer

Handwritten digit classification on MNIST, via Kaggle's
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition.
42,000 labelled 28x28 grayscale images, 10 classes.

Built as a sequence of controlled experiments rather than a single model, so that every
accuracy gain is attributable to a specific change.

## Results

| # | Model | Parameters | Validation | Leaderboard |
|---|---|---|---|---|
| 1 | Linear (784→10) | 7,850 | 89.26% | **0.89367** |
| 2 | MLP (784→128→10), SGD | 101,770 | 91.08% | — |
| 3 | MLP, Adam | 101,770 | 97.55% | — |
| 4 | CNN (16 filters) | 31,530 | 98.02% | **0.97792** |
| 5 | CNN + augmentation | 31,530 | 97.77% | — |

Validation is a stratified 20% holdout (8,400 images). Leaderboard is the 28,000-row
Kaggle test set.

## Findings

**The optimiser mattered far more than the training budget.** Moving the MLP from SGD to
Adam and from 10 to 30 epochs gained 6.46 points. A 2x2 ablation separated the causes:

|  | 10 epochs | 30 epochs |
|---|---|---|
| SGD (lr=0.01) | 91.08% | 93.81% |
| Adam (lr=0.001) | 96.95% | 97.55% |

Adam at 10 epochs beats SGD at 30 while doing a third of the work. Changing both
variables together would have wrongly attributed the gain to the longer budget.

**Convolution won with a third of the parameters.** The CNN (31,530) beat the MLP
(101,770) because one 3x3 filter is reused at every spatial position. Only 160 weights
perform feature extraction. This is an inductive-bias win, not a capacity win.

**Augmentation regularised correctly but did not improve accuracy.** Random affine
transforms pushed the overfitting onset from epoch 9 to epoch 25 and drove the train/val
loss gap from +0.051 to -0.090 — textbook regularisation behaviour. Accuracy still fell
0.25 points, because the model was no longer variance-limited. With one conv block and 16
filters there is not enough capacity to absorb both the task and the added difficulty.
Augmentation needs a deeper network to pay off.

**Errors were diagnosed before they were fixed.** The linear baseline's confusion matrix
showed failures concentrated on 4/9 and 3/5/8 — digit pairs separated by stroke topology
(closed vs open loops), which a per-pixel linear model cannot represent. That analysis
predicted convolution would help, and it did:

| Confusion | Linear | MLP | CNN |
|---|---|---|---|
| 4 → 9 | 51 | 10 | 4 |
| 7 → 9 | 41 | 4 | 2 |
| 5 → 3 | 36 | 7 | 3 |
| 3 → 5 | 33 | 15 | 7 |

## Method

Each change was measured against the previous rung with the seed, split, batch size and
architecture held constant wherever possible.

**Significance testing.** Model comparisons use McNemar's test on paired predictions over
the same validation images rather than comparing confidence intervals, which discards the
pairing and is needlessly conservative. The CNN's +0.47 gain over the MLP falls inside the
±0.66 confidence-interval noise floor but is significant under McNemar (p = 0.005).

**Uncertainty is reported.** Accuracy on 8,400 samples carries a standard error of 0.34%,
so the 95% Wilson interval spans ±0.66 points. Any change smaller than that is not
demonstrated.

**Wiring is verified before training.** Every classifier is checked against the
theoretical initial loss of `ln(C)` — 2.3026 for 10 classes. `initial_loss_check.py`
verifies this law across class counts and reproduces the failure modes it catches
(unnormalised inputs give 153.17, oversized initialisation gives 60.08, label leakage
gives 0.0004).

**A caveat on rung 4.** Its validation figure is the best of 15 epochs, so it carries
selection bias — the maximum of repeated noisy measurements is optimistic. The leaderboard
score of 0.97792 is the honest estimate, and the 0.23-point gap is consistent with that
bias. Rung 1 was reported at its final epoch and shows no such gap.

## Structure

```
step1_sanity_check.py        data integrity checks (shapes, ranges, class balance)
step2_baseline_model.py      dataloaders, stratified split, linear architecture
step3_train_loop.py          training loop
step4_submission.py          evaluation statistics and submission generation
step5_mlp.py                 hidden layer
step6_mlp_tuned.py           optimiser vs budget ablation
step7_cnn.py                 convolution, with shape tracing
step8_augmented_dataset.py   custom Dataset with on-the-fly augmentation
step9_cnn_augmented.py       augmented training
initial_loss_check.py        ln(C) verification and failure-mode reproduction
src/data.py                  dataset loading
scripts/eda.py               exploratory analysis, writes reports/figures/
results.md                   full score log and per-stage analysis
```

Scripts are numbered to be run in order; each is self-contained and prints its own
results.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Competition data is not tracked. Download from the
[competition data page](https://www.kaggle.com/competitions/digit-recognizer/data) into
`data/`, or via the Kaggle CLI:

```
kaggle competitions download -c digit-recognizer -p data
```

Then:

```
python step1_sanity_check.py
python step4_submission.py
```

All runs are seeded (`SEED = 42`) and reproduce exactly.

## Stack

Python 3.12, PyTorch 2.14 (CPU), pandas, scikit-learn, numpy, matplotlib, seaborn, scipy.
