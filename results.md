# Results

Score log. **Val** is accuracy on the held-out stratified 20% split (8,400 images).
**LB** is the actual Kaggle leaderboard score on the 28,000-row test set.

Tracking both matters: a large gap between them signals leakage or distribution shift.

| # | Model | Params | Val acc | LB score | Diff | Notes |
|---|---|---|---|---|---|---|
| 1 | Linear baseline (784→10) | 7,850 | 89.26% | **0.89367** | −0.11 | SGD lr=0.01, 10 epochs, final epoch |
| 2 | MLP (784→128→10) | 101,770 | 91.08% | — | — | SGD lr=0.01, 10 epochs — undertrained |
| 3 | MLP, tuned | 101,770 | 97.55% | not submitted | — | Adam lr=0.001, 30 epochs |
| 4 | CNN (16 filters) | 31,530 | 98.02%* | **0.97792** | +0.23 | Adam lr=0.001, 15 epochs, *best of 15 |

\* Rung 4's validation figure is the **best of 15 epochs**, so it is optimistically
biased — see the selection-bias note below. Rung 1's is the final epoch, hence the
opposite sign on the diff.

Rung 2 → 3 is the same architecture. The +6.47 points came from the optimiser and
budget, not the model — see the ablation below.

---

## Noise floor

Validation set is 8,400 images, so accuracy carries a standard error of
`sqrt(p(1-p)/n)` = **0.338%**, giving a 95% Wilson interval of **±0.66 points**.

**A new model must beat 89.26% by more than 0.66 points to be provably better.**
Anything closer is inside the noise of the sample.

On the 28,000-row test set the same accuracy carries SE 0.185%, so leaderboard
scores are resolvable to about ±0.36 points.

---

## Stage 1 — Linear baseline

**Submitted 2026-09-23.** Predicted 89.26% ± 0.36 → [88.90%, 89.62%].
Actual **0.89367**, inside the interval.

This confirms three things:

1. **The pipeline is correct end-to-end.** A misaligned submission would have scored
   near 10%, not 89.4%.
2. **Local validation tracks the leaderboard.** Predicted in step 1 from train/test
   pixel means being near-identical (33.41 vs 33.35). Local CV can now be trusted to
   guide decisions without spending submissions.
3. **No leakage and no distribution shift.** LB slightly above val is the harmless
   direction and well inside sampling noise.

### Where the errors are

| True → Predicted | Count | % of class |
|---|---|---|
| 4 → 9 | 51 | 6.3% |
| 7 → 9 | 41 | 4.7% |
| 5 → 3 | 36 | 4.7% |
| 5 → 8 | 35 | 4.6% |
| 9 → 4 | 33 | 3.9% |
| 3 → 5 | 33 | 3.8% |

Errors cluster on **4/9** and **3/5/8** — pairs distinguished by stroke topology
(closed vs open loops, crossing strokes). A linear model holds one weight per pixel
position and cannot represent "closed loop", so this is a **capacity** limit, not a
tuning limit. No learning-rate schedule fixes it.

Worst class: digit 5 (F1 82.76%, recall 79.71%) — also the rarest class (3,795).
Best class: digit 0 (F1 95.07%).

### Model confidence

Mean 81.88%, median 90.02%, with **10.0% of test predictions below 50% confidence**
(2,808 images the model is effectively guessing on).

### Diagnosis

Train loss 0.3875 vs val loss 0.3897 — essentially no gap, so the model is
**underfitting**. 7,850 parameters cannot memorise 33,600 images. Regularisation
(dropout, augmentation) would help nothing here; the bottleneck is capacity, which is
what convolution addresses.

---

## Stage 2-3 — MLP, and the optimiser ablation

### The 2x2

|  | 10 epochs | 30 epochs | Budget effect |
|---|---|---|---|
| **SGD (lr=0.01)** | 91.08% | 93.81% | +2.73 |
| **Adam (lr=0.001)** | 96.95% | 97.55% | +0.60 |
| **Optimiser effect** | +5.87 | +3.74 | |

```
total gain              +6.46 points
  optimiser (at 30ep)   +3.74
  budget (with Adam)    +0.60
  interaction           -2.13
```

**Adam at 10 epochs beats SGD at 30 epochs** (96.95% vs 93.81%) at a third of the
compute. The epoch budget was never the bottleneck — the optimiser was. Changing both
at once would have hidden this and led to wasting compute on long runs forever after.

The negative interaction: extra epochs are worth +2.73 to SGD but only +0.60 to Adam,
because Adam has already converged by epoch 10.

### Convergence

| Epoch | Train loss | Val loss | Val acc |
|---|---|---|---|
| 10 | 0.0321 | **0.0958** (min) | 96.95% |
| 20 | 0.0055 | 0.1037 | 97.57% |
| 24 | — | — | **97.60%** (max) |
| 30 | 0.0063 | 0.1251 | 97.55% |

Val loss bottomed at epoch 10; val accuracy peaked at 24. They diverge because
cross-entropy punishes confident errors while accuracy ignores confidence. Early stop
on loss if calibrated probabilities matter, on accuracy if only the argmax does.

**Now in the overfitting regime** — train/val gap went from −0.0022 (linear,
underfitting) to +0.1188 (MLP, memorising). Dropout and augmentation become useful
from here; they would have done nothing for the linear baseline.

### Significance

McNemar, MLP+Adam vs MLP+SGD on the same 8,400 images: 589 fixed, 46 broken,
net +543, chi-square 462.62, p < 0.001.

### Lesson

The first MLP at 91.08% looked like a weak architecture. It was an **undertrained**
one — same 101,770 parameters reached 97.55% once optimised properly. Confirm
convergence before concluding anything about an architecture.

---

## Stage 4 — CNN, and a lesson in selection bias

`Conv2d(1→16, k=3, pad=1) → ReLU → MaxPool2d(2,2) → Flatten → Linear(3136→10)`

31,530 parameters — **31% of the MLP's** — yet it scores higher. Only 160 of those
weights do feature extraction; one 3×3 filter is reused across all 784 positions.
That reuse, not capacity, is the win.

### Validation vs leaderboard, and why the sign flipped

| Model | Val acc | How reported | LB | Val − LB |
|---|---|---|---|---|
| Linear | 89.26% | final epoch | 0.89367 | −0.11 |
| CNN | 98.02% | best of 15 epochs | 0.97792 | **+0.23** |

Taking the max of 15 noisy epoch measurements is **optimistically biased** —
`E[max(x₁…x₁₅)] > true value`, because the maximum preferentially picks epochs where
noise happened to be positive. The linear model was reported at its final epoch, so it
carried no such bias.

**The CNN's true accuracy is nearer 97.8% than 98.02%.** The leaderboard estimate is
both larger (28,000 rows, ±0.17 at 95%) and unselected, so it is the honest one.

**Rule going forward:** compare models the same way — all at final epoch, or all at
best-of-N. Mixing favours whichever got the generous treatment.

### Significance vs the noise floor

- Gain over MLP: **+0.47 points**, which is *inside* the ±0.66 noise floor.
- McNemar: 118 fixed, 78 broken, net +40, χ² 7.76, **p = 0.005 — significant**.

Both correct; they ask different questions. The ±0.66 floor treats the two scores as
independent samples. McNemar uses the fact that both models were scored on the **same**
8,400 images, removing the shared variance. **The paired test wins when it applies.**

### Error reallocation

| Pair | Linear | MLP | CNN |
|---|---|---|---|
| 4 → 9 | 51 | 10 | **4** |
| 7 → 9 | 41 | 4 | **2** |
| 5 → 3 | 36 | 7 | **3** |
| 3 → 5 | 33 | 15 | **7** |
| 9 → 4 | 33 | 10 | 13 |
| 5 → 8 | 35 | 5 | 7 |

The loop-topology confusions identified at stage 1 largely collapsed. But **9→4 got
worse** (10→13) and 5→8 ticked up. A better model is not uniformly better — it
reallocates errors, and aggregate accuracy hides that.

### Parameters are not compute

| Model | Params | Sec/epoch |
|---|---|---|
| MLP | 101,770 | ~2.3s |
| CNN | 31,530 | ~11s |

A third of the parameters, ~5× the epoch time. Convolution re-applies its kernel at
every spatial position (~113k MACs/image for conv1 alone), plus memory traffic for the
12,544-element intermediate tensor. Never estimate training time from parameter count.

---

## Reproducing

```
.venv\Scripts\python.exe step1_sanity_check.py     # data integrity, 11/11 checks
.venv\Scripts\python.exe step2_baseline_model.py   # dataloaders + architecture
.venv\Scripts\python.exe step3_train_loop.py       # 10-epoch training loop
.venv\Scripts\python.exe step4_submission.py       # train, evaluate, write submission
.venv\Scripts\python.exe initial_loss_check.py     # ln(C) verification
```

All seeded with `SEED=42`, so results are reproducible run to run.
Artifacts: `baseline_model.pt` (checkpoint), `submissions/submission.csv`.
