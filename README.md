# Digit Recognizer

Image classification on the MNIST handwritten digit dataset, via Kaggle's
[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
competition. Each sample is a 28x28 grayscale image of a handwritten digit
(0-9), flattened to 784 pixel-intensity columns; the task is to predict the
digit.

## Approach

Baseline models first (scikit-learn), then iterate toward higher accuracy.
Status and results below are updated as work progresses.

| Stage | Status |
|---|---|
| Data loaded | done |
| Exploratory analysis | pending |
| Baseline model | pending |
| Model iteration | pending |
| Kaggle submission | pending |

## Project structure

```
data/
  raw/          competition CSVs (train.csv, test.csv, sample_submission.csv) - not tracked
  processed/    derived/cleaned datasets - not tracked
notebooks/      exploratory notebooks
src/            reusable modules (data loading, features, models)
scripts/        one-off / CLI scripts (e.g. data fetch)
submissions/    generated Kaggle submission CSVs
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data isn't tracked in git (Kaggle competition data, plus it's tens of MB).
Download it from the
[competition data page](https://www.kaggle.com/competitions/digit-recognizer/data)
into `data/raw/`, or via the Kaggle CLI once `~/.kaggle/kaggle.json` is set up:

```
kaggle competitions download -c digit-recognizer -p data/raw
```

## Stack

Python 3.11, pandas, scikit-learn, numpy, matplotlib, seaborn, JupyterLab.
