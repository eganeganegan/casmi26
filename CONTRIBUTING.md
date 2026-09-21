# Contributing

Thanks for helping with CASMI 2026! This repository separates reviewable source code from large,
licensed, or competition-controlled artifacts. A fresh clone should be useful without copying another teammate's working directory.

## Set up a development environment

Python 3.11 or newer is required. The full test suite imports the machine-learning components, so
install both development and ML extras:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev,ml]'
make check
```

Competition data is not required for unit tests. To reproduce experiments, accept the competition
rules, authenticate with Kaggle outside the repository, and run `make download`.

## Work on a branch

Create a short branch from `main`, such as `feature/formula-channel` or `fix/adduct-parser`. Keep each
pull request focused on one experiment or infrastructure change. Do not mix generated artifacts with
source changes.

Before opening a pull request:

```bash
make check
git status --short
```

Describe the validation split, comparison baseline, primary metric, runtime, and whether the result
was promoted or rejected. Record meaningful experiment conclusions in `CHANGELOG.md`; do not commit
multi-gigabyte feature matrices merely to preserve a result.

## Data, models, and secrets

Never commit:

- Kaggle tokens, `.env` files, private keys, or credential JSON;
- competition parquet files or generated `submission.csv` files;
- external databases whose licenses require separate distribution;
- trained weights, packed indexes, caches, predictions, or Kaggle upload bundles.

The relevant paths are ignored by Git. Share approved model/data artifacts through the team's Kaggle
datasets and include their version, checksum, provenance, and license in the experiment notes. Keep
placeholder `.gitkeep` files so the expected directory layout survives a clean clone.

## Code and experiment standards

- Preserve structure-disjoint validation and never use fold-0 results to select hyperparameters.
- Keep inference internet-free and below the competition's nine-hour notebook limit.
- Add or update tests for behavior changes.
- Use deterministic seeds and explicit CLI/config inputs instead of local absolute paths.
- Keep optional integrations optional; the baseline package must remain importable without them.
- Format imports and catch common defects with Ruff; run the complete tests before review.

## Pull-request review

At least one teammate should review changes that affect chemistry normalization, leakage boundaries,
candidate generation, model features, or submission formatting. A result should only replace the
selected model when its untouched report improves and the complete offline pipeline still validates.
