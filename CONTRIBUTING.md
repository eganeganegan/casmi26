# Contributing

Thanks for helping with CASMI 2026! This repository separates reviewable source code from large,
licensed, or competition-controlled artifacts. A fresh clone should be useful without copying another teammate's working directory.

Human contributors and coding assistants should read `AGENTS.md` before making changes. `CODEX.md`,
`COPILOT.md`, and `.github/copilot-instructions.md` provide tool-specific guidance without replacing
the shared repository rules. Current, claimable work is listed in `CONTRIBUTOR_TASKS.md`.

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

Do not commit directly to `main`. Start every change from an up-to-date local copy:

```bash
git switch main
git pull --ff-only origin main
git switch -c experiment/formula-conditioned-retrieval
```

Use a short, lowercase, hyphen-separated branch name with one of these prefixes:

- `experiment/` for model, retrieval, or validation experiments;
- `feature/` for production features that are not primarily experiments;
- `fix/` for defects;
- `docs/` for documentation only;
- `test/` for test-only changes;
- `chore/` for maintenance, dependencies, or CI.

Examples include `experiment/formula-channel`, `feature/polarity-filter`, `fix/adduct-parser`, and
`docs/kaggle-setup`. Keep each branch and pull request focused on one experiment or infrastructure
change. Do not mix generated artifacts with source changes.

## Create commits

Make small commits that leave the repository in a coherent state. Stage intended files explicitly,
review the staged patch, and run the quality gate before committing:

```bash
make check
git status --short
git add src/casmi/retrieval/example.py tests/test_example.py CHANGELOG.md
git diff --cached
git commit -m "experiment: add formula-conditioned retrieval"
```

Write commit subjects in the imperative mood, keep them concise, and use a descriptive prefix such as
`experiment:`, `feat:`, `fix:`, `test:`, `docs:`, or `chore:`. Avoid vague messages such as `updates`,
`changes`, or `work`. Separate unrelated work into different commits.

Never use `git add -f` to bypass an ignore rule for data, credentials, models, predictions, or other
generated artifacts. If a required source file is unexpectedly ignored, fix the ignore rule and ask
for review instead.

Before the first push, incorporate recent changes from `main` without rewriting a branch that other
people may be using:

```bash
git fetch origin
git merge origin/main
make check
git push -u origin experiment/formula-conditioned-retrieval
```

For later commits on the same branch, `git push` is sufficient. Open a pull request into `main`, fill
out the repository template, and wait for the required CI check and review. Do not merge your own pull
request while required feedback remains unresolved.

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
