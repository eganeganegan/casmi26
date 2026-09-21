# Repository instructions for coding agents

This file is the authoritative instruction set for AI coding agents working in this repository.
Read it together with `CONTRIBUTING.md` before changing code. Tool-specific instruction files may add
guidance, but they must not weaken or contradict this file.

## Project objective

Build a research-quality, leakage-safe solution for the Enveda CASMI 2026 molecule-identification
competition. The production system is a hybrid of spectral retrieval, analog propagation, database
candidate generation, fingerprint prediction, fragmentation features, and learned reranking.

The final Kaggle notebook must run with internet disabled, finish within nine hours, and write
`/kaggle/working/submission.csv` with one row per `molecule_id`, columns `molecule_id` and `smiles`,
and no more than 25 semicolon-separated candidate SMILES per row.

## Non-negotiable research rules

- Preserve structure-disjoint validation. Never introduce the query structure into its retrieval
  library, candidate labels, training features, or model-selection evidence.
- Use fold 1 for model and hyperparameter selection. Treat fold 0 as an untouched report set; never
  tune a method after inspecting fold-0 results and then report that result as untouched.
- Evaluate candidate recall separately from ranking quality. A ranker cannot recover structures that
  candidate generation omitted.
- Compare changes against the current selected baseline with MRR@25, Hits@1/5/10/25, candidate
  recall, and runtime. Record rejected as well as promoted experiments.
- Test inference with internet disabled and keep generous headroom below Kaggle's nine-hour limit.
- Match the competition's RDKit tautomer normalization and InChIKey14 comparison behavior.

## Repository and artifact boundary

Commit source, tests, configuration, lightweight notebooks, and documentation. Do not commit:

- Kaggle credentials, `.env` files, keys, tokens, or credential JSON;
- competition parquet files or generated submissions;
- external molecular databases or separately licensed data;
- trained models, feature matrices, indexes, caches, predictions, wheels, or Kaggle upload bundles.

These paths are intentionally ignored. Do not use `git add -f` to bypass the boundary. Share approved
artifacts as versioned team Kaggle datasets with checksums, provenance, and license information.

## Development workflow

Work on a focused branch using the naming and commit rules in `CONTRIBUTING.md`. Preserve unrelated
user changes and inspect the working tree before editing. Prefer explicit configuration and CLI
arguments over local paths or hidden global state.

Useful commands:

```bash
.venv/bin/pip install -e '.[dev,ml]'
make lint
make test
make smoke
make check
```

Run focused tests while iterating and `make check` before handing work back. If production inference
changes, also run `make offline-inference` and validate the resulting submission. Do not claim a model
improvement without the saved, comparable evaluation report.

## Code standards

- Support Python 3.11 and 3.12 and use type hints throughout.
- Keep large-data operations streaming or array-backed; do not materialize millions of Python objects.
- Use deterministic seeds for sampling, training, and splits.
- Add tests for behavior changes and regression tests for bug fixes.
- Keep optional integrations isolated from the baseline.
- Use `pathlib.Path`, clear error messages, and atomic or recoverable writes where practical.
- Prefer existing package utilities and scripts over duplicating chemistry, preprocessing, or metrics.
- Run Ruff and do not silence a rule unless the exception is deliberate and documented.

## Documentation and handoff

Update `README.md` for user-facing workflow changes and `CHANGELOG.md` for experiment results or
meaningful implementation boundaries. A model experiment handoff must state its hypothesis, data and
split, baseline, metrics, runtime, artifact changes, and promote/reject decision.

Do not edit generated experiment reports to manufacture consistency. If documentation and an artifact
disagree, investigate and report the discrepancy.
