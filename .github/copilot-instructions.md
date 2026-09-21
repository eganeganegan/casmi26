# Copilot repository instructions

Follow `AGENTS.md` as the authoritative project guidance and `CONTRIBUTING.md` for the collaboration
workflow.

- This is a Python 3.11+ scientific package for leakage-safe CASMI 2026 molecule identification.
- Preserve structure-disjoint validation; use fold 1 for selection and fold 0 only for untouched
  reporting.
- Keep large parquet and spectral operations streaming or array-backed and deterministically seeded.
- Reuse existing chemistry, preprocessing, metric, and submission utilities instead of duplicating
  behavior.
- Add focused tests for changes. The repository gate is `make check`.
- Do not suggest hardcoded local paths, internet access during Kaggle inference, or committed secrets,
  data, models, caches, predictions, submissions, wheels, and upload bundles.
- Production output must be `submission.csv` with `molecule_id` and `smiles`, at most 25
  semicolon-separated candidates per molecule, and complete within nine hours offline.
- Update `README.md` for workflow changes and `CHANGELOG.md` for experiment results or material
  implementation changes.
