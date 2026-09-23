# Contributor task board

Last updated: 2026-09-23

This is the working backlog for contributors onboarding to CASMI 2026. Before claiming a task, read
`AGENTS.md` and `CONTRIBUTING.md`, check open issues and pull requests, and announce the task so two
people do not build the same experiment. Keep one task per branch.

## Current baseline

- Selected model assets: EXP027, public leaderboard MRR@25 `0.226`.
- Submission candidate: EXP029, which reuses EXP027 assets and lowers the direct-library gate from
  `0.95` to `0.90`; its leaderboard result is still pending.
- EXP029 fold-0 report: MRR@25 `0.22755`, Hits@1 `0.2118`, Hits@25 `0.2642`.
- Full offline inference: 400 example molecules in `450.25` seconds.
- Main bottleneck: candidate-database coverage. Current fold-0 database coverage is `0.194`, and
  Recall@5000 is `0.188`; ranking work cannot recover absent structures.

Do not retune EXP029 on fold 0. New model and hyperparameter choices must use fold 1, followed by one
frozen fold-0 report only after the decision is made.

## Highest-priority research tasks

### 1. Expand the candidate database

Suggested branch: `experiment/pubchem-candidate-expansion`

Build a licensed, reproducible, mass-indexed PubChem or other broad metabolite/natural-product
candidate source. Keep the builder streaming and emit the repository's standard candidate parquet
schema. Measure database coverage and mass-window recall on fold 1 before doing any reranking work.

Deliverables:

- downloader or documented acquisition procedure with version, URL, license, and checksum;
- streaming normalization and InChIKey14 deduplication;
- fold-1 coverage and Recall@25/100/500/5000 versus the expanded ChEMBL baseline;
- memory, artifact-size, and offline query-runtime measurements;
- tests using a tiny synthetic source fixture.

### 2. Add direct-library evidence to candidate ranking

Suggested branch: `experiment/direct-library-ranker`

Expose candidate-level direct spectral similarity, explained intensity, matched-peak count, support
across query spectra, and score margins. Keep the Class-1 simulation separate from the
structure-disjoint Class-2 validation path; do not let a query structure or its spectra leak into a
structure-disjoint retrieval library.

Deliverables:

- explicit leakage-safe feature-generation protocol;
- candidate features with unit tests and deterministic aggregation;
- fold-1 comparison against EXP027 using MRR@25, Hits@1/5/10/25, and recall;
- frozen fold-0 report only if the fold-1 decision promotes the method;
- offline inference runtime and artifact impact.

### 3. Multi-spectrum consensus and neutral-loss retrieval

Suggested branch: `experiment/multispectrum-consensus`

Replace max-only spectrum aggregation with a tested consensus rule and add a neutral-loss similarity
channel. Test whether agreement across collision energies, adducts, or polarities improves confidence
without increasing false Class-1 matches.

Deliverables:

- configurable aggregation implementation;
- synthetic tests covering one-spectrum and multi-spectrum molecules;
- fold-1 ablations for fragment-only, neutral-loss-only, and combined evidence;
- runtime comparison against the current compact spectral search.

### 4. Formula-conditioned retrieval

Suggested branch: `experiment/formula-conditioned-retrieval`

Add an optional offline adapter for precomputed SIRIUS, MIST-CF, or another publicly available formula
predictor. First measure formula top-k accuracy, then candidate recall after applying formula evidence.
Do not make the optional dependency necessary for the baseline package.

Deliverables:

- versioned adapter/input schema and clear missing-output behavior;
- formula top-1/top-5 accuracy on fold 1;
- candidate recall and ranking metrics with and without the formula channel;
- licensing and Kaggle offline-packaging notes.

## Engineering tasks that can run in parallel

### 5. Source-only Kaggle bundle updates

Suggested branch: `feature/source-only-bundle`

Add a CLI flag or Make target that refreshes only `dist/casmi26-source` when model assets are
unchanged. The current builder recopies and hashes the roughly 1 GiB EXP027 asset bundle even for a
small source-only change.

### 6. Spectral-gate regression tests

Suggested branch: `test/spectral-gate-evaluation`

Add focused tests for `scripts/evaluate_spectral_gate.py`, including duplicate removal, missing
baseline queries, rank shifts after a wrong gate, threshold boundaries, and malformed input columns.

### 7. Move production thresholds into configuration

Suggested branch: `feature/inference-configuration`

Move the Class-1 gate, analog search, and candidate limits out of hardcoded subprocess arguments and
into a versioned inference YAML file. Preserve the exact EXP029 defaults and add command-construction
tests.

### 8. Artifact manifest and provenance audit

Suggested branch: `chore/artifact-provenance`

Verify that every Kaggle asset has a checksum, source version, license, producing command, expected
filename, and compatibility notes. Add a lightweight manifest validator without committing any
artifacts.

### 9. Contributor-friendly small-data experiment fixture

Suggested branch: `feature/mini-experiment-fixture`

Create a generated or redistributable tiny dataset that exercises candidate generation, fingerprint
ranking, analog scoring, learned ranking, gating, and submission validation. It must not contain
competition data or licensed external structures.

## Useful investigation tasks

- Analyze failures by precursor mass, polarity, adduct, spectrum count, collision energy, and
  candidate-source coverage on fold 1. Branch: `experiment/fold1-error-analysis`.
- Evaluate whether COCONUT source priors or natural-product-likeness descriptors improve ranking
  without reducing candidate recall. Branch: `experiment/natural-product-prior`.
- Profile the two analog passes and spectral retrieval, which dominate the 450-second local runtime.
  Branch: `perf/retrieval-profile`.
- Audit RDKit 2026.03.3 tautomer/InChIKey14 behavior with additional charged, isotopic, and salt-like
  structures. Branch: `test/chemistry-normalization`.
- Improve contributor documentation for obtaining shared Kaggle artifacts without exposing tokens.
  Branch: `docs/team-artifacts`.

## Work already in progress or completed

- Do not rebuild the EXP027 dual-channel ranker unless an experiment explicitly needs a control.
- Do not retune the EXP029 `0.90` gate using fold 0 or the public leaderboard.
- EXP028 polarity-aware representatives improved standalone retrieval but hurt the replacement and
  three-channel rankers; treat it as a rejected result unless a new hypothesis materially differs.
- The existing source bundle and notebook already satisfy offline execution and submission-format
  requirements.
- Large databases, feature matrices, models, indexes, predictions, wheels, and submissions belong in
  ignored local paths or versioned Kaggle datasets, never Git.

## Experiment handoff checklist

Every research pull request should state:

- hypothesis and branch/task owner;
- data sources, versions, licenses, and checksums;
- exact training, fold-1 selection, and frozen fold-0 commands;
- baseline and new MRR@25, Hits@1/5/10/25, candidate recall, and runtime;
- artifact additions or replacements and their sizes;
- promote or reject decision with a short explanation;
- `make check` result, plus `make offline-inference` when production inference changes.
