# CASMI 2026 hybrid molecule identification

Research-oriented, leakage-safe tooling for the Kaggle competition [Enveda CASMI 2026 -
Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
The current codebase completes Milestones 1 through 11: schema-aware data ingestion, chemistry and
metric utilities, structure-disjoint validation, configurable spectrum processing, mass-indexed
spectral retrieval, molecule-level evidence aggregation, offline structure databases, mass-shifted
analog search and propagation, spectrum-to-fingerprint prediction, recall reporting, and strict
submission validation, plus leakage-safe candidate features, learned ranking, bounded in-silico
fragmentation, and rank/score ensembles.

The primary strategy is retrieval because an experimental spectrum matched to a reference spectrum is
strong, interpretable evidence. Retrieval alone cannot cover database-known molecules without spectra
or truly novel molecules, so the package is structured for later analog, formula, fingerprint, and
reranking channels. A large end-to-end SMILES generator is deliberately not the baseline.

## Verified competition schema

Competition parquet files remain ignored by git. Inspection of the downloaded files found:

- Train: 2,539,608 rows, 18 columns, 275,810 unique InChIKey14 structures, and no explicit
  `molecule_id` or `spectrum_id`. The loader therefore uses `inchikey14` as the training molecule group
  and a stable parquet row number as the spectrum identifier.
- Downloaded test: 1,213 rows, 12 columns, and 400 explicit molecule IDs.
- Test fields: `molecule_id`, `spectrum_id`, `ms2_mzs`,
  `ms2_normalized_intensities`, `base_peak_intensity`, `adduct`, `ionization_mode`,
  `instrument_type`, `precursor_mz`, `collision_energy_ev`, `collision_energy_orig`, and
  `collision_energy_orig_units`.
- Train fields are `ingest_lib`, `normalized_smiles`, `inchikey`, `inchikey14`, `molecular_formula`,
  `ionization_mode`, `instrument_type`, `adduct`, `adduct_orig`, `precursor_mz`,
  `precursor_error_ppm`, `ms2_mzs`, `ms2_normalized_intensities`, `num_peaks`,
  `base_peak_intensity`, `collision_energy_ev`, `collision_energy_orig`, and
  `collision_energy_orig_units`.
- Peaks are aligned Arrow list arrays. The loader also accepts nested singleton and string-encoded
  arrays rather than assuming one serialization.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Optional ML dependencies are isolated from the baseline:

```bash
.venv/bin/pip install -e '.[ml]'
```

Teammates developing or reviewing the full pipeline should install both extras and run the complete
local gate:

```bash
.venv/bin/pip install -e '.[dev,ml]'
make check
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch, pull-request, experiment, artifact, and security
conventions. GitHub contains source and lightweight metadata only; competition data, external
databases, model weights, indexes, predictions, submissions, and Kaggle upload bundles remain in
their ignored local directories and are shared separately as versioned team datasets.

Put Kaggle files under `data/raw/` (ignored by git):

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

For local development, download them directly with Kaggle's official `kagglehub` client after
accepting the competition rules and configuring Kaggle authentication:

```bash
.venv/bin/python -c "import kagglehub; kagglehub.login()"
make download
```

Alternatively, set `KAGGLE_API_TOKEN` or place the token in `~/.kaggle/access_token`. Never commit a
token to this repository.

The downloader writes the three expected files directly to `data/raw/` and does not overwrite existing
files unless `scripts/download_data.py --force` is used. It is a local-development utility only; the
submission notebook never downloads anything and remains internet-disabled.

External candidate databases and pretrained outputs belong under `data/external/`; generated indexes
belong under `data/cache/`. No code downloads data during inference.

## External candidate database

Download the public candidate sources and build the compact, deduplicated database:

```bash
make download-external
make candidate-db
```

The production build uses COCONUT (`CC BY 4.0`) and PubChemLite (`CC0 1.0`), removes explicit isotope
labels, recomputes affected exact masses and comparison keys, then deduplicates by InChIKey14. The
result is `data/processed/production_candidates.parquet`. Downloading is a local preparation step;
for Kaggle, publish or attach the resulting parquet as a dataset and keep notebook internet disabled.

The downloader also supports the separately packaged ChEBI/LIPID MAPS bundle:

```bash
.venv/bin/python scripts/download_external_data.py --source chebi_lipidmaps
```

That Kaggle bundle is labeled `CC BY-NC-SA 4.0`, so it is excluded from the production target pending
license review for prize-competition use. It provided no coverage gain in the current controlled
1,000-structure comparison.

An optional expansion path streams the official ChEMBL 37 SDF into the same schema:

```bash
make download-chembl
make candidate-db-chembl
make candidate-db-expanded
```

ChEMBL is licensed under `CC BY-SA 3.0`; retain its attribution and ShareAlike terms when publishing
the derived parquet as a Kaggle dataset. The downloader is a local preparation utility, and the
competition notebook must consume the attached parquet without internet access. The resulting
3,408,337-structure expanded database passes the structure-disjoint coverage gate and is selected for
EXP023, then retained by EXP024 through EXP029.

Build its packed Morgan index and descriptor cache locally before packaging:

```bash
make fingerprint-index-expanded
make candidate-descriptors-expanded
```

## Baseline workflow

Inspect before preprocessing:

```bash
.venv/bin/python scripts/inspect_data.py \
  --train data/raw/train.parquet --test data/raw/test.parquet
```

The inspector uses lazy Polars scans and reports row counts, physical types, nulls, unique molecule and
structure counts, spectra per molecule, adduct frequencies, collision-energy/precursor/peak-count
distributions, examples, and duplicate counts.

Clean spectra in bounded-memory Arrow batches and create folds:

```bash
.venv/bin/python scripts/preprocess.py \
  --input data/raw/train.parquet --output data/processed/train.parquet

.venv/bin/python scripts/create_splits.py \
  --input data/processed/train.parquet --output data/processed/folds.parquet \
  --strategy structure --folds 5
```

Run a bounded benchmark fold before a full experiment:

```bash
.venv/bin/python scripts/run_cv.py \
  --train data/processed/train.parquet --splits data/processed/folds.parquet --fold 0 \
  --max-validation-molecules 100 --max-library-spectra 100000 \
  --output experiments/EXP001/predictions.parquet
```

For inference, build and save the spectral library, predict all spectra, then write the submission:

```bash
.venv/bin/python scripts/build_spectral_index.py \
  --input data/raw/train.parquet --output data/cache/pipeline_compact.joblib

.venv/bin/python scripts/predict.py \
  --test data/raw/test.parquet --model data/cache/pipeline_compact.joblib \
  --output predictions.parquet

.venv/bin/python scripts/make_submission.py \
  --predictions predictions.parquet --output submission.csv
```

`scripts/predict.py` loads a complete `CASMIPipeline`; create one in Python with `fit(...)` and
`save(...)`, or load a serialized pipeline produced by an experiment. The standalone index builder is
useful for index inspection and caching.

The production spectral index is array-backed: peaks use CSR-style float32 buffers, repeated structures
and adducts are dictionary-encoded, and identifiers are packed UTF-8. It retains no per-row Python
`Spectrum` objects. On the complete training set, 2,539,608 spectra and 57,932,354 cleaned peaks occupy
about 650 MB of index buffers and serialize to 346 MB. The artifact loads in 1.16 seconds. Retrieval on
the downloaded 400-molecule test takes about 61 seconds on CPU, and keyed submission assembly takes
about 2 seconds. These are engineering timings only; the downloaded test is a public dummy set and is
not a validation estimate.

## Baseline algorithm

Each spectrum is cleaned independently: invalid/negative peaks are removed, peaks above the configured
precursor window and below the relative-intensity floor are dropped, top-N peaks are retained, then a
raw/square-root/log transform and base-peak normalization are applied. Recipes are YAML-controlled.

The library index converts each precursor to neutral monoisotopic mass using its parsed adduct and
keeps a sorted mass index. A query uses binary search for a ppm mass window and can require matching
polarity before peak matching. Candidate evidence includes direct cosine, precursor-shifted modified
cosine, neutral-loss cosine, matched peaks, explained intensity, mass error, adduct match, and collision
energy difference. Peaks are matched through sorted windows with one-to-one greedy assignment-not an
all-pairs peak loop.

All spectra for one `molecule_id` are searched before ranking. Hits are deduplicated by tautomer-
canonical InChIKey14 and aggregated using max, mean, top-k mean, noisy OR, or log-sum-exp. The default
transparent baseline score is 60% direct cosine, 25% modified cosine, and 15% neutral-loss cosine;
no leaderboard-tuned weights are embedded.

## Validation and leakage controls

Random spectrum splits are invalid because replicate spectra of the same molecule would leak directly
into the search library. Structure folds group by competition-normalized InChIKey14. Scaffold folds
group Bemis–Murcko scaffolds, with acyclic molecules kept structure-specific. Explicit assertions fail
if a key appears in multiple folds or in both the validation set and retrieval library.

`scripts/run_cv.py` removes every spectrum in the held-out fold from the library. It also strips SMILES,
formula, exact mass, and InChIKey fields from validation `Spectrum` objects before prediction. Neutral
mass is inferred only from observed precursor m/z and adduct. Ground-truth formulas are never used.

The official metric implementation canonicalizes tautomers with RDKit, reduces to InChIKey14, preserves
raw rank positions, and reports MRR@25, Hits@1/5/10/25, median correct rank, and missing-candidate
fraction. Final submissions separately enforce unique structures. The dependency is pinned to the
competition's RDKit 2026.03.3 build so local and Kaggle normalization do not drift.

## Candidate recall versus ranking

Candidate recall asks whether the true structure exists anywhere in the generated pool. Ranking asks
how highly the model places it once present. A reranker cannot repair missing candidates. Use
`scripts/candidate_recall.py` to compare Recall@25/100/500/1000/5000 for each retrieval channel and an
RRF union before tuning ranking weights.

```bash
.venv/bin/python scripts/candidate_recall.py \
  --truth validation_truth.csv \
  --channel spectral=experiments/EXP001/predictions.parquet \
  --channel analog=experiments/EXP002/predictions.parquet \
  --output experiments/candidate_recall.csv
```

Evaluate the exact-mass database directly from observed precursor/adduct values (never truth formula):

```bash
.venv/bin/python scripts/evaluate_candidate_db.py \
  --train data/raw/train.parquet --splits data/processed/folds.parquet --fold 0 \
  --candidate-db data/processed/production_candidates.parquet \
  --max-validation-molecules 1000 --seed 42 --mass-tolerance-ppm 20 \
  --output experiments/EXP009/predictions.parquet
```

On the deterministic random fold-0 sample, the 776,060-structure production database has 11.6%
ground-truth coverage. At 20 ppm it reaches Recall@25/100/500 of 7.8%/9.7%/10.9%, mass-only MRR@25
of 0.0306, and a median of 83 candidates per query. A 1,000 ppm diagnostic window reaches the complete
11.6% database-coverage ceiling but raises the median pool to 1,371. Accordingly, configuration keeps
20 ppm as the primary channel and exposes 50 and 1,000 ppm membership as fallback ranker features.

`CandidateDatabase.query_mass_channels(...)` performs one widest-window lookup and annotates every
candidate with its narrowest matching tolerance. `AnalogRetriever` separately performs bounded broad
mass-shifted spectral search; `propagate_analog_candidates(...)` uses the retrieved analog structures
to score exact-mass database candidates by Morgan similarity. Keeping these channels distinct prevents
a noisy analog match from being mistaken for an exact molecular match.

## Spectrum-to-fingerprint baseline

Milestone 4 predicts 2,048-bit Morgan probabilities from a 1 Da binned spectrum plus observed neutral
mass, precursor m/z, collision energy, and polarity. PyTorch remains optional so retrieval-only use does
not import it. Install the ML extras, build the packed candidate index, and train with:

```bash
.venv/bin/pip install -e '.[ml]'
make fingerprint-index
make train-fingerprint
```

The default CV protocol trains on folds 2–4, uses fold 1 for early stopping and blend calibration, and
reports fold 0 once. Training samples at most two spectra per structure to prevent large reference
libraries from dominating the loss. The configured 200,000-spectrum CPU run completes in about two
minutes. Candidate fingerprints cover 776,042 of 776,060 production structures, use 199.8 MB in
memory, serialize to 54 MB, and load in 0.22 seconds.

Generate molecule-level probabilities and rank exact-mass candidates with:

```bash
.venv/bin/python scripts/predict_fingerprint.py \
  --input data/raw/test.parquet --model experiments/EXP018/model.pt \
  --output predictions/fingerprint_probabilities.parquet

.venv/bin/python scripts/rank_fingerprint_candidates.py \
  --spectra data/raw/test.parquet \
  --probabilities predictions/fingerprint_probabilities.parquet \
  --candidate-db data/processed/production_candidates.parquet \
  --fingerprint-index data/cache/production_morgan_2048.joblib \
  --output predictions/fingerprint_candidates.parquet
```

On the deterministic 1,000-molecule fold-0 sample, fingerprint-only ranking is weaker than mass order
(MRR@25 0.02485 versus 0.03015). It is therefore not used alone. A 50/50 reciprocal-rank blend selected
on fold 1 reaches fold-0 MRR@25 0.03537, a 17.3% relative improvement over mass-only ranking. Oracle
fingerprints reach 0.09942, showing substantial remaining model headroom. These are local CV results,
not leaderboard measurements.

## Learned candidate reranker

Milestone 5 materializes one row per query/candidate pair and trains a LightGBM LambdaRank model. Its
29 identity-free inputs include signed and absolute mass error, within-query mass and fingerprint
ranks, predicted-fingerprint compatibility, query spectrum metadata, candidate-source flags, and four
basic RDKit descriptors. The label, oracle fingerprint score, molecule ID, InChIKey, SMILES, and raw
formula/source strings are never model inputs. Queries without a positive candidate are retained for
reporting but excluded from LambdaRank fitting because they contain no pairwise learning signal.

Build the reusable descriptor cache and the fold-specific ranking tables, then train:

```bash
make candidate-descriptors
make ranking-features RANK_EXPERIMENT=experiments/EXP020
make train-ranker RANK_EXPERIMENT=experiments/EXP020
```

The feature commands expect candidate rows previously generated by `scripts/evaluate_fingerprint.py`
under `fold1/` (training/early stopping) and `fold0/` (one-time reporting). On deterministic 5,000-query
samples, fold 0 contains the true structure for 655 queries, setting a 13.1% candidate-recall ceiling.
The learned model reaches MRR@25 0.04364, compared with 0.03750 for the preselected 50/50 RRF blend,
0.03519 for mass-only, and 0.03377 for fingerprint-only ranking. That is a 16.4% relative improvement
over RRF and 24.0% over mass-only. The fit uses 469 positive-containing fold-1 queries, validates on
117 disjoint fold-1 queries, and never fits on fold 0. These are local CV results, not leaderboard
measurements.

## Lightweight fragmentation and ensembles

Milestone 6 adds an optional MetFrag-lite channel. For each candidate it breaks at most 32 non-ring
single bonds, caches the resulting exact product and neutral-loss masses, generates simple
polarity-aware product-ion hypotheses, and matches them to at most eight observed spectra. It reports
best and mean peak coverage, explained intensity, matched-fragment count, top-intensity coverage,
mass-weighted coverage, and neutral-loss coverage. This is intentionally a heuristic structural
compatibility feature, not a quantum fragmentation simulator.

The expensive stage is restricted to the union-like top 50 candidates by mass/fingerprint priority.
On the 5,000-query fold samples this scores about 220,000 rows per fold in roughly four minutes on CPU.
It can read query spectra from the compact training index for CV or directly from a raw test parquet
for offline inference. Run the CV workflow with:

```bash
make fragment-features \
  FRAGMENT_SOURCE_EXPERIMENT=experiments/EXP020 \
  FRAGMENT_EXPERIMENT=experiments/EXP021
make ranking-features RANK_EXPERIMENT=experiments/EXP021
make train-ranker RANK_EXPERIMENT=experiments/EXP021
make ensemble RANK_EXPERIMENT=experiments/EXP021
```

On the untouched 5,000-query fold-0 report, fragmentation-only ranking reaches MRR@25 0.04880. The
43-feature LightGBM model reaches 0.06039, Hits@1 0.0446, Hits@25 0.1092, and a median correct rank of
2. This improves MRR@25 by 38.4% over the Milestone 5 ranker and 71.6% over mass-only ranking, while
the candidate-recall ceiling remains 13.1%. Fixed, untuned three-channel RRF, Borda, and z-score blend
ablations score 0.04813, 0.04570, and 0.05951 respectively, so none replaces the learned ranker.

The ensemble utility supports weighted RRF, Borda, and per-query normalized score blending. Blend
weights must be selected on a secondary validation fold; the fold-0 numbers above are reports, not a
license to tune against fold 0.

## Candidate-recall diagnosis

EXP022 separates retrieval failures from database-coverage failures with
`scripts/analyze_candidate_failures.py`. For the 5,000-query fold-0 report, 4,315 truths (86.3%) are
absent from the production candidate database, 30 are present but missed by candidate generation, 108
are retrieved below rank 25, and 547 are hits at 25. Database expansion is therefore higher leverage
than merely widening the mass window.

A controlled 1,000 ppm fallback confirms that diagnosis. Keeping its top 100 candidates raises
Recall@5000 from 0.1310 to 0.1328 but grows the fold-0 pool from 490,865 to 938,440 rows. After cached
fragment scoring and retraining with five fallback-rank/count features, MRR@25 falls from EXP021's
0.06039 to 0.05824. The fallback is rejected; EXP021 remained the selected model until the legitimate
offline database expansion below was evaluated.

The ChEMBL expansion in EXP023 succeeds where the broad mass fallback does not. Fold-0 database
coverage rises from 0.1370 to 0.1940, and actual 20 ppm Recall@5000 rises from 0.1310 to 0.1880. The
49-feature fragment-aware ranker reaches MRR@25 0.07550, Hits@1 0.0520, and Hits@25 0.1424—a 25.0%
MRR improvement over EXP021. EXP023 is therefore the selected validation model. Its final audit still
finds 4,030 of 5,000 truths absent from the expanded database, making additional legitimate structure
coverage the primary research bottleneck.

The reproducible feature/ranker stage is:

```bash
make fragment-features \
  FRAGMENT_SOURCE_EXPERIMENT=experiments/EXP023 \
  FRAGMENT_EXPERIMENT=experiments/EXP023 \
  FRAGMENT_OUTPUT_NAME=candidates_fragment.parquet \
  FRAGMENT_FOLD1_CACHE=experiments/EXP021/fold1/candidates.parquet \
  FRAGMENT_FOLD0_CACHE=experiments/EXP021/fold0/candidates.parquet
make ranking-features \
  RANK_EXPERIMENT=experiments/EXP023 \
  RANK_CANDIDATE_NAME=candidates_fragment.parquet \
  RANK_DESCRIPTOR_CACHE=data/cache/expanded_candidate_descriptors.parquet
make train-ranker RANK_EXPERIMENT=experiments/EXP023
```

## Class-1 spectral gate

EXP024 adds a deliberately conservative direct-library channel inspired by the public Apache-2.0
four-channel CASMI baseline. The compact index covers all 2,539,608 training spectra. Its validation
holds out one spectrum while explicitly excluding that spectrum from retrieval, so another public
reference spectrum must support the prediction. On 500 deterministic multi-spectrum structures it
reaches MRR@25 0.79363, Hits@1 0.728, and Hits@25 0.962. A cosine >= 0.95 gate with at least 70%
explained query intensity and six matched peaks had 100% precision on the original 27 accepted
validation queries. Lower-confidence queries retained the EXP023 ranking unchanged.

On all 250 structures from the instrument- and chemistry-matched `enveda-np-examples` cohort, the
same leave-one-spectrum-out retrieval reaches MRR@25 0.93405, Hits@1 0.892, and Hits@25 1.0. Although
a 0.90 cosine gate is perfect on the 20 matched-cohort queries it accepts. EXP024 initially retained
0.95 because the smaller broad-library audit exposed false positives between 0.90 and 0.95.

Reproduce that audit with `make class1-eval`.

The downloadable example test contains training examples and is therefore only an engineering check,
not a leaderboard estimate. EXP024 gates 395 of its 400 molecules and completes the full local path in
140.6 seconds. Its public leaderboard score is 0.124, up from EXP023's 0.116 (+0.008 absolute, +6.9%
relative), confirming that the conservative direct-library channel adds value without lowering its
validation-selected threshold.

Reference: `denpugovkin/casmi26-v17-adduct-grouped-merged-fpnet`, derived from the public 0.339
`haideptry` engine under Apache-2.0.

## Mass-shifted analog propagation

EXP025 adds a structure-disjoint Class-2 channel. It selects the richest spectrum for each of 275,808
training structures, entropy-weights the retained peaks once, and exhaustively searches a +/-200 Da
neutral-mass window with both direct and precursor-mass-shifted peak matching. The top 100 spectral
analogs are propagated to the exact-mass candidate pool through 2,048-bit Morgan Tanimoto similarity;
the primary evidence is `Tanimoto * spectral_similarity^4`.

Fusion strength was selected only on 500 deterministic fold-1 queries. A 0.1 analog / 0.9 frozen
ranker weighted RRF improves fold-1 MRR@25 from 0.08586 to 0.09246. With that weight frozen, the
untouched 500-query fold-0 report improves from 0.05438 to 0.06046 (+11.2%), while Hits@25 rises from
0.122 to 0.128. Broader analog weights are not used even though some look better on fold 0, because
fold 1 did not select them. Both audits exclude the query structure from the representative library.

Reproduce the selection and report with `make analog-eval`.

EXP025 scored 0.128 on the public leaderboard, up from EXP024's 0.124 and EXP023's 0.116. EXP026
replaces the fixed analog vote with 12 raw and query-relative analog features learned jointly with the
existing mass, fingerprint, descriptor, and fragment evidence. On the untouched 5,000-query fold-0
report, its 61-feature LambdaRank model reaches MRR@25 0.11711, Hits@1 0.0950, and Hits@25 0.1712.
That is a 55.1% MRR improvement over the otherwise identical EXP023 ranker. Analog Tanimoto, analog
score z-normalization, and analog rank percentile are the three largest features by gain. Reproduce
the full feature generation and learned-ranker audit with `make analog-ranker`. EXP026 scored 0.185
on the public leaderboard, a 0.057 absolute (+44.5%) jump over EXP025 and a 0.069 absolute (+59.5%)
gain over EXP023.

EXP027 keeps EXP026's transformed-spectrum evidence and adds an independent raw-intensity entropy
channel. The new standalone index is built directly from `train.parquet` with a 0.2% base-peak
cutoff, linear intensities, a 256-peak cap, and entropy weighting. Its 12 features complement rather
than replace the original 12 analog features. On the same untouched 5,000-query fold-0 report, the
73-feature ranker reaches MRR@25 0.12368, Hits@1 0.1026, and Hits@25 0.1748: a 5.6% MRR gain over
EXP026. Rebuild and validate this model with `make dual-analog-ranker`.

EXP027 scored 0.226 on the public leaderboard, up 0.041 absolute (+22.2%) from EXP026. EXP029 then
revisits the conservative Class-1 gate using every eligible structure in the fold manifests rather
than a 500-query sample. With the gate fixed at cosine >= 0.90, explained intensity >= 0.70, and at
least six matched peaks, fold-1 MRR@25 rises from 0.09061 to 0.20442. On untouched fold 0, the same
gate raises the complete blended MRR@25 from 0.14802 at the former 0.95 threshold to 0.22755, Hits@1
from 0.1284 to 0.2118, and Hits@25 from 0.1948 to 0.2642. Reproduce the selection and frozen report
with `make class1-gate-eval`.

## Offline Kaggle submission

Run the complete EXP029 inference path locally with:

```bash
make offline-inference
```

This executes spectrum-to-fingerprint prediction, exact-mass candidate generation, bounded top-50
fragment scoring, feature construction, dual mass-shifted analog propagation, learned LightGBM ranking,
high-confidence direct spectral gating, and strict submission validation. On the downloaded
400-molecule example test set EXP029 completes locally in 450.25 seconds and writes the required
`submission.csv`. The verified file has all 400 sample-submission IDs and exactly 25 valid,
structure-deduplicated SMILES per row.

Build the two datasets expected by the Kaggle inference notebook with:

```bash
make kaggle-bundles
```

This refreshes `dist/casmi26-source/` and the frozen `dist/casmi26-exp027-assets/`. EXP029 changes
the source-side gate only, so the EXP027 model assets remain valid. The asset bundle contains the
expanded candidate parquet, packed Morgan index, compact spectral index, raw entropy index, descriptor
cache, fingerprint model, ranker, and feature schema, plus a SHA-256/license manifest. Publish those directories as
private Kaggle datasets, attach them and the competition data to
`notebooks/06_kaggle_inference.ipynb`, disable internet, and
run all cells. The source bundle also carries pinned CPython 3.11 and 3.12 manylinux RDKit wheels; the
notebook installs the compatible wheel with `pip --no-index` when the Kaggle image does not already
provide RDKit 2026.3.3. The notebook writes `/kaggle/working/submission.csv`; no internet access,
training, or download occurs in the notebook, and the measured runtime is far below the nine-hour
code-competition limit.

## Testing

```bash
make test
make smoke
```

Tests cover SMILES/tautomer/stereochemistry normalization, InChIKey14, every required hidden-test
adduct, round-trip and ppm mass calculations, spectrum cleaning and similarities, metric behavior,
structure leakage, molecule aggregation, and submission constraints. The smoke test writes a synthetic
parquet, inspects and streams it, retrieves a structure from multiple spectra, scores it, and validates
a submission.

## Configuration and extension points

The YAML files in `configs/` isolate spectrum, retrieval, fingerprint, fragmentation, reranker, and
ensemble choices.
Optional integrations such as FAISS, matchms, DreaMS, SIRIUS/MIST-CF, XGBoost, and PyTorch Geometric
should remain adapters. PyTorch and LightGBM are installed only through the `ml` extra and are not
imported by retrieval-only workflows.

Offline Kaggle packaging is complete. The next research milestone should improve the candidate-recall
ceiling; an optional learned structure encoder remains lower priority unless it improves retrieval
coverage. External
candidate databases should be normalized into canonical SMILES/InChIKey14/exact-mass parquet files and
attached as Kaggle datasets. External models should ship weights plus their exact preprocessing config;
the inference notebook must never contact the network.

See `CHANGELOG.md` for implemented boundaries and current limitations.

## License

The source code in this repository is available under the [MIT License](LICENSE). Competition data,
external databases, pretrained artifacts, and other separately distributed assets retain their own
licenses and terms; the MIT License does not override them.
