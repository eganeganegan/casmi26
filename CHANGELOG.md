# Changelog

## Milestone 12 - 2026-09-21

- recorded EXP027's 0.226 public leaderboard score, up 0.041 absolute (+22.2%) from
  EXP026's 0.185 and 0.110 absolute (+94.8%) from EXP023's 0.116;
- evaluated an EXP028 polarity-aware entropy index with 371,919 representatives for 275,808
  structures; standalone fold-1 MRR@25 improved from 0.57521 to 0.58147, but replacement and
  third-channel rankers fell below the frozen EXP027 internal-validation MRR, so it was rejected;
- added exact split-manifest support to the leave-one-spectrum-out Class-1 audit and a reproducible
  spectral-gate evaluator over frozen candidate rankings;
- selected a 0.90 direct-library cosine gate on fold 1, where it raises full-split MRR@25 from
  0.09061 to 0.20442 while retaining 83.4% precision across 843 gated queries;
- with that threshold frozen, improved untouched fold-0 blended MRR@25 from 0.14802 at the old 0.95
  gate to 0.22755, Hits@1 from 0.1284 to 0.2118, and Hits@25 from 0.1948 to 0.2642;
- promoted the gate change as EXP029, verified the full 400-molecule offline path in 450.25 seconds,
  retained the frozen EXP027 model assets, and bumped the project to 0.11.0.

## Milestone 11 - 2026-09-21

- prepared the repository for team collaboration with credential and large-artifact exclusions,
  GitHub Actions CI, issue and pull-request templates, security guidance, contribution standards,
  and a verified Ruff/test/smoke quality gate;
- licensed the repository source code under the MIT License while preserving the separate terms of
  competition data, external databases, and pretrained artifacts;
- recorded EXP026's 0.185 public leaderboard score, a 44.5% relative improvement over EXP025;
- built a standalone entropy index directly from raw training intensities using a 0.2% base-peak
  cutoff, linear intensity scaling, entropy weighting, and one richest spectrum per structure;
- retained both the proven transformed-spectrum analog channel and the complementary raw-spectrum
  channel, exposing 12 independently normalized features from each to a 73-feature LambdaRank model;
- improved untouched 5,000-query fold-0 MRR@25 from 0.11711 to 0.12368 (+5.6%), Hits@1 from
  0.0950 to 0.1026, and Hits@25 from 0.1712 to 0.1748;
- promoted the dual-channel model to EXP027, validated the complete 400-query offline path in
  498.6 seconds, and bumped the project to 0.10.0.

## Milestone 10 - 2026-09-21

- EXP026 scored 0.185 on the public leaderboard, up 0.057 absolute (+44.5%) from EXP025's
  0.128 and 0.069 absolute (+59.5%) from EXP023's 0.116;
- converted EXP025's fixed analog vote into 12 candidate- and query-relative analog features and
  trained a leakage-safe 61-feature LambdaRank model as EXP026;
- limited expensive CV feature generation to the 862 fold-1 and 940 fold-0 groups containing a
  correct generated candidate; zero-positive groups remain in the final report but provide no ranker
  training signal;
- on all learnable fold-1 groups, mass-shifted analog fusion improves conditional MRR@25 from 0.52557
  to 0.58824 at the selection-only 0.3 RRF weight;
- on the untouched 5,000-query fold-0 report, the learned ranker improves MRR@25 from EXP023's
  0.07550 to 0.11711 (+55.1%), Hits@1 from 0.0520 to 0.0950, and Hits@25 from 0.1424 to 0.1712;
- analog Tanimoto, analog score z-normalization, and analog rank percentile are the three largest
  model features by gain;
- reordered offline inference so analog features feed the learned ranker directly, retained the
  high-confidence Class-1 gate, and verified the nine-stage 400-molecule path in 132.7 seconds;
- added reproducible analog-feature merge/training targets, prepared EXP026 Kaggle packaging, and
  bumped the project to 0.9.0.

## Milestone 9 - 2026-09-20

- recorded the first two Kaggle results: EXP023 scored 0.116 and EXP024 scored 0.124 on the public
  leaderboard, a gain of 0.008 absolute and 6.9% relative;
- EXP025 scored 0.128 on the public leaderboard, adding 0.004 absolute over EXP024 and reaching a
  10.3% relative improvement over EXP023;
- added an exhaustive +/-200 Da mass-shifted spectral-entropy search over one richest representative
  spectrum for each of 275,808 training structures, with strict same-structure exclusion in CV;
- propagated the top 100 spectral analogs to exact-mass candidates using 2,048-bit Morgan Tanimoto
  similarity and spectral-similarity-to-the-fourth-power evidence;
- selected a conservative 0.1 analog / 0.9 frozen-ranker weighted RRF on 500 fold-1 queries, improving
  MRR@25 from 0.08586 to 0.09246;
- on 500 untouched fold-0 queries, the frozen fusion improves MRR@25 from 0.05438 to 0.06046 (+11.2%)
  and Hits@25 from 0.122 to 0.128;
- integrated analog scoring ahead of the EXP024 high-confidence Class-1 gate and verified the complete
  nine-stage 400-molecule offline pipeline in 166.0 seconds;
- added Numba as a runtime dependency, extended the Kaggle preflight, added reproducible EXP025
  evaluation and bundle targets, and bumped the project to 0.8.0.

## Milestone 8 - 2026-09-20

- added an EXP024 high-confidence Class-1 spectral gate backed by the compact 2.54M-spectrum
  training-library index;
- added leave-one-spectrum-out evaluation that excludes the query spectrum from retrieval, preventing
  self-match leakage while retaining other public reference spectra for the same structure;
- on 500 deterministic multi-spectrum structures, exact-library retrieval reaches MRR@25 0.79363,
  Hits@1 0.728, and Hits@25 0.962;
- on all 250 `enveda-np-examples` structures, the same self-excluded retrieval reaches MRR@25
  0.93405, Hits@1 0.892, and Hits@25 1.0;
- selected a conservative gate requiring cosine >= 0.95, explained query intensity >= 0.70, and at
  least six matched peaks; it covers 5.4% of the validation sample with 100% top-1 precision;
- integrated spectral retrieval and gating into the offline orchestrator, increasing the verified
  400-molecule example-test runtime from 58.5 to 140.6 seconds, still far below the nine-hour limit;
- added the 346 MB compact spectral index to the EXP024 asset bundle and bumped the project to 0.7.0.

## Offline Kaggle packaging - 2026-09-20

- added enriched test-time candidate generation, frozen LightGBM inference, and a six-stage offline
  orchestrator from `test.parquet` to the required `submission.csv`;
- added an internet-free Kaggle notebook with explicit competition/source/asset dataset mounts and
  dependency/input preflight checks;
- made Kaggle input discovery independent of mount-folder names and bundled pinned manylinux RDKit
  wheels for automatic offline installation on CPython 3.11 or 3.12;
- added reproducible source and EXP023 asset bundles with SHA-256 manifests and candidate-source
  license metadata;
- verified the complete local inference path in 58.5 seconds for all 400 downloaded test molecules;
- validated a 400-row `submission.csv` with exact sample-submission IDs, no nulls or duplicates, and
  exactly 25 valid, structure-deduplicated SMILES per molecule.

## Milestone 7 - 2026-09-20

Implemented:

- official ChEMBL 37 acquisition with explicit `CC BY-SA 3.0` provenance and an offline-only Kaggle
  packaging boundary;
- bounded-memory SDF/SDF.GZ ingestion, InChIKey14 deduplication, isotope normalization, and atomic
  parquet writes;
- preallocated packed Morgan-index construction, parallel batched descriptor generation, streamed
  descriptor joins, and chunked multi-million-row candidate output;
- a ChEMBL source indicator and deterministic saved-rank tie-breaking consistent with reported
  metrics;
- reusable Make targets for the expanded database, fingerprint index, descriptor cache, fragment
  cache reuse, and fold-specific ranking inputs.

Measured on deterministic 5,000-query fold samples:

- 2,897,819 ChEMBL records yield 2,756,566 unique valid structures; merging with the production
  sources yields 3,408,337 isotope-normalized candidates;
- fold-0 database coverage rises from 0.1370 to 0.1940 (+285 present truths), and 20 ppm
  Recall@5000 rises from 0.1310 to 0.1880;
- the expanded fold-0 pool contains 2,907,441 rows with a median 573 candidates per query, while
  bounded fragment scoring still selects only 241,476 rows;
- the 49-feature EXP023 ranker reaches MRR@25 0.07550, Hits@1 0.0520, Hits@25 0.1424, and median
  correct rank 2, improving MRR@25 by 25.0% over EXP021;
- the final audit contains 4,030 absent-from-database, 30 missed-generation, 228 retrieved-below-25,
  and 712 hit-at-25 queries.

EXP023 replaces EXP021 as the selected validation model. The remaining dominant limitation is still
candidate-database coverage (80.6% of fold-0 truths absent).

## Candidate-recall experiment EXP022 - 2026-09-20

Implemented:

- exact failure categorization into absent-from-database, missed-by-generation, retrieved-below-25,
  and hit-at-25 cases;
- optional 1,000 ppm mass/fingerprint fallback generation, deterministic top-N pruning, and five
  fallback-aware rank/count features;
- fragment-feature cache reuse so widened pools score only newly selected query/candidate pairs;
- streaming SDF/SDF.GZ candidate ingestion and an official ChEMBL 37 download/build workflow.

Measured on deterministic 5,000-query fold samples:

- the production database is the dominant limitation: 4,315 of 5,000 fold-0 truths (86.3%) are
  absent, while only 30 database-present truths are missed by the 20 ppm generator;
- a top-100 fallback increases fold-0 Recall@5000 from 0.1310 to 0.1328 (+9 truths), but expands the
  pool from 490,865 to 938,440 rows;
- the resulting 48-feature ranker reaches MRR@25 0.05824, below EXP021's 0.06039, so the wide fallback
  is rejected and EXP021 remains selected;
- ChEMBL expansion is gated on a direct structure-coverage measurement before any expensive index or
  ranker rebuild.

## Milestone 6 - 2026-09-20

Implemented:

- cached single-bond product and neutral-loss mass enumeration with strict bond/spectrum limits;
- polarity-aware product-ion matching and six best/mean multi-spectrum evidence families;
- top-50 mass/fingerprint-priority scoring with explicit scored/unscored indicators;
- CV scoring from the compact spectral cache and deployment scoring from raw test parquet;
- 43-feature fragment-aware LambdaRank training and a standalone fragmentation ablation;
- deterministic weighted RRF, Borda, and z-score/min-max/raw score-fusion utilities;
- CLI and Make workflows for fragment enrichment and ensemble evaluation.

Measured on deterministic 5,000-query fold samples:

- 220,326 fold-1 and 220,450 fold-0 candidate rows scored in 242 and 247 seconds on CPU;
- no candidate failed fragmentation, and the bounded selection includes 573 of 655 fold-0 truths;
- fragmentation-only MRR@25: 0.04880;
- fragment-aware LightGBM MRR@25: 0.06039, Hits@1: 0.0446, Hits@25: 0.1092, median rank: 2;
- improvement over Milestone 5 LightGBM: +38.4%; improvement over mass-only: +71.6%;
- fixed RRF/Borda/z-score ensemble ablations: 0.04813/0.04570/0.05951, so the learned ranker remains
  selected and the ensemble is retained as an ablatable utility;
- candidate Recall@5000 remains 0.131, confirming that candidate generation is now the main ceiling.

Next: improve candidate coverage with additional legitimate offline sources/channels; treat the
Milestone 7 learned structure encoder as optional unless it can also expand retrieval recall.

## Milestone 5 - 2026-09-20

Implemented:

- enriched fold-safe candidate rows with mass, predicted-fingerprint, query-context, and target fields;
- reusable RDKit descriptor generation for all 776,060 candidates (20 MB parquet, 18 invalid rows);
- 29 identity-free numeric ranking features with duplicate/label/cardinality validation;
- deterministic fold-1 fit/validation partitioning and LightGBM LambdaRank training with early stopping;
- fold-0 comparison against mass-only, fingerprint-only, and preselected 50/50 RRF baselines;
- serialized model, prediction rows, feature schema, metrics, and gain/split importance artifacts;
- Make targets for descriptor caching, ranking-table construction, and reranker training.

Measured on deterministic 5,000-query fold samples:

- the fold-0 candidate pool contains 655 truths, for a 13.1% recall ceiling;
- mass-only MRR@25: 0.03519; fingerprint-only: 0.03377; 50/50 RRF: 0.03750;
- LightGBM MRR@25: 0.04364 (+16.4% over RRF and +24.0% over mass-only);
- Hits@1/5/10/25: 0.0268/0.0640/0.0816/0.1032;
- training uses 469 positive-containing fold-1 queries and validates on 117 disjoint fold-1 queries;
  fold 0 is used only for the final report.

Fragmentation-aware ranking is complete; see the Milestone 6 entry above. Candidate recall remains
the limiting factor before complete offline Kaggle packaging.

## Milestone 4 - 2026-09-20

Implemented:

- bit-packed Morgan fingerprint index with 776,042 valid production candidates, 199.8 MB in-memory
  storage, 54 MB serialized size, and 0.22-second load time;
- binned-spectrum plus precursor/adduct/collision-energy/polarity feature encoding;
- optional PyTorch MLP with sparse-bit positive weighting, deterministic structure-balanced sampling,
  early stopping, model/config serialization, and sampled bitwise AUROC reporting;
- strict fold roles: folds 2–4 train, fold 1 selects the checkpoint and fusion weight, fold 0 reports;
- mean and confidence-weighted multi-spectrum probability pooling;
- expected-Tanimoto, binary-Tanimoto, cosine, and BCE candidate compatibility features;
- molecule probability prediction, exact-mass candidate ranking, oracle comparison, and calibrated RRF
  command-line workflows.

Measured on 1,000 deterministic fold-0 molecules:

- mass-only MRR@25: 0.03015;
- predicted-fingerprint-only MRR@25: 0.02485, so this channel is not a standalone replacement;
- 50/50 mass/fingerprint RRF selected on fold 1: MRR@25 0.03537 (+17.3% relative);
- oracle-fingerprint MRR@25: 0.09942, indicating large remaining representation/model headroom;
- the configured 200,000-spectrum CPU training run takes 123 seconds and the 400-molecule fingerprint
  inference plus candidate-ranking path takes under 10 seconds locally.

The first leakage-safe candidate reranker is complete; see the Milestone 5 entry above.

## Scalability hardening - 2026-09-20

- replaced retained Python `Spectrum` objects with packed string dictionaries, encoded metadata,
  CSR-style peak offsets, and contiguous NumPy arrays;
- added batch-local Arrow filtering for bounded validation reads and streamed CV spectra directly into
  the compact index;
- built the complete 2,539,608-spectrum/57,932,354-peak artifact: 649.8 MB of buffers, 346 MB serialized,
  and 1.16-second load time;
- measured 61-second CPU retrieval for the downloaded 400-molecule test;
- reused trusted candidate InChIKey14 values during final deduplication, reducing submission assembly
  from 21.1 seconds to 1.94 seconds while retaining SMILES sanitization;
- verified a 400-row, at-most-25-candidate submission and 56 passing tests.

## Milestone 3 - 2026-09-20

Implemented:

- compact Parquet candidate databases with sorted exact-mass and lazy formula indexes;
- reproducible KaggleHub download and build commands for COCONUT and PubChemLite;
- restricted loading for the NumPy/pickle ChEBI/LIPID MAPS bundle and explicit license isolation;
- source merging, InChIKey14 deduplication, and explicit-isotope normalization with recomputed mass;
- nested 20/50/1,000 ppm candidate channels that preserve fallback membership as ranking features;
- bounded mass-shifted spectral retrieval and Morgan-similarity analog propagation;
- deterministic random validation sampling, stratified recall, and failure-category reports.

Measured on 1,000 random fold-0 structures (seed 42):

- COCONUT + PubChemLite contains 116 truths (11.6% database-coverage ceiling);
- 20 ppm mass filtering gives Recall@25/100/500 = 7.8%/9.7%/10.9%, MRR@25 = 0.0306,
  and median 83 candidates;
- 1,000 ppm recovers all 116 database-present truths but increases the median pool to 1,371;
- isotope normalization repairs one 20 ppm miss;
- adding the noncommercial ChEBI/LIPID MAPS bundle did not improve coverage on this sample and slightly
  worsened mass-only ranking through additional distractors, so it is not in the production build.

Next bottleneck:

- replace the object-heavy full spectral index before million-spectrum CV, then train the fingerprint
  channel and candidate reranker on leakage-safe out-of-fold features.

## Data acquisition - 2026-09-19

- added an official `kagglehub` development dependency and `make download` command;
- added a non-destructive downloader targeting `data/raw/`;
- kept download behavior outside the offline Kaggle inference path.
- adapted ingestion to the verified train schema, which has no explicit molecule or spectrum IDs;
- derived training molecule groups from InChIKey14 and stable spectrum IDs from parquet row numbers;
- allowed multiple stereochemical/tautomer SMILES representations within one competition key;
- verified inspection on 2,539,608 train rows and 1,213 downloaded test rows.
- generated and leakage-checked five balanced structure folds with 55,162 structures per fold.

## Milestone 2 - 2026-09-19

Implemented:

- formal monoisotopic adduct definitions for all ten hidden-test adducts plus common charge states;
- neutral-mass conversion, theoretical precursor m/z, and signed ppm error with round-trip tests;
- sorted exact-mass prefiltering and polarity filtering before spectral scoring;
- direct, precursor-shifted, and neutral-loss similarity evidence;
- molecule-level max/mean/top-k/noisy-OR/log-sum-exp aggregation;
- per-channel Recall@25/100/500/1000/5000 reports and RRF union accounting;
- bounded structure-disjoint CV runner with runtime and peak-memory reporting.

Milestone 3 is complete; see the 2026-09-20 entry above.

## Milestone 1 - 2026-09-19

Implemented:

- Python package, configs, data directories, scripts, and test harness;
- lazy parquet schema/statistics inspection and streaming normalized `Spectrum` records;
- cached RDKit canonicalization, tautomer normalization, InChIKey14, formula, exact mass, and fingerprints;
- configurable peak cleaning, normalization, binning, and efficient sorted peak matching;
- exact competition-style MRR@25 and diagnostic ranking metrics;
- deterministic structure- and scaffold-disjoint folds with leakage assertions;
- spectral library baseline, molecule-level candidate deduplication, and submission writer;
- synthetic end-to-end smoke test.

Validation status at completion: unit tests and synthetic smoke test passed. Real-data measurements
were added in later milestone entries after the competition files became available.
