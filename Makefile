PYTHON ?= .venv/bin/python

.PHONY: test smoke check lint inspect download download-external candidate-db spectral-index fingerprint-index \
	train-fingerprint candidate-descriptors fragment-features ranking-features train-ranker ensemble \
	download-chembl candidate-db-chembl candidate-db-expanded fingerprint-index-expanded \
	candidate-descriptors-expanded offline-inference kaggle-bundles class1-eval analog-eval \
	analog-ranker entropy-index dual-analog-ranker class1-gate-eval

ANALOG_FEATURES := analog_score_power4 analog_score_linear analog_tanimoto_max \
	analog_tanimoto_top analog_tanimoto_weighted_mean analog_top_similarity analog_rank \
	reciprocal_analog_rank analog_rank_percentile analog_score_max analog_score_delta_max analog_score_z
ANALOG_FEATURE_ARGS := $(foreach feature,$(ANALOG_FEATURES),--extra-feature $(feature))
RAW_ANALOG_FEATURES := $(addprefix raw_,$(ANALOG_FEATURES))
RAW_ANALOG_FEATURE_ARGS := $(foreach feature,$(RAW_ANALOG_FEATURES),--extra-feature $(feature))

RANK_EXPERIMENT ?= experiments/EXP020
FRAGMENT_SOURCE_EXPERIMENT ?= experiments/EXP020
FRAGMENT_EXPERIMENT ?= experiments/EXP021
FRAGMENT_OUTPUT_NAME ?= candidates.parquet
FRAGMENT_FOLD1_CACHE ?=
FRAGMENT_FOLD0_CACHE ?=
RANK_DESCRIPTOR_CACHE ?= data/cache/production_candidate_descriptors.parquet
RANK_CANDIDATE_NAME ?= candidates.parquet
test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/smoke_test.py

lint:
	$(PYTHON) -m ruff check src scripts tests

check: lint test smoke

inspect:
	$(PYTHON) scripts/inspect_data.py --train data/raw/train.parquet --test data/raw/test.parquet

download:
	$(PYTHON) scripts/download_data.py --output data/raw

download-external:
	$(PYTHON) scripts/download_external_data.py --output data/external

download-chembl:
	$(PYTHON) scripts/download_chembl.py

candidate-db:
	$(PYTHON) scripts/build_candidate_db.py \
		--input data/external/coconut/coconut_csv_lite-09-2026.csv \
		--output data/processed/coconut_candidates.parquet --source coconut \
		--smiles-column canonical_smiles --exact-mass-column exact_molecular_weight \
		--candidate-id-column identifier --inchikey-column standard_inchi_key \
		--formula-column molecular_formula
	$(PYTHON) scripts/build_candidate_db.py \
		--input data/external/pubchemlite/PubChemLite_31Oct2020_exposomics.csv \
		--output data/processed/pubchemlite_candidates.parquet --source pubchemlite \
		--smiles-column SMILES --exact-mass-column MonoisotopicMass \
		--candidate-id-column Identifier --inchikey-column InChIKey \
		--formula-column MolecularFormula
	$(PYTHON) scripts/merge_candidate_dbs.py \
		--input data/processed/coconut_candidates.parquet \
		--input data/processed/pubchemlite_candidates.parquet \
		--output data/processed/production_candidates_raw.parquet
	$(PYTHON) scripts/normalize_candidate_db.py \
		--input data/processed/production_candidates_raw.parquet \
		--output data/processed/production_candidates.parquet

candidate-db-chembl:
	$(PYTHON) scripts/build_sdf_candidate_db.py \
		--input data/external/chembl/chembl_37.sdf.gz \
		--output data/processed/chembl37_candidates.parquet \
		--source chembl37 --id-property chembl_id

candidate-db-expanded:
	$(PYTHON) scripts/merge_candidate_dbs.py \
		--input data/processed/production_candidates.parquet \
		--input data/processed/chembl37_candidates.parquet \
		--output data/processed/expanded_candidates_raw.parquet
	$(PYTHON) scripts/normalize_candidate_db.py \
		--input data/processed/expanded_candidates_raw.parquet \
		--output data/processed/expanded_candidates.parquet

spectral-index:
	$(PYTHON) scripts/build_spectral_index.py \
		--input data/raw/train.parquet --output data/cache/pipeline_compact.joblib

entropy-index:
	$(PYTHON) scripts/build_entropy_index.py \
		--input data/raw/train.parquet \
		--output data/cache/raw_entropy_representatives.joblib

class1-eval:
	$(PYTHON) scripts/evaluate_class1_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--output experiments/EXP024/class1_retrieval.csv --queries 500
	$(PYTHON) scripts/evaluate_class1_retrieval.py \
		--model data/cache/pipeline_compact.joblib --train data/raw/train.parquet \
		--ingest-lib enveda-np-examples \
		--output experiments/EXP024/class1_enveda_np.csv --queries 250

class1-gate-eval:
	$(PYTHON) scripts/evaluate_class1_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--query-ids experiments/EXP023/fold1/query_ids.csv --queries 0 \
		--output experiments/EXP029/fold1/class1_retrieval.csv
	$(PYTHON) scripts/evaluate_spectral_gate.py \
		--ranked experiments/EXP025/fold1_baseline.parquet \
		--spectral experiments/EXP029/fold1/class1_retrieval.csv \
		--query-ids experiments/EXP023/fold1/query_ids.csv \
		--output experiments/EXP029/fold1/spectral_gate.json
	$(PYTHON) scripts/evaluate_class1_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--query-ids experiments/EXP023/fold0/query_ids.csv --queries 0 \
		--output experiments/EXP029/fold0/class1_retrieval.csv
	$(PYTHON) scripts/evaluate_spectral_gate.py \
		--ranked experiments/EXP027/ranker/predictions.parquet \
		--spectral experiments/EXP029/fold0/class1_retrieval.csv \
		--query-ids experiments/EXP023/fold0/query_ids.csv \
		--output experiments/EXP029/fold0/spectral_gate.json

analog-eval:
	$(PYTHON) scripts/predict_ranker.py \
		--features experiments/EXP023/fold1/features.parquet \
		--model experiments/EXP023/ranker/ranker.txt \
		--feature-columns experiments/EXP023/ranker/feature_columns.json \
		--output experiments/EXP025/fold1_baseline.parquet
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP025/fold1_baseline.parquet \
		--query-ids experiments/EXP023/fold1/query_ids.csv \
		--output experiments/EXP025/fold1_analog_500.parquet --queries 500
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP023/ranker/predictions.parquet \
		--query-ids experiments/EXP023/fold0/query_ids.csv \
		--output experiments/EXP025/fold0_analog_500.parquet --queries 500

analog-ranker:
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP025/fold1_baseline.parquet \
		--query-ids experiments/EXP023/fold1/query_ids.csv \
		--output experiments/EXP026/fold1/analog_features.parquet \
		--queries 0 --positive-only --top-analogs 100
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP023/ranker/predictions.parquet \
		--query-ids experiments/EXP023/fold0/query_ids.csv \
		--output experiments/EXP026/fold0/analog_features.parquet \
		--queries 0 --positive-only --top-analogs 100
	$(PYTHON) scripts/merge_analog_features.py \
		--features experiments/EXP023/fold1/features.parquet \
		--analog-features experiments/EXP026/fold1/analog_features.parquet \
		--output experiments/EXP026/fold1/features.parquet
	$(PYTHON) scripts/merge_analog_features.py \
		--features experiments/EXP023/fold0/features.parquet \
		--analog-features experiments/EXP026/fold0/analog_features.parquet \
		--output experiments/EXP026/fold0/features.parquet
	$(PYTHON) scripts/train_ranker.py --config configs/ranker.yaml \
		--train-features experiments/EXP026/fold1/features.parquet \
		--report-features experiments/EXP026/fold0/features.parquet \
		--output experiments/EXP026/ranker $(ANALOG_FEATURE_ARGS)

dual-analog-ranker: entropy-index
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--analog-index data/cache/raw_entropy_representatives.joblib \
		--train-spectra data/raw/train.parquet \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP025/fold1_baseline.parquet \
		--query-ids experiments/EXP023/fold1/query_ids.csv \
		--output experiments/EXP027/fold1/analog_features.parquet \
		--queries 0 --positive-only --top-analogs 100
	$(PYTHON) scripts/evaluate_analog_retrieval.py \
		--model data/cache/pipeline_compact.joblib \
		--analog-index data/cache/raw_entropy_representatives.joblib \
		--train-spectra data/raw/train.parquet \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--predictions experiments/EXP023/ranker/predictions.parquet \
		--query-ids experiments/EXP023/fold0/query_ids.csv \
		--output experiments/EXP027/fold0/analog_features.parquet \
		--queries 0 --positive-only --top-analogs 100
	$(PYTHON) scripts/merge_analog_features.py \
		--features experiments/EXP026/fold1/features.parquet \
		--analog-features experiments/EXP027/fold1/analog_features.parquet \
		--prefix raw_ --output experiments/EXP027/fold1/features.parquet
	$(PYTHON) scripts/merge_analog_features.py \
		--features experiments/EXP026/fold0/features.parquet \
		--analog-features experiments/EXP027/fold0/analog_features.parquet \
		--prefix raw_ --output experiments/EXP027/fold0/features.parquet
	$(PYTHON) scripts/train_ranker.py --config configs/ranker.yaml \
		--train-features experiments/EXP027/fold1/features.parquet \
		--report-features experiments/EXP027/fold0/features.parquet \
		--output experiments/EXP027/ranker $(ANALOG_FEATURE_ARGS) $(RAW_ANALOG_FEATURE_ARGS)

fingerprint-index:
	$(PYTHON) scripts/build_fingerprint_index.py \
		--candidate-db data/processed/production_candidates.parquet \
		--output data/cache/production_morgan_2048.joblib

fingerprint-index-expanded:
	$(PYTHON) scripts/build_fingerprint_index.py \
		--candidate-db data/processed/expanded_candidates.parquet \
		--output data/cache/expanded_morgan_2048.joblib

train-fingerprint:
	$(PYTHON) scripts/train_fingerprint.py --config configs/fingerprint.yaml \
		--spectral-index data/cache/pipeline_compact.joblib \
		--splits data/processed/folds.parquet --output experiments/fingerprint_baseline

candidate-descriptors:
	$(PYTHON) scripts/build_candidate_descriptors.py \
		--candidate-db data/processed/production_candidates.parquet \
		--output data/cache/production_candidate_descriptors.parquet

candidate-descriptors-expanded:
	$(PYTHON) scripts/build_candidate_descriptors.py \
		--candidate-db data/processed/expanded_candidates.parquet \
		--output data/cache/expanded_candidate_descriptors.parquet

fragment-features:
	$(PYTHON) scripts/score_fragment_candidates.py --config configs/fragmentation.yaml \
		--candidates $(FRAGMENT_SOURCE_EXPERIMENT)/fold1/candidates.parquet \
		--spectral-index data/cache/pipeline_compact.joblib \
		$(if $(FRAGMENT_FOLD1_CACHE),--feature-cache $(FRAGMENT_FOLD1_CACHE)) \
		--output $(FRAGMENT_EXPERIMENT)/fold1/$(FRAGMENT_OUTPUT_NAME)
	$(PYTHON) scripts/score_fragment_candidates.py --config configs/fragmentation.yaml \
		--candidates $(FRAGMENT_SOURCE_EXPERIMENT)/fold0/candidates.parquet \
		--spectral-index data/cache/pipeline_compact.joblib \
		$(if $(FRAGMENT_FOLD0_CACHE),--feature-cache $(FRAGMENT_FOLD0_CACHE)) \
		--output $(FRAGMENT_EXPERIMENT)/fold0/$(FRAGMENT_OUTPUT_NAME)

ranking-features:
	$(PYTHON) scripts/build_ranking_dataset.py \
		--candidates $(RANK_EXPERIMENT)/fold1/$(RANK_CANDIDATE_NAME) \
		--descriptor-cache $(RANK_DESCRIPTOR_CACHE) \
		--output $(RANK_EXPERIMENT)/fold1/features.parquet
	$(PYTHON) scripts/build_ranking_dataset.py \
		--candidates $(RANK_EXPERIMENT)/fold0/$(RANK_CANDIDATE_NAME) \
		--descriptor-cache $(RANK_DESCRIPTOR_CACHE) \
		--output $(RANK_EXPERIMENT)/fold0/features.parquet

train-ranker:
	$(PYTHON) scripts/train_ranker.py --config configs/ranker.yaml \
		--train-features $(RANK_EXPERIMENT)/fold1/features.parquet \
		--report-features $(RANK_EXPERIMENT)/fold0/features.parquet \
		--output $(RANK_EXPERIMENT)/ranker

ensemble:
	$(PYTHON) scripts/ensemble_predictions.py --config configs/ensemble.yaml \
		--channel mass=$(RANK_EXPERIMENT)/ranker/predictions.parquet \
		--channel fingerprint=$(RANK_EXPERIMENT)/ranker/predictions.parquet \
		--channel reranker=$(RANK_EXPERIMENT)/ranker/predictions.parquet \
		--rank-column mass=mass_rank --rank-column fingerprint=fingerprint_rank \
		--rank-column reranker=final_rank \
		--truth-manifest $(RANK_EXPERIMENT)/fold0/query_ids.csv \
		--output $(RANK_EXPERIMENT)/ensemble/predictions.parquet

offline-inference:
	$(PYTHON) scripts/run_offline_inference.py \
		--test data/raw/test.parquet \
		--sample-submission data/raw/sample_submission.csv \
		--candidate-db data/processed/expanded_candidates.parquet \
		--fingerprint-model experiments/EXP018/model.pt \
		--fingerprint-index data/cache/expanded_morgan_2048.joblib \
		--spectral-index data/cache/pipeline_compact.joblib \
		--analog-index data/cache/raw_entropy_representatives.joblib \
		--descriptor-cache data/cache/expanded_candidate_descriptors.parquet \
		--ranker-dir experiments/EXP027/ranker \
		--work-dir predictions/exp029 --output submission.csv

kaggle-bundles:
	$(PYTHON) scripts/build_kaggle_bundles.py --force
