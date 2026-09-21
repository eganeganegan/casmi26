from pathlib import Path

from scripts.run_offline_inference import build_commands


def test_offline_inference_builds_nine_ordered_steps() -> None:
    commands = build_commands(
        python="python",
        project_root=Path("/project"),
        test=Path("/input/test.parquet"),
        candidate_db=Path("/assets/candidates.parquet"),
        fingerprint_model=Path("/assets/model.pt"),
        fingerprint_index=Path("/assets/fingerprints.joblib"),
        spectral_index=Path("/assets/spectral.joblib"),
        analog_index=None,
        descriptor_cache=Path("/assets/descriptors.parquet"),
        ranker_dir=Path("/assets/ranker"),
        work_dir=Path("/work"),
        output=Path("/work/submission.csv"),
    )
    assert [name for name, _ in commands] == [
        "fingerprint_prediction",
        "candidate_generation",
        "fragment_scoring",
        "ranking_features",
        "analog_scoring",
        "ranker_prediction",
        "spectral_retrieval",
        "spectral_gate",
        "submission",
    ]
    assert commands[-1][1][-2:] == ["--max-candidates", "25"]


def test_offline_inference_adds_dual_analog_steps() -> None:
    commands = build_commands(
        python="python",
        project_root=Path("/project"),
        test=Path("/input/test.parquet"),
        candidate_db=Path("/assets/candidates.parquet"),
        fingerprint_model=Path("/assets/model.pt"),
        fingerprint_index=Path("/assets/fingerprints.joblib"),
        spectral_index=Path("/assets/spectral.joblib"),
        analog_index=Path("/assets/raw_entropy.joblib"),
        descriptor_cache=Path("/assets/descriptors.parquet"),
        ranker_dir=Path("/assets/ranker"),
        work_dir=Path("/work"),
        output=Path("/work/submission.csv"),
    )
    names = [name for name, _ in commands]
    assert names[4:8] == [
        "analog_scoring",
        "raw_analog_scoring",
        "analog_merge",
        "ranker_prediction",
    ]
