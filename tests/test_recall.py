from casmi.validation import candidate_recall_report, stratified_candidate_recall


def test_channel_and_union_recall_report() -> None:
    truth = {"a": "CCO", "b": "CCC"}
    channels = {
        "spectral": {"a": ["CCO"], "b": ["CC"]},
        "mass": {"a": ["CC"], "b": ["CCC"]},
    }
    report = candidate_recall_report(truth, channels, (1, 25))
    at_25 = report[report.cutoff == 25].set_index("channel").recall.to_dict()
    assert at_25 == {"spectral": 0.5, "mass": 0.5, "union": 1.0}


def test_stratified_recall() -> None:
    report = stratified_candidate_recall(
        {"a": "CCO", "b": "CCC"},
        {"a": ["CCO"], "b": ["CC"]},
        {"adduct": {"a": "[M+H]+", "b": "[M-H]-"}},
        ks=(25,),
    )
    values = report.set_index("level").recall.to_dict()
    assert values["[M+H]+"] == 1.0
    assert values["[M-H]-"] == 0.0
