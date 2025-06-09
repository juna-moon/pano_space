import json, os, pytest, glob

QC_FILES = glob.glob("data/**/qc_stats.json", recursive=True)

@pytest.mark.parametrize("js", QC_FILES)
def test_qc_json(js):
    stats = json.load(open(js))
    assert stats["n_fail"] == 0
    assert 0.0 < min(stats["mean"]) < 1.0