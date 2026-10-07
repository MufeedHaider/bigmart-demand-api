"""Guards the committed results file, so the README can never quietly drift from the numbers."""
import json, pathlib, pytest
P = pathlib.Path(__file__).parent.parent / "reports" / "metrics.json"
pytestmark = pytest.mark.skipif(not P.exists(), reason="run bigmart.train first")


def test_locked_test_beats_baseline_and_ci_contains_estimate():
    m = json.loads(P.read_text())["test"]
    assert m["model"]["r2"] > m["baseline_outlet_type_mean"]["r2"] + 0.2
    lo, hi = m["r2_ci95"]; assert lo <= m["model"]["r2"] <= hi


def test_interval_coverage_is_close_to_nominal():
    assert 0.74 <= json.loads(P.read_text())["intervals"]["coverage_test"] <= 0.86


def test_test_items_were_never_in_dev():
    assert json.loads(P.read_text())["split"]["test_items_unseen_in_dev"] is True
