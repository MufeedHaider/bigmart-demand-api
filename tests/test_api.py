import pytest
from conftest import GOOD_ROW


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_ordered_interval(client):
    p = client.post("/predict", json=GOOD_ROW).json()
    assert 0 <= p["lower"] <= p["prediction"] <= p["upper"]
    assert p["interval_level"] == 0.8 and p["outlet_known"] is True


def test_messy_input_is_accepted_and_flagged(client):
    row = dict(GOOD_ROW, item_fat_content="LF", item_weight=None, item_visibility=0.0)
    p = client.post("/predict", json=row).json()
    assert any("imputed" in w for w in p["warnings"])


def test_unknown_outlet_is_flagged_as_less_reliable(client):
    p = client.post("/predict", json=dict(GOOD_ROW, outlet_identifier=None)).json()
    assert p["outlet_known"] is False and any("New outlet" in w for w in p["warnings"])


def test_price_outside_training_range_warns(client):
    p = client.post("/predict", json=dict(GOOD_ROW, item_mrp=400)).json()
    assert any("extrapolation" in w for w in p["warnings"])


@pytest.mark.parametrize("patch", [{"item_mrp": -1}, {"item_mrp": 0}, {"item_fat_content": "Spicy"}, {"item_identifier": "XX123"},
                                   {"outlet_type": "Mega Store"}, {"outlet_establishment_year": 2030}, {"item_visibility": 3}])
def test_invalid_payloads_are_rejected(client, patch):
    assert client.post("/predict", json=dict(GOOD_ROW, **patch)).status_code == 422


def test_batch_matches_single_and_enforces_limit(client):
    single = client.post("/predict", json=GOOD_ROW).json()["prediction"]
    batch = client.post("/predict/batch", json={"rows": [GOOD_ROW, GOOD_ROW]}).json()
    assert [p["prediction"] for p in batch["predictions"]] == [single, single]
    assert client.post("/predict/batch", json={"rows": [GOOD_ROW] * 501}).status_code == 422
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422


def test_explain_returns_top_factors(client):
    e = client.post("/explain", json=GOOD_ROW).json()
    assert len(e["top_factors"]) == 5 and all(f["factor"] > 0 for f in e["top_factors"])


def test_every_prediction_is_audit_logged(client):
    client.post("/predict", json=GOOD_ROW); client.post("/predict/batch", json={"rows": [GOOD_ROW] * 3})
    assert client.get("/stats").json()["predictions_served"] == 4
