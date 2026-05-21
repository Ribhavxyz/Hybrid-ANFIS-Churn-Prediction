"""End-to-end tests for the ANFIS churn prediction API."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _base_payload() -> dict:
    return {
        "Age": 35,
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "Number of Dependents": 0,
        "Tenure in Months": 12,
        "Internet Service": "Fiber Optic",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Unlimited Data": "Yes",
        "Monthly Charge": 75,
        "Total Charges": 900,
        "Satisfaction Score": 3,
    }


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ANFIS API running"}


def test_predict_valid_payload():
    response = client.post("/predict", json={"data": _base_payload()})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}


def test_internet_service_three_values_give_three_probabilities():
    """Regression test for the historical 'all three options predict identically' bug."""
    probabilities = []
    for value in ("Fiber Optic", "DSL", "No"):
        payload = {**_base_payload(), "Internet Service": value}
        response = client.post("/predict", json={"data": payload})
        assert response.status_code == 200
        probabilities.append(response.json()["churn_probability"])

    assert len(set(probabilities)) == 3, (
        f"Expected three distinct probabilities, got {probabilities}"
    )


def test_predict_missing_field_returns_422():
    response = client.post("/predict", json={"data": {"Age": 35}})
    assert response.status_code == 422


def test_predict_age_string_equals_int():
    """Verify form-submitted string numerics produce the same probability as ints."""
    int_payload = _base_payload()
    str_payload = {**int_payload, "Age": "35", "Monthly Charge": "75", "Total Charges": "900"}

    r_int = client.post("/predict", json={"data": int_payload})
    r_str = client.post("/predict", json={"data": str_payload})
    assert r_int.status_code == 200
    assert r_str.status_code == 200
    assert r_int.json()["churn_probability"] == r_str.json()["churn_probability"]
