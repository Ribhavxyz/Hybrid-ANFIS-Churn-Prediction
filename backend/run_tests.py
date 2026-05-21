"""Ad-hoc test battery against the live backend (port 8000)."""
import httpx

URL = "http://127.0.0.1:8000/predict"


def call(label: str, payload: dict, expect: int = 200) -> dict | None:
    r = httpx.post(URL, json={"data": payload}, timeout=10)
    status = "PASS" if r.status_code == expect else "FAIL"
    print(f"[{status}] {label:<55s} HTTP {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        print(f"        prob={body['churn_probability']:.4f}  pred={body['churn_prediction']}")
        return body
    try:
        err = r.json()
        if isinstance(err.get("detail"), list):
            for e in err["detail"]:
                loc = ".".join(str(x) for x in e.get("loc", []))
                print(f"        422 {loc}: {e.get('msg')}")
        else:
            print(f"        error: {err}")
    except Exception:
        print(f"        error: {r.text[:200]}")
    return None


print("=" * 72)
print("VALID PROFILES")
print("=" * 72)

call("Low-risk: long tenure, high satisfaction, DSL", {
    "Age": 55, "Senior Citizen": "No", "Married": "Yes", "Dependents": "Yes",
    "Number of Dependents": 2, "Tenure in Months": 60,
    "Internet Service": "DSL", "Phone Service": "Yes", "Multiple Lines": "Yes",
    "Unlimited Data": "No", "Monthly Charge": 55, "Total Charges": 3300,
    "Satisfaction Score": 5,
})

call("High-risk: short tenure, low satisfaction, Fiber", {
    "Age": 28, "Senior Citizen": "No", "Married": "No", "Dependents": "No",
    "Number of Dependents": 0, "Tenure in Months": 2,
    "Internet Service": "Fiber Optic", "Phone Service": "Yes", "Multiple Lines": "No",
    "Unlimited Data": "Yes", "Monthly Charge": 95, "Total Charges": 190,
    "Satisfaction Score": 1,
})

call("Senior citizen, no internet, phone-only", {
    "Age": 72, "Senior Citizen": "Yes", "Married": "Yes", "Dependents": "No",
    "Number of Dependents": 0, "Tenure in Months": 36,
    "Internet Service": "No", "Phone Service": "Yes", "Multiple Lines": "No",
    "Unlimited Data": "No", "Monthly Charge": 25, "Total Charges": 900,
    "Satisfaction Score": 4,
})

call("Family plan: multi-line + dependents, Fiber", {
    "Age": 42, "Senior Citizen": "No", "Married": "Yes", "Dependents": "Yes",
    "Number of Dependents": 3, "Tenure in Months": 48,
    "Internet Service": "Fiber Optic", "Phone Service": "Yes", "Multiple Lines": "Yes",
    "Unlimited Data": "Yes", "Monthly Charge": 120, "Total Charges": 5760,
    "Satisfaction Score": 4,
})

call("New customer, mid-satisfaction, DSL", {
    "Age": 33, "Senior Citizen": "No", "Married": "No", "Dependents": "No",
    "Number of Dependents": 0, "Tenure in Months": 6,
    "Internet Service": "DSL", "Phone Service": "Yes", "Multiple Lines": "No",
    "Unlimited Data": "No", "Monthly Charge": 45, "Total Charges": 270,
    "Satisfaction Score": 3,
})

call("Edge: minimal valid values", {
    "Age": 0, "Senior Citizen": "No", "Married": "No", "Dependents": "No",
    "Number of Dependents": 0, "Tenure in Months": 0,
    "Internet Service": "No", "Phone Service": "No", "Multiple Lines": "No",
    "Unlimited Data": "No", "Monthly Charge": 0, "Total Charges": 0,
    "Satisfaction Score": 1,
})

call("Edge: maximal valid values", {
    "Age": 120, "Senior Citizen": "Yes", "Married": "Yes", "Dependents": "Yes",
    "Number of Dependents": 20, "Tenure in Months": 120,
    "Internet Service": "Fiber Optic", "Phone Service": "Yes", "Multiple Lines": "Yes",
    "Unlimited Data": "Yes", "Monthly Charge": 1000, "Total Charges": 100000,
    "Satisfaction Score": 5,
})

print()
print("=" * 72)
print("REGRESSION: 3 Internet Service values, identical other fields")
print("=" * 72)

base = {
    "Age": 35, "Senior Citizen": "No", "Married": "Yes", "Dependents": "No",
    "Number of Dependents": 0, "Tenure in Months": 12,
    "Phone Service": "Yes", "Multiple Lines": "No", "Unlimited Data": "Yes",
    "Monthly Charge": 75, "Total Charges": 900, "Satisfaction Score": 3,
}
probs = []
for v in ["Fiber Optic", "DSL", "No"]:
    r = httpx.post(URL, json={"data": {**base, "Internet Service": v}}, timeout=10)
    p = r.json()["churn_probability"]
    probs.append(p)
    print(f"   Internet Service={v:<12s} -> churn_probability={p}")
assert len(set(probs)) == 3, probs
print(f"[PASS] all 3 distinct: {probs}")
print()

print("=" * 72)
print("STRING vs INT COERCION")
print("=" * 72)
int_p = httpx.post(URL, json={"data": {**base, "Internet Service": "Fiber Optic"}}).json()["churn_probability"]
str_p = httpx.post(URL, json={"data": {**base, "Internet Service": "Fiber Optic",
                                       "Age": "35", "Monthly Charge": "75", "Total Charges": "900"}}).json()["churn_probability"]
print(f"   int payload  -> {int_p}")
print(f"   str payload  -> {str_p}")
print(f"[{'PASS' if int_p == str_p else 'FAIL'}] equal: {int_p == str_p}")
print()

print("=" * 72)
print("VALIDATION (expect 422)")
print("=" * 72)

call("Missing required fields", {"Age": 35}, expect=422)
call("Age out of range (200)",
     {**base, "Internet Service": "DSL", "Age": 200}, expect=422)
call("Satisfaction Score out of range (9)",
     {**base, "Internet Service": "DSL", "Satisfaction Score": 9}, expect=422)
call("Negative Monthly Charge",
     {**base, "Internet Service": "DSL", "Monthly Charge": -10}, expect=422)
call("Invalid Internet Service (Cable)",
     {**base, "Internet Service": "Cable"}, expect=422)
call("Invalid Yes/No enum (Maybe for Married)",
     {**base, "Internet Service": "DSL", "Married": "Maybe"}, expect=422)
