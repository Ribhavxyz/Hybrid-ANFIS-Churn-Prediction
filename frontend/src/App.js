import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    Age: 35,
    "Senior Citizen": "No",
    Married: "Yes",
    Dependents: "No",
    "Number of Dependents": 0,
    "Tenure in Months": 12,
    "Internet Service": "Fiber Optic",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Unlimited Data": "Yes",
    "Monthly Charge": 75,
    "Total Charges": 900,
    "Satisfaction Score": 3
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const predictChurn = async () => {
    setLoading(true);
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: formData })
    });
    const data = await response.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Hybrid ANFIS Churn Predictor</h1>

      <div className="grid">
        <label>Age</label>
        <input type="number" name="Age" value={formData.Age} onChange={handleChange} />

        <label>Senior Citizen</label>
        <select name="Senior Citizen" value={formData["Senior Citizen"]} onChange={handleChange}>
          <option>No</option>
          <option>Yes</option>
        </select>

        <label>Married</label>
        <select name="Married" value={formData.Married} onChange={handleChange}>
          <option>Yes</option>
          <option>No</option>
        </select>

        <label>Dependents</label>
        <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
          <option>No</option>
          <option>Yes</option>
        </select>

        <label>Number of Dependents</label>
        <input type="number" name="Number of Dependents" value={formData["Number of Dependents"]} onChange={handleChange} />

        <label>Tenure (Months)</label>
        <input type="number" name="Tenure in Months" value={formData["Tenure in Months"]} onChange={handleChange} />

        <label>Internet Service</label>
        <select name="Internet Service" value={formData["Internet Service"]} onChange={handleChange}>
          <option>Fiber Optic</option>
          <option>DSL</option>
          <option>No</option>
        </select>

        <label>Phone Service</label>
        <select name="Phone Service" value={formData["Phone Service"]} onChange={handleChange}>
          <option>Yes</option>
          <option>No</option>
        </select>

        <label>Multiple Lines</label>
        <select name="Multiple Lines" value={formData["Multiple Lines"]} onChange={handleChange}>
          <option>No</option>
          <option>Yes</option>
        </select>

        <label>Unlimited Data</label>
        <select name="Unlimited Data" value={formData["Unlimited Data"]} onChange={handleChange}>
          <option>Yes</option>
          <option>No</option>
        </select>

        <label>Monthly Charge</label>
        <input type="number" name="Monthly Charge" value={formData["Monthly Charge"]} onChange={handleChange} />

        <label>Total Charges</label>
        <input type="number" name="Total Charges" value={formData["Total Charges"]} onChange={handleChange} />

        <label>Satisfaction Score</label>
        <select name="Satisfaction Score" value={formData["Satisfaction Score"]} onChange={handleChange}>
          <option>1</option>
          <option>2</option>
          <option>3</option>
          <option>4</option>
          <option>5</option>
        </select>
      </div>

      <button onClick={predictChurn} disabled={loading}>
        {loading ? "Predicting..." : "Predict Churn"}
      </button>

      {result && (
        <div className="result">
          <h2>Prediction Result</h2>
          <p>Churn Probability: <b>{result.churn_probability}</b></p>
          <p>Prediction: <b>{result.churn_prediction}</b></p>
        </div>
      )}
    </div>
  );
}

export default App;
