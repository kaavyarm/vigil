import { lazy, Suspense, useEffect, useState } from "react";
import {
  fetchPatients,
  predictCustomPatient,
} from "./api";
import ExplanationPanel from "./ExplanationPanel";

const PatientDashboard = lazy(() => import("./PatientDashboard"));

// ── Constants ────────────────────────────────────────────────

const FORM_FIELDS = [
  { key: "Age",        label: "Age",                   unit: "yrs",    min: 18,  max: 100, step: 1   },
  { key: "HR",         label: "Heart Rate",             unit: "bpm",    min: 20,  max: 250, step: 1   },
  { key: "MAP",        label: "Mean Art. Pressure",     unit: "mmHg",   min: 20,  max: 200, step: 1   },
  { key: "GCS",        label: "Consciousness (GCS)",    unit: "/15",    min: 3,   max: 15,  step: 1   },
  { key: "RespRate",   label: "Respiratory Rate",       unit: "br/min", min: 4,   max: 60,  step: 1   },
  { key: "Temp",       label: "Temperature",            unit: "°C",     min: 30,  max: 42,  step: 0.1 },
  { key: "SaO2",       label: "O₂ Saturation",          unit: "%",      min: 50,  max: 100, step: 1   },
  { key: "Creatinine", label: "Creatinine",             unit: "mg/dL",  min: 0.1, max: 20,  step: 0.1 },
  { key: "Lactate",    label: "Lactate",                unit: "mmol/L", min: 0.1, max: 20,  step: 0.1 },
  { key: "BUN",        label: "BUN",                    unit: "mg/dL",  min: 1,   max: 200, step: 1   },
  { key: "Urine",      label: "Urine Output (1h)",      unit: "mL",     min: 0,   max: 500, step: 5   },
];

const PRESETS = {
  stable: {
    Age: 58, HR: 76, MAP: 88, GCS: 15, RespRate: 14, Temp: 37.1,
    SaO2: 98, Creatinine: 0.9, Lactate: 1.1, BUN: 16, Urine: 65, MechVent: 0,
  },
  critical: {
    Age: 74, HR: 128, MAP: 52, GCS: 6, RespRate: 30, Temp: 38.9,
    SaO2: 83, Creatinine: 3.8, Lactate: 5.5, BUN: 48, Urine: 8, MechVent: 1,
  },
};

// ── Shared components ─────────────────────────────────────────

function Header() {
  return (
    <div className="header">
      <h1 className="vigil-title">Vigil</h1>
      <p className="subtitle">ICU Mortality Risk Monitoring Dashboard</p>
    </div>
  );
}

// ── Patient list ──────────────────────────────────────────────

function PatientList({ onSelect }) {
  const [patients, setPatients] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        setPatients(await fetchPatients());
      } catch {
        setError("Failed to load patients. Is the backend running?");
      }
    }
    load();
  }, []);

  if (error) return <p className="load-error">{error}</p>;

  return (
    <div className="patient-grid">
      {patients.map((patient) => (
        <button
          className="patient-card"
          key={patient.record_id}
          onClick={() => onSelect(patient.record_id)}
        >
          <div>
            <h2>Patient {patient.record_id}</h2>
            <p>Age {patient.age} · {patient.icu_type_label}</p>
          </div>
          <div className="risk-block">
            <span className={`badge ${patient.status.toLowerCase()}`}>
              {patient.status}
            </span>
            <strong className={patient.status.toLowerCase()}>
              {patient.mortality_risk_percent}%
            </strong>
          </div>
        </button>
      ))}
    </div>
  );
}

// ── Custom prediction tab ─────────────────────────────────────

function PredictTab() {
  const [vitals, setVitals] = useState(PRESETS.stable);
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  function loadPreset(name) {
    setVitals(PRESETS[name]);
    setResult(null);
    setError(null);
  }

  function setField(key, value) {
    setVitals((v) => ({ ...v, [key]: value }));
  }

  async function handlePredict() {
    setPredicting(true);
    setError(null);
    try {
      setResult(await predictCustomPatient(vitals));
    } catch {
      setError("Prediction failed. Is the backend running?");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="predict-tab">
      <div className="panel">
        <div className="predict-header">
          <h2>Custom Patient</h2>
          <p className="predict-disclaimer">
            Demo only — model trained on 2012 PhysioNet data, not validated for clinical use.
          </p>
        </div>

        <div className="preset-row">
          <span className="preset-label">Load preset:</span>
          <button className="preset-btn" onClick={() => loadPreset("stable")}>Stable patient</button>
          <button className="preset-btn" onClick={() => loadPreset("critical")}>Critical patient</button>
        </div>

        <div className="form-grid">
          {FORM_FIELDS.map((field) => (
            <div className="form-field" key={field.key}>
              <label>{field.label}</label>
              <div className="input-row">
                <input
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={Number.isFinite(vitals[field.key]) ? vitals[field.key] : ""}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setField(field.key, Number.isFinite(val) ? val : null);
                  }}
                />
                <span className="field-unit">{field.unit}</span>
              </div>
            </div>
          ))}

          <div className="form-field">
            <label>Mechanical Ventilation</label>
            <div className="input-row">
              <input
                type="checkbox"
                className="vent-checkbox"
                checked={vitals.MechVent === 1}
                onChange={(e) => setField("MechVent", e.target.checked ? 1 : 0)}
              />
              <span className="field-unit">{vitals.MechVent === 1 ? "On" : "Off"}</span>
            </div>
          </div>
        </div>

        <button className="predict-btn" onClick={handlePredict} disabled={predicting}>
          {predicting ? "Predicting…" : "Predict Risk"}
        </button>
      </div>

      {error && <p className="load-error">{error}</p>}

      {result && (
        <div className="predict-results">
          <div className="dashboard-grid">
            <div className="panel risk-summary">
              <h2>Prediction Result</h2>
              <div className={`big-risk ${result.risk_level.toLowerCase()}`}>
                {result.mortality_risk_percent}%
              </div>
              <div className="risk-meta">
                <span className={`badge ${result.risk_level.toLowerCase()}`}>
                  {result.risk_level}
                </span>
                <span className="confidence-label">
                  Confidence: {result.confidence.label}
                </span>
              </div>
              <p className="recommendation">{result.recommended_action}</p>
            </div>
            <ExplanationPanel title="Top Risk Factors" items={result.risk_factors} type="risk" />
          </div>
          <ExplanationPanel title="Protective Factors" items={result.protective_factors} type="protective" />
        </div>
      )}
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("list");
  const [selectedPatient, setSelectedPatient] = useState(null);

  if (selectedPatient) {
    return (
      <Suspense fallback={<div className="app loading">Loading…</div>}>
        <PatientDashboard
          recordId={selectedPatient}
          onBack={() => setSelectedPatient(null)}
        />
      </Suspense>
    );
  }

  return (
    <div className="app">
      <Header />

      <nav className="tabs">
        <button
          className={`tab-btn${tab === "list" ? " active" : ""}`}
          onClick={() => setTab("list")}
        >
          Patient List
        </button>
        <button
          className={`tab-btn${tab === "predict" ? " active" : ""}`}
          onClick={() => setTab("predict")}
        >
          Try a Patient
        </button>
      </nav>

      {tab === "list"
        ? <PatientList onSelect={setSelectedPatient} />
        : <PredictTab />
      }
    </div>
  );
}
