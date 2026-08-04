import { useEffect, useState } from "react";
import {
  fetchPatients,
  fetchExplanation,
  fetchRiskTrend,
  fetchTimelineEvents,
  predictCustomPatient,
} from "./api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

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

function ExplanationPanel({ title, items, type }) {
  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="factor-list">
        {items.map((item, index) => (
          <div className="factor" key={index}>
            <div className="factor-info">
              <p className="factor-label">{item.label}</p>
              <p className="factor-value">{item.value_label}</p>
            </div>
            <div className="factor-right">
              <span className={`factor-pct ${type === "risk" ? "risk-pct" : "protective-pct"}`}>
                {type === "risk" ? "+" : "−"}{item.contribution_pct}%
              </span>
            </div>
          </div>
        ))}
      </div>
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

// ── Patient detail dashboard ──────────────────────────────────

function PatientDashboard({ recordId, onBack }) {
  const [explanation, setExplanation] = useState(null);
  const [riskTrend, setRiskTrend] = useState(null);
  const [timelineEvents, setTimelineEvents] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [explanationData, trendData, eventsData] = await Promise.all([
          fetchExplanation(recordId),
          fetchRiskTrend(recordId),
          fetchTimelineEvents(recordId),
        ]);
        setExplanation(explanationData);
        setRiskTrend(trendData.risk_trend);
        setTimelineEvents(eventsData.timeline_events);
      } catch {
        setError("Failed to load patient data. Is the backend running?");
      }
    }
    load();
  }, [recordId]);

  if (error) {
    return (
      <div className="app loading">
        <p>{error}</p>
        <button className="back-button" onClick={onBack} style={{ marginTop: 16 }}>
          ← Back
        </button>
      </div>
    );
  }

  if (!explanation || !riskTrend || !timelineEvents) {
    return <div className="app loading">Loading patient data…</div>;
  }

  return (
    <div className="app">
      <button className="back-button" onClick={onBack}>← Back</button>
      <Header />

      <section className="dashboard-grid">
        <div className="panel risk-summary">
          <h2>Risk Summary</h2>
          <div className={`big-risk ${explanation.risk_level.toLowerCase()}`}>
            {explanation.mortality_risk_percent}%
          </div>
          <div className="risk-meta">
            <span className={`badge ${explanation.risk_level.toLowerCase()}`}>
              {explanation.risk_level}
            </span>
            <span className="confidence-label">
              Confidence: {explanation.confidence.label}
            </span>
          </div>
          <p className="recommendation">{explanation.recommended_action}</p>
        </div>

        <div className="panel">
          <h2>Risk Trend</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={riskTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0f1a2e" />
              <XAxis
                dataKey="hour"
                tick={{ fill: "#4a6080", fontSize: 12 }}
                label={{ value: "ICU Hour", position: "insideBottom", offset: -2, fill: "#4a6080", fontSize: 12 }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: "#4a6080", fontSize: 12 }}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                contentStyle={{ background: "#0a0f1e", border: "1px solid #1e3a5f", borderRadius: 10, color: "#e2e8f0" }}
                formatter={(v) => [`${v}%`, "Mortality Risk"]}
              />
              <Line
                type="monotone"
                dataKey="mortality_risk_percent"
                stroke="#3b82f6"
                strokeWidth={2.5}
                dot={{ fill: "#3b82f6", r: 4 }}
                activeDot={{ r: 6, fill: "#60a5fa" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="dashboard-grid">
        <ExplanationPanel title="Top Risk Factors" items={explanation.risk_factors} type="risk" />
        <ExplanationPanel title="Protective Factors" items={explanation.protective_factors} type="protective" />
      </section>

      <section className="panel">
        <h2>Clinical Timeline</h2>
        <div className="timeline">
          {timelineEvents.slice(0, 40).map((event, index) => (
            <div className="timeline-item" key={index}>
              <div className="timeline-time">{event.time}</div>
              <div className="timeline-event">
                <strong>{event.event}</strong>
                <p>
                  <span className={`sev-tag ${event.severity.toLowerCase()}`}>
                    {event.severity}
                  </span>
                  {event.parameter}: {event.value}
                  {event.category ? ` · ${event.category}` : ""}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("list");
  const [selectedPatient, setSelectedPatient] = useState(null);

  if (selectedPatient) {
    return (
      <PatientDashboard
        recordId={selectedPatient}
        onBack={() => setSelectedPatient(null)}
      />
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
