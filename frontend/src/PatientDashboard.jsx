import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { fetchExplanation, fetchRiskTrend, fetchTimelineEvents } from "./api";
import ExplanationPanel from "./ExplanationPanel";

function Header() {
  return (
    <div className="header">
      <h1 className="vigil-title">Vigil</h1>
      <p className="subtitle">ICU Mortality Risk Monitoring Dashboard</p>
    </div>
  );
}

export default function PatientDashboard({ recordId, onBack }) {
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
