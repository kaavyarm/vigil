const API_BASE = "http://127.0.0.1:8001";

export async function fetchPatients() {
  const res = await fetch(`${API_BASE}/patients?limit=500`);
  if (!res.ok) throw new Error("Failed to fetch patients");
  return res.json();
}

export async function fetchPatient(recordId) {
  const res = await fetch(`${API_BASE}/patients/${recordId}`);
  if (!res.ok) throw new Error("Failed to fetch patient");
  return res.json();
}

export async function fetchExplanation(recordId) {
  const res = await fetch(`${API_BASE}/patients/${recordId}/explanation`);
  if (!res.ok) throw new Error("Failed to fetch explanation");
  return res.json();
}

export async function fetchRiskTrend(recordId) {
  const res = await fetch(`${API_BASE}/patients/${recordId}/risk-trend`);
  if (!res.ok) throw new Error("Failed to fetch risk trend");
  return res.json();
}

export async function fetchTimelineEvents(recordId) {
  const res = await fetch(`${API_BASE}/patients/${recordId}/timeline-events`);
  if (!res.ok) throw new Error("Failed to fetch timeline events");
  return res.json();
}

export async function predictCustomPatient(vitals) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vitals),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}