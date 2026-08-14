export default function ExplanationPanel({ title, items, type }) {
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
