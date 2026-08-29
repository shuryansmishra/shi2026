import React from "react";

/**
 * Renders the graded execution trace (backend/models/schemas.py
 * ExecutionTrace). Per the PS: only this observable trace -- task selected,
 * tool/model used, parameters, outputs -- is evaluated. Showing it plainly
 * in the UI doubles as a transparency feature for end users.
 */
export default function TraceViewer({ trace, route }) {
  if (!trace) return null;

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>Execution trace</h3>
      {route && (
        <p style={styles.routeLine}>
          Routed to <strong>{route.task_type}</strong> — {route.reason}
        </p>
      )}
      <ol style={styles.list}>
        {trace.steps.map((step, i) => (
          <li key={i} style={styles.step}>
            <div style={styles.stepHeader}>
              <span style={styles.stepName}>{step.step}</span>
              <span style={styles.component}>{step.component}</span>
            </div>
            <div style={styles.params}>
              params: {JSON.stringify(step.parameters)}
            </div>
            <div style={styles.output}>{step.output_summary}</div>
          </li>
        ))}
      </ol>
    </div>
  );
}

const styles = {
  panel: { border: "1px solid #ddd", borderRadius: 8, padding: 16 },
  heading: { margin: "0 0 8px 0" },
  routeLine: { fontSize: 13, color: "#333" },
  list: { paddingLeft: 20 },
  step: { marginBottom: 12, fontSize: 13 },
  stepHeader: { display: "flex", justifyContent: "space-between", fontWeight: 600 },
  component: { color: "#0a6", fontWeight: 400 },
  params: { color: "#777", fontFamily: "monospace", fontSize: 11, marginTop: 2 },
  output: { marginTop: 2 },
};
