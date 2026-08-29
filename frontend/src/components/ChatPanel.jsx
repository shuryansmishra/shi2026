import React, { useState } from "react";

export default function ChatPanel({ onSubmit, loading }) {
  const [text, setText] = useState("");

  function submit(e) {
    e.preventDefault();
    if (!text.trim()) return;
    onSubmit(text.trim());
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>2. Ask your question</h3>
      <form onSubmit={submit} style={styles.form}>
        <input
          style={styles.input}
          placeholder='e.g. "What changed between these two images?"'
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" style={styles.button} disabled={loading}>
          {loading ? "Running..." : "Ask"}
        </button>
      </form>
    </div>
  );
}

const styles = {
  panel: { border: "1px solid #ddd", borderRadius: 8, padding: 16, marginBottom: 16 },
  heading: { margin: "0 0 8px 0" },
  form: { display: "flex", gap: 8 },
  input: { flex: 1, padding: 8, fontSize: 14 },
  button: { padding: "8px 16px", fontSize: 14, cursor: "pointer" },
};
