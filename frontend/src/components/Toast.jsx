import React from "react";

export default function Toast({ message, visible }) {
  if (!message) return null;
  return (
    <div className={`toast-msg ${visible ? "show" : ""}`} role="status" aria-live="polite">
      <span>🛰️</span>
      <span>{message}</span>
    </div>
  );
}
