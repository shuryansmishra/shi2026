import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Frontend calls /api/*, /health, /static, Vite forwards to the
      // FastAPI backend -- avoids CORS friction during local development.
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/static": "http://localhost:8000",
    },
  },
});
