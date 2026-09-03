import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // All /api, /health, /static requests go to the FastAPI backend.
      // changeOrigin rewrites the Host header so FastAPI CORS middleware
      // sees the request coming from localhost:8000, not localhost:5173.
      // This means the browser never makes a cross-origin request → no CORS preflight → no 405.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/static": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
