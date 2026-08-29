#!/bin/bash
set -e

echo "=========================================="
echo " SatQuery AI - Full Demo Launcher"
echo "=========================================="

cd "$(dirname "$0")"

# 1. Setup Backend
echo "[1/4] Setting up backend and ML dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# 2. Generate Demo Data
echo "[2/4] Generating demo data (GeoTIFFs)..."
# Use the backend virtualenv which has rasterio installed
./backend/venv/bin/python demo_data/generate.py

# 3. Start Backend in Background
echo "[3/4] Starting FastAPI backend (mock mode with real PyTorch / RL)..."
# Ensure we load local HuggingFace and real Tiny ML models
export VQA_MOCK_MODE=False
export LLM_PROVIDER="local"
cd backend
uvicorn main:app --port 8000 &
BACKEND_PID=$!
cd ..

# 4. Start Frontend
echo "[4/4] Starting React frontend..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!

echo "=========================================="
echo " Services are starting up!"
echo " Backend: http://localhost:8000"
echo " Frontend: http://localhost:5173"
echo " Press Ctrl+C to stop all services."
echo "=========================================="

# Wait and catch Ctrl+C
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" SIGINT SIGTERM
wait
