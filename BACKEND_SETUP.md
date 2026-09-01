# 🚀 Backend Setup Guide - SatQuery AI

## ⚠️ Current Status: Backend Connection Issues

You're seeing:
```
⚠️ Backend Connection Warning: Could not reach FastAPI model server at http://localhost:8000
Status: Offline
Port: 8000
```

This means the **backend is not running** or **not responding correctly**.

---

## ✅ Step-by-Step Setup

### **Step 1: Ensure You're in the Backend Directory**
```bash
cd shi2026/backend
```

### **Step 2: Create Python Virtual Environment**
```bash
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### **Step 3: Create .env File**
Copy the example and fill in your values:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
VQA_MOCK_MODE=True  # This is crucial - allows running without GPU/models
ENV=development
DEBUG=True
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8000
```

### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

⚠️ **If you get GDAL errors** (for rasterio):
```bash
# macOS:
brew install gdal

# Ubuntu/Debian:
sudo apt-get install gdal-bin libgdal-dev

# Then retry:
pip install -r requirements.txt
```

### **Step 5: Start the Backend**
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### **Step 6: Verify Backend is Running**
Open a **new terminal** and test:
```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected output:
# {"status":"ok","app":"SatQuery AI","mock_mode":true}
```

---

## 🛠️ Troubleshooting

### **Error: "ModuleNotFoundError: No module named 'rasterio'"**
```bash
# Install geospatial dependencies
pip install rasterio shapely numpy

# If still failing, try:
pip install --upgrade rasterio
```

### **Error: "ModuleNotFoundError: No module named 'torch'"**
```bash
# Install PyTorch (CPU version for development)
pip install torch torchvision

# For GPU support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Error: "Connection refused" on port 8000**
- Make sure backend is actually running (check Step 5)
- Port 8000 might be in use: `lsof -i :8000` (macOS/Linux)
- Try a different port: `python -m uvicorn main:app --port 8001`

### **Error: "CORS error" in browser console**
- Make sure `ALLOWED_ORIGINS` in `.env` includes your frontend URL
- Default includes `http://localhost:5173` (Vite dev server)

---

## 📊 Expected Behavior

When running correctly:

1. **Backend starts:**
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   INFO:     Application startup complete
   ```

2. **Health check returns:**
   ```json
   {
     "status": "ok",
     "app": "SatQuery AI",
     "mock_mode": true
   }
   ```

3. **Frontend displays:** ✅ Status: Active (not Offline)

4. **Upload images and submit query** → Backend processes and returns results

---

## 🎯 Next: Frontend Setup

Once backend is running, open a **new terminal** and:

```bash
cd shi2026/frontend

# Create .env.local
cp .env.example .env.local

# Edit .env.local with your Mapbox token (optional for demo)
# VITE_MAPBOX_TOKEN=pk.your_token_here
# VITE_BACKEND_URL=http://localhost:8000

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend should be at: **http://localhost:5173**

---

## 🚨 Common Issues & Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| **405 Method Not Allowed** | Frontend sending GET instead of POST | Check frontend API client (should use `fetch(..., {method: 'POST'})`) |
| **CORS Error** | Frontend origin not in allowed list | Add origin to `ALLOWED_ORIGINS` in .env |
| **Backend offline** | Backend not running on port 8000 | Run `uvicorn main:app --reload --port 8000` |
| **Import errors** | Missing dependencies | Run `pip install -r requirements.txt` |
| **GDAL/Rasterio errors** | Geospatial library not installed | Install GDAL system package first |
| **Torch errors** | PyTorch not installed | Run `pip install torch torchvision` |

---

## ✅ Verification Checklist

- [ ] Backend running at `http://localhost:8000`
- [ ] `curl http://localhost:8000/health` returns status `ok`
- [ ] `.env` file created with `VQA_MOCK_MODE=True`
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend sees "Status: Active" (not Offline)
- [ ] Can upload images without 405 errors

---

## 📞 Need Help?

1. Check the execution trace in the UI (if available)
2. Look at backend terminal output for error messages
3. Test endpoints directly with curl:
   ```bash
   # Test health
   curl http://localhost:8000/health
   
   # Test with sample files
   curl -X POST http://localhost:8000/api/query \
     -F "query_text=test query" \
     -F "files=@test_image.png"
   ```
