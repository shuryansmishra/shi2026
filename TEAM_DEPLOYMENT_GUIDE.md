# 🛰️ ISRO SatQuery AI — Team Deployment & Architecture Guide

> **Official Deployment & Engineering Handbook**  
> *Everything required to deploy Frontend, Backend, PyTorch Models, and Qwen VLM with zero crashes.*

---

## 📑 Table of Contents
1. [Why Hosting the Full ML Backend on Vercel Failed](#1-why-hosting-the-full-ml-backend-on-vercel-failed)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Recommended Strategy: 100% Free 24/7 Deployment (Oracle Cloud VM)](#3-recommended-strategy-100-free-247-deployment-oracle-cloud-vm)
4. [Alternative Strategy: Hybrid Deployment (Vercel Frontend + Dedicated Backend)](#4-alternative-strategy-hybrid-deployment)
5. [Optional: Qwen Vision-Language Model (VLM) Deployment](#5-optional-qwen-vision-language-model-vlm-deployment)
6. [Operations & Maintenance Playbook](#6-operations--maintenance-playbook)

---

## 1. Why Hosting the Full ML Backend on Vercel Failed

When deploying the application to Vercel, the backend crashed and threw `405 / 500 / Function Size Exceeded` errors. Here is the technical post-mortem:

| Constraint | Vercel Serverless Function Limit | SatQuery ML Backend Requirement | Outcome on Vercel |
|---|---|---|---|
| **Bundle Size** | Max **250 MB** (uncompressed) | `torch` (~800 MB) + `torchvision` (~50 MB) + models (~176 MB) = **> 1.0 GB** | ❌ **Deployment fails / Function rejected** |
| **Execution Memory** | 1024 MB max (Hobby) | PyTorch model initialization requires **1.5 GB – 4 GB RAM** | ❌ **OOM (Out Of Memory) silent crash** |
| **Execution Timeout** | 10s – 60s hard timeout | Large GeoTIFF raster processing + SSIM matrix + VLM inference takes **5s – 30s** | ❌ **504 Gateway Timeout** |
| **File System** | Read-only ephemeral `/tmp` | Raster uploads, tiles, and GeoJSON polygon caching require local storage | ❌ **I/O write failures** |
| **Routing Wildcards** | Wildcard `/(.*) -> index.html` | Intercepts `/api/*` and `/health` before hitting serverless handlers | ❌ **API returns HTML string instead of JSON (405 error)** |

### ✅ How We Fixed This in the Codebase:
1. **Graceful Fallback on Vercel (`api/index.py` & `vision_models.py`):**
   - Wrapped all `torch` and `transformers` imports in `try / except` blocks with a `HAS_TORCH` flag.
   - When deployed to Vercel, the app automatically switches to **`VQA_MOCK_MODE=True`** (instant response, zero GPU/PyTorch requirement, under the 250MB limit).
2. **Fixed `vercel.json` SPA Regex:**
   - Changed rewrite rule to `/((?!api|health|static).*) -> /index.html` so API requests are never swallowed by React router.
3. **Dedicated Backend for Real ML:**
   - Real PyTorch models and Qwen VLM must run on a persistent Linux server (Oracle Cloud Always Free or Render).

---

## 2. System Architecture Overview

SatQuery AI consists of three interconnected layers:

```
                                    ┌────────────────────────────────────────┐
                                    │        FRONTEND (React + Vite)         │
                                    │  Mapbox GL, Tailwind, Telemetry Charts │
                                    └───────────────────┬────────────────────┘
                                                        │
                                                        │ REST API (JSON / FormData)
                                                        ▼
                                    ┌────────────────────────────────────────┐
                                    │         BACKEND (FastAPI API)          │
                                    │     LangGraph Agentic State Router     │
                                    └───────┬───────────┬───────────┬────────┘
                                            │           │           │
                     ┌──────────────────────┘           │           └──────────────────────┐
                     ▼                                  ▼                                  ▼
      ┌─────────────────────────────┐    ┌─────────────────────────────┐    ┌─────────────────────────────┐
      │     SingleImageEngine       │    │        ChangeEngine         │    │        FusionEngine         │
      │  TinySatCNN (ResNet18)      │    │  TinySiameseChange          │    │  TinyDualEncoderFusion      │
      │  Weights: 43 MB (.pt)       │    │  Weights: 44 MB (.pt)       │    │  Weights: 89 MB (.pt)       │
      │  BigEarthNet 10-class land  │    │  Bi-temporal + SSIM metric  │    │  Optical + SAR Cross-Attn   │
      └──────────────┬──────────────┘    └─────────────────────────────┘    └─────────────────────────────┘
                     │
                     ▼ (Optional)
      ┌─────────────────────────────┐
      │     Qwen2.5-VL VLM API      │
      │  Local 2B/3B or Colab 7B    │
      └─────────────────────────────┘
```

---

## 3. Recommended Strategy: 100% Free 24/7 Deployment (Oracle Cloud VM)

Oracle Cloud provides an **Always Free** tier with an Ampere A1 ARM compute instance:
* **4 OCPU Cores**
* **24 GB RAM**
* **200 GB Storage**
* **$0.00 / month forever**

This single instance can host the **Frontend (Nginx) + Backend (FastAPI) + All 3 Real PyTorch Models** with zero cross-origin issues.

### Step 3.1: Create the Free Instance on Oracle Cloud
1. Sign up at [cloud.oracle.com](https://cloud.oracle.com).
2. Go to **Compute** ➔ **Instances** ➔ **Create Instance**.
3. **Image:** Ubuntu 22.04 LTS.
4. **Shape:** `VM.Standard.A1.Flex` ➔ **4 OCPUs, 24 GB RAM**.
5. **Networking:** Assign a Public IPv4 address.
6. **SSH Key:** Download your private key (`.key`).
7. Click **Create** and copy your **Public IP**.

---

### Step 3.2: SSH into the Server
```bash
# On your local terminal (Mac/Linux/WSL):
chmod 400 ~/Downloads/ssh-key-*.key
ssh -i ~/Downloads/ssh-key-*.key ubuntu@<YOUR_ORACLE_PUBLIC_IP>
```

---

### Step 3.3: Install System Dependencies & Git LFS
```bash
# Update and install build tools
sudo apt update && sudo apt install -y python3-pip python3-venv git git-lfs nginx curl iptables-persistent

# Install Node.js 20 LTS (for building frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Initialize Git LFS
git lfs install
```

---

### Step 3.4: Clone Codebase & Build Frontend
```bash
# Clone the repository (Git LFS automatically downloads the 176MB .pt weights)
git clone https://github.com/shuryansmishra/shi2026.git
cd shi2026

# Build the React production bundle
cd frontend
npm install
npm run build
cd ..

# Verify models were downloaded
ls -lh backend/checkpoints/
# Should display: tinysat_cnn_best.pt (43M), siamese_change_best.pt (44M), optical_sar_fusion_best.pt (89M)
```

---

### Step 3.5: Set Up Python Backend with PyTorch
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create production .env file
cat > .env << 'EOF'
ENV=production
DEBUG=false
VQA_MOCK_MODE=false
CHANGE_MODEL_PATH=./checkpoints/siamese_change_best.pt
FUSION_MODEL_PATH=./checkpoints/optical_sar_fusion_best.pt
ALLOWED_ORIGINS=*
EOF
```

---

### Step 3.6: Configure Systemd Daemon (Auto-restart on boot/crash)
```bash
sudo tee /etc/systemd/system/satquery.service << 'EOF'
[Unit]
Description=SatQuery AI PyTorch Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/shi2026/backend
Environment="PATH=/home/ubuntu/shi2026/backend/venv/bin"
ExecStart=/home/ubuntu/shi2026/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable satquery
sudo systemctl start satquery

# Verify status:
sudo systemctl status satquery
```

---

### Step 3.7: Configure Nginx Reverse Proxy
```bash
sudo tee /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    # Serve built React static files
    root /home/ubuntu/shi2026/dist;
    index index.html;

    # SPA Client Routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API endpoints to FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 200M;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
    }

    location /static/ {
        proxy_pass http://127.0.0.1:8000;
        client_max_body_size 200M;
    }
}
EOF

# Ensure permissions & restart Nginx
chmod 755 /home/ubuntu
sudo nginx -t && sudo systemctl restart nginx

# Open firewall for HTTP traffic
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

*(Also ensure Port 80 ingress rule is added in Oracle Cloud Console: **Networking** ➔ **VCN** ➔ **Security Lists** ➔ **Add Ingress Rule: TCP Port 80**).*

---

## 4. Alternative Strategy: Hybrid Deployment

If you prefer keeping the frontend hosted on **Vercel** (`https://shi2026.vercel.app`):

1. Deploy the backend on **Oracle Cloud** or **Render** (as configured in `render.yaml`).
2. Generate an HTTPS URL for your backend (e.g., using Cloudflare Tunnel on Oracle):
   ```bash
   # On Oracle VM:
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared.deb
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Copy the resulting HTTPS tunnel URL (e.g., `https://your-tunnel.trycloudflare.com`).
4. In **Vercel Dashboard** ➔ `shi2026` project ➔ **Settings** ➔ **Environment Variables**:
   - `VITE_BACKEND_URL` = `https://your-tunnel.trycloudflare.com`
5. **Redeploy** on Vercel.

---

## 5. Optional: Qwen Vision-Language Model (VLM) Deployment

The project includes hooks for **Qwen2.5-VL** (found in `SatQuery_Preprocessing (5).ipynb`). You can deploy Qwen in one of two ways depending on GPU availability:

### Option A: Free & Fast GPU Inference via Google Colab (Recommended for Live Demos)
Run the 7B model on Colab's free NVIDIA T4 GPU and tunnel responses to your backend:

1. Open `SatQuery_Preprocessing (5).ipynb` in Google Colab (Set Runtime to **T4 GPU**).
2. Add this cell at the bottom and run it:
   ```python
   !pip install flask pyngrok -q
   from flask import Flask, request, jsonify
   from pyngrok import ngrok
   import torch

   app = Flask(__name__)

   @app.route("/infer", methods=["POST"])
   def infer():
       data = request.json
       question = data.get("query_text", "Describe terrain features in this satellite image.")
       image_b64 = data.get("image_b64")

       messages = [{
           "role": "user",
           "content": [
               {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
               {"type": "text", "text": question}
           ]
       }]
       text = processor.apply_chat_template(messages, add_generation_prompt=True)
       image_inputs, _ = process_vision_info(messages)
       inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")

       with torch.inference_mode():
           generated_ids = model.generate(**inputs, max_new_tokens=128)
           
       out = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
       return jsonify({"answer": out, "confidence": 0.95})

   # Set ngrok token from ngrok.com (free)
   ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
   tunnel = ngrok.connect(5000)
   print("🚀 Set this in backend/.env -> QWEN_REMOTE_URL =", f"{tunnel.public_url}/infer")
   app.run(port=5000)
   ```
3. In `backend/.env`, set:
   ```ini
   QWEN_REMOTE_URL=https://your-ngrok-subdomain.ngrok-free.app/infer
   ```

---

### Option B: 24/7 Self-Hosted VLM directly on Oracle CPU (No Colab Needed)
If you want Qwen running 24/7 on the Oracle VM without opening Colab, use the lightweight **2B parameter model** (executes in 3–5 seconds on 4 ARM CPU cores):

In `backend/.env`:
```ini
VQA_MOCK_MODE=false
QWEN_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct
QWEN_DEVICE=cpu
```

---

## 6. Operations & Maintenance Playbook

### Checking Server Health
```bash
# Test local FastAPI process
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# Expected Output:
# {
#   "status": "ok",
#   "app": "SatQuery AI",
#   "mock_mode": false
# }
```

### Viewing Backend Live Logs
```bash
sudo journalctl -u satquery -f
```

### Updating Code on the Server
Whenever new code is pushed to GitHub:
```bash
cd /home/ubuntu/shi2026
git pull

# Rebuild frontend if UI changed
cd frontend && npm install && npm run build && cd ..

# Restart backend service
sudo systemctl restart satquery
```

---

## 📊 Summary Cheat Sheet for Teammates

| Item | Recommendation | Cost |
|---|---|---|
| **Hosting Platform** | Oracle Cloud Always Free (Ampere A1) | **$0 / month** |
| **System Specs** | 4 ARM Cores, 24 GB RAM, 200 GB Disk | Included Free |
| **Core Vision ML** | `TinySatCNN` + `SiameseChange` + `FusionEngine` | Real PyTorch (`mock_mode: false`) |
| **Vision-Language** | `Qwen2-VL-2B` (on VM) or `Qwen2.5-VL-7B` (via Colab Tunnel) | Fully Supported |
| **Frontend Server** | Nginx Reverse Proxy (Port 80/443) | Included Free |

*(Document prepared for ISRO SatQuery AI Development Team).*
