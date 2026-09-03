#!/usr/bin/env bash
# =============================================================================
# SatQuery AI - Oracle Cloud Backend Deployment Script
# =============================================================================
# Run this script on your Oracle Cloud Ubuntu instance:
#   chmod +x deploy_oracle.sh && ./deploy_oracle.sh
# =============================================================================

set -e

echo "🚀 Starting SatQuery AI Backend Setup on Oracle Cloud..."

# 1. Update system and install required system packages
sudo apt update
sudo apt install -y python3-pip python3-venv git git-lfs curl iptables-persistent

# 2. Setup Git LFS to pull real model checkpoints
git lfs install
git lfs pull || true

# 3. Create Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure Production Environment File
if [ ! -f ".env" ]; then
    echo "⚙️ Creating backend .env..."
    cat > .env << 'EOF'
ENV=production
DEBUG=false
VQA_MOCK_MODE=false
ALLOWED_ORIGINS=*
UPLOAD_DIR=./storage/uploads
PROCESSED_DIR=./storage/processed
PUBLIC_STORAGE_DIR=./storage/public
EOF
fi

# 5. Open Firewall Port 8000 on Oracle Cloud Ubuntu
echo "🔓 Configuring firewall for port 8000..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save || true

# 6. Create and Enable Systemd Service
CURRENT_DIR=$(pwd)
CURRENT_USER=$(whoami)

echo "🛠️ Creating systemd service /etc/systemd/system/satquery.service..."
sudo tee /etc/systemd/system/satquery.service > /dev/null << EOF
[Unit]
Description=SatQuery AI PyTorch Backend
After=network.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment="PATH=$CURRENT_DIR/venv/bin"
ExecStart=$CURRENT_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable satquery.service
sudo systemctl restart satquery.service

echo ""
echo "================================================================="
echo "✅ SatQuery AI Backend is now RUNNING on Oracle Cloud!"
echo "📡 Check status: sudo systemctl status satquery"
echo "🔍 Live test: curl http://localhost:8000/health"
echo "🌐 Connect from Vercel: Set VITE_BACKEND_URL=http://<YOUR_PUBLIC_IP>:8000"
echo "================================================================="
