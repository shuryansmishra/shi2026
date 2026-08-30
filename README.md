# 🛰️ ISRO SatQuery AI — Next-Gen Multimodal Geospatial VQA & Change Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Vite](https://img.shields.io/badge/Vite-8.2.2-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Mapbox GL](https://img.shields.io/badge/Mapbox_GL_JS-v3.18-000000?style=for-the-badge&logo=mapbox&logoColor=white)](https://www.mapbox.com)
[![Vercel](https://img.shields.io/badge/Vercel-Fullstack_Ready-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**SatQuery AI** is an advanced agentic, multimodal remote-sensing AI platform designed for the **Smart India Hackathon (SIH26167 - ISRO / Space Technology Track)**. It bridges high-resolution satellite Earth Observation (EO) data with vision-language foundation models and computer vision pipelines to enable natural language visual question answering (VQA), bi-temporal change detection, and optical-SAR sensor fusion.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Frontend Architecture & UI Design System](#-frontend-architecture--ui-design-system)
3. [Backend Pipeline & Agentic Core](#-backend-pipeline--agentic-core)
4. [Machine Learning Redefinition & Training Strategy](#-machine-learning-redefinition--training-strategy)
5. [Frontend-Backend Integration & Communication](#-frontend-backend-integration--communication)
6. [Known Errors, Edge Cases & Hallucination Mitigation](#-known-errors-edge-cases--hallucination-mitigation)
7. [API Specification & Endpoints](#-api-specification--endpoints)
8. [Getting Started & Local Development](#-getting-started--local-development)
9. [Vercel Fullstack Deployment Guide](#-vercel-fullstack-deployment-guide)
10. [Repository Structure](#-repository-structure)

---

## 🏛️ System Architecture

SatQuery AI uses a modular, decoupled architecture where user queries pass through an **Agentic Router**, get processed by specialized **Vision Engines**, pass through a strict **Evidence Engine (Hallucination Firewall)**, and are synthesized by an **LLM Synthesis Layer** with auditable visual traces.

```
                               ┌───────────────────────────────────────────────────────────┐
                               │                    React 18 Frontend                      │
                               │  Mapbox GL High-Res Satellites • Lottie • Glassmorphic UI  │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
                                        POST /api/query or /api/query_by_location
                                                             │
                                                             ▼
                               ┌───────────────────────────────────────────────────────────┐
                               │                   FastAPI / Serverless                    │
                               │                (main.py / api/index.py)                   │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
                                                             ▼
                               ┌───────────────────────────────────────────────────────────┐
                               │                      Agentic Router                       │
                               │            (Intent Classification & Dispatch)             │
                               └──────┬──────────────────────┼──────────────────────┬──────┘
                                      │                      │                      │
                   Single Image Scene │     Bi-Temporal Pair │     Multi-Sensor Pair│
                                      ▼                      ▼                      ▼
                          ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
                          │ Single-Image Engine  ││    Change Engine     ││    Fusion Engine     │
                          │   (ViT / Qwen-VL)    ││   (Siamese / SSIM)   ││ (Cross-Attention)    │
                          └──────────┬───────────┘└──────────┬───────────┘└──────────┬───────────┘
                                     │                       │                       │
                                     └───────────────────────┼───────────────────────┘
                                                             ▼
                               ┌───────────────────────────────────────────────────────────┐
                               │               Evidence Engine & Firewall                  │
                               │          (Locked Evidence JSON + Numerical Facts)         │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
                                                             ▼
                               ┌───────────────────────────────────────────────────────────┐
                               │                   LLM Synthesis Layer                     │
                               │               (Grounded Answer Generation)                │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
                                                             ▼
                               ┌───────────────────────────────────────────────────────────┐
                               │              Structured Execution Trace                   │
                               │          Answer + Evidence Details + Confidence           │
                               └───────────────────────────────────────────────────────────┘
```

---

## 🎨 Frontend Architecture & UI Design System

The frontend is built using **React 18 + Vite** with a bespoke **Apple / Figma Glassmorphic Design System** (`index.css`):

### 🌟 Key UI/UX Highlights:
1. **Interactive Dual-Mapbox Change Detection (`ChangeDetection.jsx`)**:
   - Two synchronized high-resolution satellite maps (T1 Baseline vs T2 Target).
   - **Linked Pan/Zoom Toggle**: One-click lock/unlock button with vector SVG chain links.
   - **Top-Docked Metadata Headers**: T1 Baseline is anchored to the top-left (`left: 14px`) and T2 Target to the top-right (`right: 14px`), preventing any visual overlap with the center synchronization button.
   - **Dynamic Difference Overlay**: Real-time opacity slider (0% to 100%) and toggleable change vector masks.

2. **Single-Image Geospatial VQA (`SingleImageVQA.jsx`)**:
   - Smooth Mapbox viewport animations (`flyTo`) for preset Indian regions (Delhi NCR, Mumbai Coast, Bengaluru Corridor, Chilika Lake, Punjab Farmlands).
   - Natural language search bar and manual Coordinate Quick Jump (`lat`/`lng`).
   - Custom GeoTIFF / PNG / JPEG drag-and-drop file uploader.

3. **Live Satellite Explorer (`LiveMapSelection.jsx`)**:
   - Multi-layer Satellite Base Switcher (Satellite High-Res, Street, Dark, Outdoors) with dynamic sliding capsule pill animation.
   - **Exact Figma Coordinate Combo Boxes**: 24px height, 6px border radius, multiply blend-mode border, and dedicated `°N` / `°E` indicator buttons.

4. **Lottie Vector Login Modal (`LoginModal.jsx` & `Navbar.jsx`)**:
   - Ultra-sleek **Obsidian Zinc (`#18181B`) & ISRO Blue (`#0056A7`) Sign In button** with natural specular light glint and interactive hover micro-animations.
   - 60fps vector Lottie animation (`Login.json`) rendered via lightweight player.
   - Live pulsating prototype phase notice informing users of upcoming features and roadmap status.

5. **2-Tier Mobile & Tablet Responsiveness**:
   - Dynamic 2-tier navigation bar on screens `< 840px` with segmented slider pill controls.
   - Smart viewport reordering (`order: 1` on maps) so satellite imagery is immediately visible on top of sidebars on mobile devices.

---

## 🧠 Backend Pipeline & Agentic Core

The backend is built on **FastAPI** with modular engines:

### 1. Agentic Router (`backend/core/router.py`)
- Analyzes incoming requests (image count, sensor modalities, question intent keywords).
- Automatically routes queries to:
  - `TaskType.SINGLE_IMAGE_VQA` (e.g. object counts, infrastructure identification, land-use classification).
  - `TaskType.CHANGE_DETECTION` (e.g. flood inundation extent, deforestation delta, urban expansion).
  - `TaskType.CROSS_MODAL_FUSION` (e.g. cloud-penetrating SAR radar + optical RGB fusion).

### 2. Evidence Engine / Hallucination Firewall (`backend/core/evidence_engine.py`)
- Vision models output raw bounding boxes, segmentation masks, and pixel deltas.
- The **Evidence Engine** converts raw computer vision tensors into a **Locked Evidence JSON Object**.
- The LLM is strictly prohibited from inventing numbers or facts; it can only explain facts verified within the evidence object.

### 3. Location Resolver & Geocoding (`backend/ingestion/location_resolver.py`)
- Geocodes Indian districts, cities, and landmarks into precise WGS84 bounding boxes.
- In mock/offline mode, generates deterministic synthetic GeoTIFF rasters with geospatial CRS metadata.
- Prepares automated catalog queries for Sentinel-2, Landsat-8, and ISRO Bhoonidhi APIs.

---

## 🔬 Machine Learning Redefinition & Training Strategy

SatQuery AI is designed with a **two-phase machine learning lifecycle**:

```
┌───────────────────────────────────────────────────────────┐
│              PHASE 1: Standalone Mock Mode                │
│ (VQA_MOCK_MODE=True • Fast, GPU-Free, Deterministic Demos)│
└─────────────────────────────┬─────────────────────────────┘
                              │
                    Fine-Tuning on Real Data
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│            PHASE 2: Production Deep Learning              │
│  (Qwen 2.5-VL LoRA • Siamese VisTA CNN • Dual Encoders)   │
└───────────────────────────────────────────────────────────┘
```

### 1. Single-Image VQA (Qwen 2.5-VL / BigEarthNet)
- **Architecture**: Vision Transformer (ViT) backbone + Language Model decoder.
- **Training Script**: `backend/training/train_qwen_vl_lora.py`
- **Dataset**: BigEarthNet, RSVQA (Remote Sensing Visual Question Answering), and ISRO Cartosat-2/3 imagery.
- **Activation**: Set `VQA_MOCK_MODE=False` and `VQA_MODEL_PATH=/path/to/checkpoint` in `.env`.

### 2. Bi-Temporal Change Detection (Siamese ResNet / VisTA)
- **Architecture**: Dual-branch Siamese network with difference feature pyramid and Structural Similarity (SSIM) delta thresholding.
- **Dataset**: CDVQA (Change Detection Visual Question Answering), LEVIR-CD, and WHU-CD.
- **Activation**: Set `CHANGE_MODEL_PATH=/path/to/checkpoint` in `.env`.

### 3. Optical-SAR Cross-Modal Fusion
- **Architecture**: Dual-encoder cross-attention mechanism aligning Optical RGB bands (Sentinel-2 / LISS-IV) with SAR C-band / L-band Radar (RISAT / Sentinel-1).
- **Activation**: Set `FUSION_MODEL_PATH=/path/to/checkpoint` in `.env`.

---

## 🔗 Frontend-Backend Integration & Communication

### Dynamic Base URL Resolution (`frontend/src/api.js`)
```javascript
const API_BASE = import.meta.env.VITE_BACKEND_URL 
  ? import.meta.env.VITE_BACKEND_URL.replace(/\/$/, "") 
  : "";
```

- **Local Development**: `API_BASE` is `""`. Vite proxy forwards `/api` and `/health` to `http://localhost:8000`.
- **Vercel Fullstack Deployment**: `vercel.json` routes `/api/(.*)` directly to `api/index.py` serverless function on the same domain.
- **Distributed Deployment (e.g. Vercel Frontend + Render Backend)**: Set `VITE_BACKEND_URL=https://your-backend.onrender.com` in Vercel Environment Variables.

---

## ⚠️ Known Errors, Edge Cases & Hallucination Mitigation

| Area | Challenge / Edge Case | System Mitigation Strategy |
| :--- | :--- | :--- |
| **Model Inferences** | ML models in beta phase may misclassify ambiguous land cover. | **Beta Warning Banner** is permanently rendered on UI informing users of ongoing active training. |
| **LLM Hallucinations** | Vision models might overestimate crop area or building counts. | **Evidence Engine Firewall**: The LLM prompt is mathematically constrained to only cite numbers from the locked evidence JSON. |
| **Serverless Memory Limits** | Vercel serverless functions have a 250MB limit (heavy PyTorch weights cannot load). | `os.environ.setdefault("VQA_MOCK_MODE", "True")` in `api/index.py` ensures 100% uptime with deterministic simulation on serverless. |
| **Mapbox Coordinate Boundary** | Clicking coordinates outside valid raster boundaries. | Auto-clamping to WGS84 standard `[-90, 90]` latitude and `[-180, 180]` longitude with visual toast error alerts. |
| **CORS Access** | Cross-Origin errors when deploying on distinct domains. | FastAPI backend includes pre-configured `CORSMiddleware` with `allow_origins=["*"]`. |

---

## 📡 API Specification & Endpoints

### 1. `POST /api/query`
Executes multimodal VQA or Change Detection on uploaded images.
- **Parameters (Multipart Form-Data)**:
  - `query_text` (string): Natural language user question.
  - `files` (array of binary files): 1 or 2 satellite rasters (PNG / TIFF / GeoTIFF).
  - `capture_dates` (optional array of strings): ISO dates for temporal analysis (e.g. `["2020-01-15", "2024-08-20"]`).
- **Response**:
  ```json
  {
    "answer": "Detected 12.4 km² of urban expansion in northwest quadrant.",
    "task_type": "change_detection",
    "evidence": {
      "change_detected": true,
      "change_area_km2": 12.4,
      "confidence": 0.942
    },
    "trace": [
      { "title": "Router", "desc": "Assigned to Bi-Temporal Change Detection Engine." },
      { "title": "Vision Engine", "desc": "Calculated structural similarity delta across T1 and T2." },
      { "title": "Evidence Synthesis", "desc": "Grounded response generated with 94.2% confidence." }
    ]
  }
  ```

### 2. `POST /api/query_by_location`
Resolves a location name, retrieves satellite imagery, and runs analysis.
- **Parameters**:
  - `query_text` (string): User prompt.
  - `place_name` (string): Location name (e.g. `"Delhi NCR"`).

### 3. `GET /health`
Returns real-time backend pipeline and engine status.
- **Response**: `{"status": "ok", "app": "SatQuery AI", "mock_mode": true}`

---

## 💻 Getting Started & Local Development

### Prerequisites:
- **Node.js**: v18.0 or higher
- **Python**: v3.10 or v3.11

### 1. Clone Repository & Switch to Frontend Branch:
```bash
git clone https://github.com/shuryansmishra/shi2026.git
cd shi2026
git checkout Frontend
```

### 2. Start FastAPI Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start React Frontend:
```bash
cd ../frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 🚀 Vercel Fullstack Deployment Guide

Both the **React Frontend** and **FastAPI Backend** can be deployed together on **Vercel** with zero extra infrastructure:

1. Go to [Vercel New Project](https://vercel.com/new) and import `shuryansmishra/shi2026`.
2. Select branch: **`Frontend`**.
3. **Root Directory**: Keep as **`./`** (Vercel automatically detects the root `vercel.json`).
4. **Environment Variables**:
   - `VITE_MAPBOX_TOKEN` = `pk.eyJ1IjoiYWpsYWFuOTkxOSIsImEiOiJjbXQ4dzV3NHowMWF1MndzaGJjeGdmaHYyIn0.ztQua4BZO5JbZanQqVrKWw`
   - `VQA_MOCK_MODE` = `True`
5. Click **Deploy**. Your fullstack AI application will be live in seconds!

---

## 📂 Repository Structure

```
shi2026/
├── vercel.json                  # Monorepo fullstack Vercel edge routing
├── api/                         # Vercel Serverless Python Adapter
│   ├── index.py                 # Serverless FastAPI entrypoint
│   └── requirements.txt         # Serverless Python dependencies
├── backend/                     # FastAPI Core Pipeline
│   ├── main.py                  # API endpoints & middleware
│   ├── config.py                # Environment configurations
│   ├── core/                    # Agentic Router & Evidence Firewall
│   │   ├── router.py            # Task intent classifier
│   │   └── evidence_engine.py   # Hallucination firewall
│   ├── engines/                 # Vision & Change Detection Engines
│   │   ├── single_image_engine.py
│   │   ├── change_engine.py
│   │   └── fusion_engine.py
│   ├── ingestion/               # Geocoding & GeoTIFF preprocessing
│   ├── llm/                     # Grounded LLM synthesis layer
│   ├── models/schemas.py        # Pydantic data schemas
│   └── requirements.txt         # Full backend dependencies
├── frontend/                    # React 18 + Vite Application
│   ├── src/
│   │   ├── App.jsx              # Main application shell
│   │   ├── index.css            # Glassmorphic Apple/Figma design system
│   │   ├── api.js               # Dynamic API client
│   │   ├── assets/Login.json    # Vector Lottie animation
│   │   └── components/
│   │       ├── Navbar.jsx       # 2-tier responsive nav & Sign In button
│   │       ├── SingleImageVQA.jsx   # Single image satellite VQA
│   │       ├── ChangeDetection.jsx  # Dual split Mapbox change detection
│   │       ├── LiveMapSelection.jsx # Multi-style satellite explorer
│   │       ├── LoginModal.jsx   # Prototype phase Lottie modal
│   │       └── Toast.jsx        # Haptic alert toast notifications
│   ├── package.json
│   ├── vercel.json              # SPA client routing configuration
│   └── vite.config.js
└── README.md                    # System documentation
```

---

## 👥 Team & Acknowledgments
- **Project**: SatQuery AI (SIH26167)
- **Track**: ISRO / Space Technology
- **Repository**: [https://github.com/shuryansmishra/shi2026](https://github.com/shuryansmishra/shi2026)

---
*Built with ❤️ for Space Technology & Geospatial AI Innovation.*
