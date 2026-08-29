# SatQuery AI — SIH26167 (ISRO, Space Technology Track)

Agentic, multi-modal remote-sensing assistant: ask a natural-language
question of 1-2 satellite images, and a rule-based router sends it to the
right specialist engine (single-image VQA, bi-temporal change detection, or
optical-SAR fusion), then returns a grounded answer with a map overlay and
an auditable execution trace. See `SatQuery_AI_PRD_v2.md` for the full
research/architecture writeup this code implements.

## What's here

```
satquery-ai/
├── backend/            # FastAPI app -- runs fully end-to-end in mock mode, no GPU/keys needed
│   ├── config.py           # all settings, env-driven, safe defaults
│   ├── main.py             # /api/query and /health endpoints
│   ├── models/schemas.py   # shared Pydantic contracts (the "locked evidence object" lives here)
│   ├── core/
│   │   ├── router.py           # rule-based agentic router (image_count, modality) -> task
│   │   └── evidence_engine.py  # hallucination firewall: only place raw CV output becomes "facts"
│   ├── engines/
│   │   ├── single_image_engine.py  # VQA + caption/grounding (mock now, BigEarthNet.txt VLM later)
│   │   ├── change_engine.py        # bi-temporal change-VQA (mock now, CDVQA/VisTA later)
│   │   └── fusion_engine.py        # optical-SAR fusion (mock now, dual-encoder later)
│   ├── llm/synthesis.py    # answers ONLY from the evidence object; Anthropic API or template fallback
│   ├── ingestion/preprocessing.py  # modality detection, CRS reprojection, cloud-cover estimate
│   ├── data_access/bhoonidhi_client.py  # real ISRO Bhoonidhi community client wrapper
│   ├── tests/               # pytest -- router + evidence engine logic, all passing
│   └── requirements.txt
├── frontend/            # React + Vite + Leaflet -- upload, chat, map overlay, trace viewer
│   └── src/
├── demo_data/           # cache real Cartosat-2S/RISAT scenes here for offline-proof demos
└── SatQuery_AI_PRD_v2.md
```

## Run it right now (mock mode, zero setup)

```bash
# backend
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env                                       # defaults already work
uvicorn main:app --reload --port 8000

# frontend, in a second terminal
cd frontend
npm install
npm run dev   # opens on http://localhost:5173, proxies /api and /health to :8000
```

With `VQA_MOCK_MODE=true` (the .env.example default), every task — single-image,
change detection, fusion — returns deterministic-but-synthetic evidence, so you
can build and demo the whole product before any model is trained. Verified
working end-to-end: `cd backend && pytest tests/ -v` (7/7 passing).

## What you need to give me / do yourselves, in priority order

1. **Team details for the pitch deck** — team name and member names for the
   title slide (I used placeholders in the .pptx).
2. **Bhoonidhi/UOPS registration** — register at bhoonidhi.nrsc.gov.in and
   uops.nrsc.gov.in on day 1; approval isn't instant. Until it comes through,
   develop against `demo_data/`.
3. **A few real or public-benchmark scenes** to drop in `demo_data/` (see its
   README) — lets you sanity-check ingestion/preprocessing against real
   GeoTIFFs instead of only the mock pipeline.
4. **GPU access** (Colab Pro / Kaggle / a cloud box) if you want to actually
   fine-tune the three engines on BigEarthNet.txt / CDVQA / QAG-360K rather
   than ship mock mode to judges.
5. **An Anthropic API key** (optional) — without it, `llm/synthesis.py` falls
   back to a deterministic template answer, which is fully functional but
   less fluent. Set `ANTHROPIC_API_KEY` in `.env` to switch it on.
6. **Decide how far to push mock mode for the demo itself** — it's a legitimate
   strategy to demo end-to-end on mock outputs and show the *architecture* as
   your differentiator, while training runs in parallel; just be upfront with
   judges about which parts are live-trained vs. scaffolded.

Once real checkpoints exist, only three files change: set `VQA_MODEL_PATH`,
`CHANGE_MODEL_PATH`, `FUSION_MODEL_PATH` in `.env`, flip `VQA_MOCK_MODE=false`,
and implement each engine's `_run_real_model()` method — the router, evidence
engine, API, and frontend need zero changes.
