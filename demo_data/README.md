# demo_data/

Cache 15-20 real (or realistic) scenes here so the live demo never depends on
a reachable Bhoonidhi portal (PRD Risk Register: "Bhoonidhi API/portal
unreachable during live judging" is a Medium-likelihood risk).

Suggested layout:

```
demo_data/
├── single_image/        # 1 optical scene each, for VQA/caption/grounding demo
├── change_pairs/        # 2 same-modality scenes, two dates, per subfolder
└── fusion_pairs/         # 1 optical + 1 SAR, co-registered, per subfolder
```

Name files so `ingestion/preprocessing.detect_modality()`'s filename
heuristic picks up the right modality automatically, e.g.:
- `cartosat2s_optical_2024-01-15.tif`
- `risat_sar_2024-01-15.tif`

Get real scenes via:
1. Bhoonidhi smart search (`backend/data_access/bhoonidhi_client.py`) once
   BHOONIDHI_USER/PASSWORD are set, or
2. Public benchmark samples from BigEarthNet.txt / CDVQA / QAG-360K for
   development, while you wait on Bhoonidhi/UOPS approval.
