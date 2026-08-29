# SatQuery AI — Product Requirement Document (v2, Corrected & Cited)
### SIH26167 · Indian Space Research Organisation (ISRO) · Space Technology Track

> **What changed from v1:** v1 (built from earlier drafts) cited a dataset called "BigEarthNet" for fine-tuning and an unverified Bhoonidhi REST API schema. Both were factually wrong or unverifiable. This version replaces every unverified technical claim with a real, checked source — arXiv IDs, DOIs, or official ISRO pages — and adds a competitive-landscape section that materially changes the "novelty" pitch.

---

## 0. How to read this document

Section 1–3 answer *why this problem exists and what ISRO wants* (grounded in the PS text + published RS-VLM literature). Section 4 is the corrected architecture. Section 5 is competitive landscape — **read this before writing the novelty slide**, because two existing published systems already do pieces of what this project builds. Section 6 is data access. Section 7 is the week-by-week plan. Section 8 is the risk register. Section 9 is the full, checked reference list.

---

## 1. The problem ISRO is actually facing

Two distinct problems are bundled into one PS:

**(a) An accessibility problem.** Most remote-sensing AI today is single-task and expert-only: land-cover classifiers, object detectors, and change-detection pipelines each need their own workflow, and a non-expert cannot just *ask a question* of satellite imagery. The PS states this directly — existing systems require users to understand satellite-data characteristics, GIS workflows, model selection, and task-specific parameters before they can extract anything useful.

**(b) A completeness problem.** A single optical image often can't answer the question being asked. Change happens over time (needs a second image), and clouds block optical sensors entirely (needs SAR). ISRO's own framing in the PS is explicit: optical/multispectral imagery gives spectral and contextual information, SAR gives structural information and works day-and-night through cloud cover, and many real questions need both.

## 2. Why this problem occurs (root causes, not symptoms)

| Root cause | Evidence |
|---|---|
| **Domain gap** | General-purpose VLMs are trained on internet photos, not nadir-view satellite imagery. GeoChat's paper states plainly that general-domain VLMs perform poorly for remote-sensing scenarios, leading to inaccurate or fabricated information, because RS imagery has diverse scale changes and many small objects requiring region-level reasoning that natural-image VLMs were never trained for (Kuckreja et al., CVPR 2024). |
| **Modality gap** | Optical and SAR measure different physics (solar reflectance vs. radar backscatter), so naive feature concatenation fails. Published fusion work uses dual-branch encoders with cross-modal attention specifically because of this (see §4.3). |
| **Task fragmentation** | Single-image VQA/grounding (GeoChat, VRSBench), bi-temporal change-VQA (CDVQA/CDQAG), and optical-SAR fusion have each developed as *separate* research threads. No published system before 2026 unifies all three under one agentic router with a hidden defense-grade evaluation set. |

## 3. What ISRO wants as the solution (from the PS)

1. At least one visual/vision-language component fine-tuned or adapted on **BigEarthNet.txt** or equivalent open data.
2. **Mandatory** single-image VQA, plus either captioning or grounding.
3. **Mandatory** change description or change-VQA from a bi-temporal image pair.
4. **Mandatory** cross-modal analysis extracting complementary information from a co-registered optical + SAR pair.
5. An **agentic controller** that automatically selects, sequences, and executes the right specialist tool per query — only the *observable execution trace* (task selected, tool/model used, parameters, outputs) is graded, not internal chain-of-thought.
6. Evaluation uses public benchmark test splits **plus a hidden ISRO/SAC set** of pre-georeferenced, co-registered Cartosat-2S optical and RISAT SAR pairs.

---

## 4. Technical Architecture

### 4.1 The critical dataset correction

**"BigEarthNet.txt" is not the 2019 BigEarthNet land-cover archive.** It is a distinct, purpose-built 2026 dataset:

> Herzog, Adler, Hackel, Shu, Zavras, Papoutsis, Rota, Demir. *"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation."* arXiv:2603.29630 (2026). Hosted at https://txt.bigearth.net.

It contains 464,044 co-registered Sentinel-1 SAR + Sentinel-2 multispectral pairs with 9.6 million text annotations across exactly the three annotation types the PS mandates: geographically-anchored captions, VQA pairs, and referring-expression/grounding instructions. Built by the same BIFOLD/TU Berlin lab (Begüm Demir) that made the original BigEarthNet — the direct, intended successor for VLM instruction-tuning on the optical+SAR modality pair.

### 4.2 Full system flow

```
USER (natural-language query + 1-2 satellite images)
        |
        v
INPUT VALIDATION & INGESTION
  - detect image count / modality (optical / SAR)
  - reproject to common CRS, tile large scenes
  - reject invalid combos
        |
        v
AGENTIC ROUTER (rule-based on image_count + modality)
  1 image              -> single-image branch
  2 images, same mod    -> bi-temporal change branch
  2 images, mixed mod   -> cross-modal fusion branch
        |
        +----------------+----------------+
        v                v                v
 SINGLE-IMAGE     BI-TEMPORAL       OPTICAL-SAR
 ENGINE           CHANGE ENGINE     FUSION ENGINE
 (VLM fine-tuned  (VisTA/CDVQA-     (dual encoder +
  on BigEarthNet   trained model)    cross-attention)
  .txt)
        |                |                |
        +----------------+----------------+
                         v
        EVIDENCE ENGINE (deterministic, non-LLM)
   locks raw CV/GIS output into a fixed schema:
   {area_ha, bbox_latlon, confidence, classes, ...}
                         |
                         v
        LLM SYNTHESIS LAYER (grounded, not free)
   "Answer ONLY using the evidence object below.
    Do not invent numbers or coordinates."
                         |
                         v
   OUTPUT: text answer + map overlay + execution trace (graded!)
```

**Why the Evidence Engine sits between CV and the LLM as its own component:** the LLM is structurally prevented from computing or inventing a coordinate or an area — it can only rephrase numbers it was handed. This is also the cheapest way to satisfy the PS's execution-trace grading requirement, since the evidence object *is* the trace.

### 4.3 Optical–SAR fusion — corrected, cited

Do not treat SAR as a second RGB channel. The published pattern is a **dual-branch encoder with cross-modal attention**, motivated by cloud-cover physics: optical branch carries spectral/texture detail (blocked by clouds), SAR branch carries structural detail (cloud-penetrating); when optical cloud cover exceeds a threshold, up-weight the SAR branch. Real published precedent: a fusion network with a Sparse Feature Extraction Module combining sparse self-attention with a hybrid-scale feed-forward network, specifically to suppress SAR speckle noise while enhancing cross-modal discriminative features — a known, solved engineering pattern to implement rather than research from scratch.

### 4.4 Change detection — corrected dataset chain

- **CDVQA** (Yuan, Mou et al., IEEE TGRS, arXiv:2112.06343): 2,968 bi-temporal image pairs at 512×512, ~122,000 auto-generated QA pairs, pre-split train/val/test by geography.
- **CDQAG / QAG-360K + VisTA baseline** (Li et al., arXiv:2410.23828): over 360,000 question-answer-**mask** triplets across 10 land-cover categories — for pixel-level change masks in addition to text.
- **Change-Agent / LEVIR-MCI** (Liu et al., IEEE TGRS 2024, arXiv:2403.19646) — prior art to differentiate from (see §5).

---

## 5. Competitive Landscape

| Existing system | What it does | Gap this project fills |
|---|---|---|
| **RS-Agent** (Xu et al. 2024, arXiv:2406.07089) | Integrates RS tools + retrieval-augmented knowledge for professional RS questions | No mandatory optical-SAR fusion; no hidden defense-grade eval set |
| **Change-Agent** (Liu et al. 2024, arXiv:2403.19646) | Multi-level change interpretation (pixel masks + captions) via an MCI model + LLM "brain" | Only bi-temporal change — no single-image VQA/grounding, no SAR fusion |
| **ThinkGeo** (Shabbir et al. 2025, arXiv:2505.23752) | Evaluation benchmark for tool-augmented RS agents, includes optical+SAR queries | A benchmark, not a deployable product with a UI and evidence-grounded output |
| **GeoLLM-QA** (Singh, Fore, Stamoulis — Microsoft, arXiv:2405.00709) | Benchmark for agents handling real UI-grounded, click-based RS tasks | Also a benchmark, not a shippable assistant |
| **RS-ChatGPT / GeoGPT / GeoAgent** | Connect ChatGPT-style models to RS visual tools for task planning | General-purpose tool chaining; none mandates the specific single+bitemporal+cross-modal trio under one router |
| **GeoChat** (Kuckreja et al., CVPR 2024) | Grounded conversational VLM for single RS images (captioning, VQA, grounding, referring detection) | Single-image only — no temporal or cross-modal capability at all |

**USP #1:** the first system in this list required to unify all three task families — single-image, bi-temporal, and cross-modal optical-SAR — under one agentic router, validated against a real ISRO Cartosat-2S/RISAT hidden evaluation set rather than an open academic benchmark.

**USP #2 — hallucination-proofing as a stated contribution:** RS-specific hallucination is a genuinely new, active research area — RSHallu (arXiv:2602.10799, 2026) proposes the first dual-mode hallucination evaluation for RS multimodal LLMs with domain-tailored mitigation. Architecturally separating deterministic CV/GIS computation from LLM text generation (§4.2's Evidence Engine) is exactly the kind of mitigation this literature calls for.

---

## 6. Data Access

- **Bhoonidhi** (https://bhoonidhi.nrsc.gov.in) is ISRO's real Earth Observation data hub, covering 47 satellites since 1986, with an API now released (contact bhoonidhi@nrsc.gov.in for access).
- A real, working **community Python client** exists: `pip install bhoonidhi`, providing an NLP-driven "smart search" plus a download function requiring login credentials from Bhoonidhi or registration at `uops.nrsc.gov.in`.
- Register for Bhoonidhi/UOPS access on day 1 — approval isn't instant — and in parallel cache 15–20 demo scenes locally so the live demo never depends on the portal being reachable.

**Satellite specs for the demo narrative:** Cartosat-2S: 0.65 m panchromatic, 2.0 m multispectral, 9.6 km swath. RISAT-1A SAR: C-band, 5.35 GHz, five imaging modes spanning 1–50 m resolution and 30–240 km swath.

---

## 7. Week-by-Week Execution Plan

```
WEEK 1 — Foundation                          WEEK 2 — Model Training
Day 1-2: Register Bhoonidhi/UOPS access      Day 8-9: Fine-tune VLM on BigEarthNet.txt
Day 3-4: Download BigEarthNet.txt/CDVQA/     Day 10-11: Train/adapt VisTA on CDVQA +
         QAG-360K subsets                              QAG-360K for change-VQA
Day 5-6: Build GeoTIFF ingestion             Day 12-14: Build optical-SAR dual-encoder
         (reprojection, tiling, cloud mask)             + cross-attention fusion module
Day 7: Cache 15-20 offline demo scenes       Checkpoint: each engine hits a minimum
                                                        working bar before Week 3

WEEK 3 — Agentic Orchestration               WEEK 4 — UI, Trace, Demo Polish
Day 15-17: Build router (rule-based on       Day 22-24: React/Leaflet dashboard —
           image_count + modality)                     upload, split-screen viewer, chat
Day 18-19: Build Evidence Engine             Day 25-26: Execution-trace JSON viewer
Day 20-21: Wire LLM synthesis layer with     Day 27-28: Rehearse all demo scenarios
           evidence-only prompting +                    end-to-end, under 90s each
           adversarial "invent a coordinate" tests      Final: dry run with countdown
```

---

## 8. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Fine-tuning diverges under time pressure | Medium | QLoRA (4-bit), small LR, validate every 100 steps; frozen zero-shot GeoChat/Qwen2-VL fallback as Plan B |
| Bhoonidhi API/portal unreachable during live judging | Medium | Pre-cache 15–20 demo scenes locally — never depend on a live fetch during the demo |
| Cross-modal misregistration (optical/SAR not aligned) | High | Reproject both to a common CRS at ingestion, not inference |
| LLM invents a coordinate or area figure | High if unmitigated | Evidence Engine architecture structurally prevents this — test with adversarial prompts |
| Judge asks "how is this different from Change-Agent/RS-Agent?" | Certain | §5's table — answer with the specific gap each one leaves |

---

## 9. Full Reference List

| # | Citation | Link/ID |
|---|---|---|
| 1 | Herzog, Adler, Hackel, Shu, Zavras, Papoutsis, Rota, Demir. "BigEarthNet.txt." 2026. | arXiv:2603.29630 · txt.bigearth.net |
| 2 | Sumbul, Charfuelan, Demir, Markl. "BigEarthNet." IGARSS 2019. | arXiv:1902.06148 |
| 3 | Sumbul et al. "BigEarthNet-MM." IEEE GRSM, 2021. | DOI 10.1109/MGRS.2021.3089174 |
| 4 | Lobry, Marcos, Murray, Tuia. "RSVQA." IEEE TGRS 58(12), 2020. | DOI 10.1109/TGRS.2020.2988782 |
| 5 | Li, Ding, Elhoseiny. "VRSBench." NeurIPS 2024. | arXiv:2406.12384 |
| 6 | Yuan, Mou et al. "CDVQA." IEEE TGRS, 2022. | arXiv:2112.06343 |
| 7 | Li et al. "CDQAG / QAG-360K / VisTA." 2024. | arXiv:2410.23828 |
| 8 | Liu et al. "Change-Agent." IEEE TGRS, 2024. | arXiv:2403.19646 |
| 9 | Kuckreja, Danish, Naseer, Das, Khan, Khan. "GeoChat." CVPR 2024. | openaccess.thecvf.com |
| 10 | Xu et al. "RS-Agent." 2024. | arXiv:2406.07089 |
| 11 | Shabbir et al. "ThinkGeo." 2025. | arXiv:2505.23752 |
| 12 | Singh, Fore, Stamoulis (Microsoft). "GeoLLM-QA." 2024. | arXiv:2405.00709 |
| 13 | Wang et al. "Qwen2-VL." 2024. | arXiv:2409.12191 |
| 14 | Zhou et al. "RSHallu." 2026. | arXiv:2602.10799 |
| 15 | ISRO Bhoonidhi Earth Observation Data Hub | bhoonidhi.nrsc.gov.in |
| 16 | Community `bhoonidhi` Python client | pypi.org/project/bhoonidhi |
| 17 | Cartosat-2S specifications | ISRO / NRSC published specs |
| 18 | RISAT-1A (EOS-04) SAR specifications | eoportal.org/satellite-missions/risat-1 |
