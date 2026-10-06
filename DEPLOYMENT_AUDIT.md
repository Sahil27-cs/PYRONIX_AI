# Deployment Audit & Architectural Assessment
**System:** AI-Powered Satellite Wildfire & Burned-Area Intelligence Platform (Pyronix AI)  
**Date:** October 2026  
**Auditor:** Autonomous Lead DevOps & ML Deployment Engineer  

---

## 1. Current Architecture Summary
The existing application is a multi-tier system combining an asynchronous Python web server, an autonomous multi-agent reasoning hierarchy, deep semantic segmentation models (ResNet-34 U-Net), and a custom aerospace-themed single-page frontend.

- **Current Serving Mode (Localhost):**
  Served via Uvicorn (`python -m uvicorn app.server:app --host 127.0.0.1 --port 8000`), hosting FastAPI endpoints and mounting `app/static/` for CSS/JS assets and serving `app/templates/index.html`.
- **Frontend Framework:** Vanilla HTML5, CSS3 (Aerospace Tactical HUD design system, glassmorphism, responsive grid), and client-side reactive JavaScript (`app/static/js/app.js`).
  - No bundler or Node build step currently exists (`package.json` not present).
  - High performance, zero runtime JavaScript framework overhead.
- **Backend Framework:** FastAPI (Python 3.11/3.12).
- **Inference Engine:** PyTorch 2.6.0 with CUDA 12.4 acceleration (RTX 4050 Laptop GPU locally, CPU fallback supported).
- **Multi-Agent Hierarchy (`src/agents/`):**
  - `SensorArbitratorAgent`: Evaluates optical cloud/smoke cover and SAR SNR, selects model.
  - `DelineationAgent`: Executes deep ResNet-34 U-Net sliding window inference, probability mapping, and Shannon entropy uncertainty calculation.
  - `SeverityQuantifierAgent`: Calculates Copernicus EMS damage tiers (Grades 1–3), perimeter length (km), and shape compactness.
  - `RiskAssessmentAgent`: Quantifies debris flow hazard and BAER prescriptions.
  - `WildfireOrchestrator`: Master orchestrator generating JSON and Markdown Situation Reports (SitReps).

---

## 2. Model & Inference Inventory
| Model Name | Checkpoint File | In Channels | Input Modality | File Size | Parameters |
|:---|:---|:---:|:---|:---:|:---:|
| **Sentinel-2 Baseline** | `best_s2_baseline_model.pt` | 6 | Optical ($B2, B3, B4, B8, B11, B12$) | 293.6 MB | 24,445,649 |
| **Sentinel-1 Baseline** | `best_s1_baseline_model.pt` | 3 | SAR Radar ($VV, VH, \text{ratio}$) | 293.5 MB | 24,436,241 |
| **Multimodal Fusion** | `best_fusion_model.pt` | 9 | Optical (6) + SAR (3) | 293.8 MB | 24,455,057 |
| **Ablation S1 VV** | `ablation_S1_VV_only.pt` | 1 | SAR $VV$ | 97.9 MB | 24,429,969 |
| **Ablation S1 VH** | `ablation_S1_VH_only.pt` | 1 | SAR $VH$ | 97.9 MB | 24,429,969 |
| **Ablation S1 VV+VH** | `ablation_S1_VV_VH.pt` | 2 | SAR $VV+VH$ | 97.9 MB | 24,433,105 |
| **Ablation S2 RGB** | `ablation_S2_RGB_only.pt` | 3 | Optical $B2, B3, B4$ | 97.9 MB | 24,436,241 |
| **Ablation S2 NIR+SWIR** | `ablation_S2_NIR_SWIR.pt` | 3 | Optical $B8, B11, B12$ | 97.9 MB | 24,436,241 |
| **Total Model Storage** | — | — | — | **~1.32 GB** | — |

---

## 3. Vercel Deployment Blockers for Monolithic Upload
1. **Serverless Function Size Limit (250 MB Uncompressed):**
   - Individual model weights range from 97.9 MB to 293.8 MB each. A single baseline checkpoint exceeds the 250 MB total package limit for Vercel functions.
   - Bundling PyTorch (`torch` + `torchvision` on Linux) alone requires ~700 MB–1 GB uncompressed.
2. **Native C-Extensions & GeoTIFF Handling:**
   - `rasterio` and `gdal` require compiled C/C++ shared libraries that are incompatible with standard lightweight Vercel serverless containers without extensive custom Lambda layers.
3. **Request Body Size Limit (4.5 MB):**
   - Full satellite scenes or multi-band GeoTIFF files can range from 10 MB to several gigabytes. Vercel serverless functions reject incoming payloads larger than 4.5 MB with an `HTTP 413 Payload Too Large` error.
4. **Execution Timeout (10–15 Seconds):**
   - CPU-only sliding window inference across a full $10,000 \times 10,000$ satellite scene exceeds the standard Vercel 10-second timeout.
5. **Git LFS / Repository Limits:**
   - Git and GitHub reject files larger than 100 MB without Git LFS. Pushing 1.3 GB of `.pt` files directly into a Vercel Git deployment will fail.

---

## 4. Files that Must NOT be Deployed to Vercel
- `data/` (All raw, extracted, and processed patches: several GBs).
- `models/` (All 8 `.pt` checkpoint weights: 1.32 GB).
- `models/checkpoints/` (Epoch resume checkpoints).
- `notebooks/` and scratch scripts.
- Virtual environments (`.venv`, `env`, conda environments).
- Inspection caches and raw TIFF rasters.

---

## 5. Required Environment Variables
| Variable Name | Required / Optional | Purpose |
|:---|:---:|:---|
| `MODEL_API_URL` | Optional | External ML inference endpoint for offloading deep GPU/CPU PyTorch inference. |
| `MODEL_API_KEY` | Optional | Private authentication token for the external ML inference service. |
| `GEMINI_API_KEY` | Optional | LLM API key for the conversational tactical chatbot (when AI natural language reasoning is enabled). |
| `NEXT_PUBLIC_API_URL` | Optional | Base URL if frontend needs explicit override (defaults to relative `/api`). |

---

## 6. Recommended Production Architecture on Vercel
To achieve 100% genuine scientific accuracy without fabricating results and without breaking Vercel resource boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER BROWSER / CLIENT                   │
│      (HTML5 / Vanilla CSS Aerospace HUD / JavaScript)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
            HTTPS Requests (/api/..., static assets)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  VERCEL PRODUCTION PLATFORM                 │
│                                                             │
│  ├── Static Edge Network:                                   │
│  │   • app/templates/index.html                             │
│  │   • app/static/css/style.css                             │
│  │   • app/static/js/app.js                                 │
│  │                                                          │
│  ├── Vercel Serverless Functions (Python / Node /api/):     │
│  │   • /api/status     (System telemetry & connectivity)    │
│  │   • /api/presets    (Calibrated operational presets)     │
│  │   • /api/analyze    (Proxies to MODEL_API_URL or runs    │
│  │   │                  serverless raster analysis)         │
│  │   • /api/chat       (Tactical assistant & SitRep)        │
│  │   • /api/sitrep     (Markdown & JSON SitRep exports)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
            Secure API Proxy (MODEL_API_URL + Key)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              EXTERNAL ML INFERENCE SERVICE (Optional)       │
│    (Self-hosted FastAPI / Modal / RunPod / Dedicated GPU)   │
│   • Runs full 1.3GB ResNet-34 U-Net checkpoints on PyTorch  │
│   • Full multi-spectral GeoTIFF & SAR radar inference       │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Guarantees:
1. **Zero Fabrication:** If `MODEL_API_URL` is configured, Vercel securely routes live inference to the dedicated PyTorch pipeline. If unconfigured or unavailable, the system transparently indicates external inference status and serves validated mission benchmarks (e.g. Pacific Palisades, Copernicus Encinedo) with authentic ground-truth metrics.
2. **Ground-Truth Truthfulness:** When ground truth is unavailable on arbitrary uploads, the system explicitly marks validation as `Ground Truth: Not Available | Validation Status: Unverified` rather than fabricating IoU or Dice scores.
3. **Geospatial Integrity:** Affected area in $\text{km}^2$ is only reported when valid geospatial spatial resolution ($10\text{ m/pixel}$) is present; otherwise reported as `Geospatial area estimate unavailable`.
4. **Vercel Build Command:** A standard Node build (`npm run build`) or clean static export that produces a production-ready output in `<project-root>/public` or serves directly via `vercel.json` rewrites.
