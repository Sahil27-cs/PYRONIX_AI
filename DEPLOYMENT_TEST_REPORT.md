# Deployment Test Report — Production Pre-Flight Audit
**System:** AI-Powered Satellite Wildfire & Burned-Area Intelligence Platform (Pyronix AI)  
**Date:** October 2026  
**Auditor:** Autonomous Lead DevOps & ML Engineer  

---

## 1. Build Verification
- **Command:** `npm run build` (`node scripts/build.js`)
- **Status:** **PASS** (Exit Code 0)
- **Artifacts Generated:**
  - `public/index.html` (Production single-page interface with updated title & headers)
  - `public/static/css/style.css` (Aerospace HUD styling system)
  - `public/static/js/app.js` (Client-side reactive controller)
- **Serverless API Files:**
  - `api/index.py` (FastAPI production gateway)
  - `api/requirements.txt` (Pinned lightweight dependencies)
  - `api/presets_data.json` & `api/presets_previews.json` (Calibrated benchmarks)

---

## 2. Frontend Routes & Visual Checks
| Route | Method | Expected Output | Status |
|:---|:---:|:---|:---:|
| `/` | `GET` | HTML5 Single-Page Dashboard (`public/index.html`) | **PASS (200 OK)** |
| `/static/css/style.css` | `GET` | Production Aerospace HUD CSS (Glassmorphism) | **PASS (200 OK)** |
| `/static/js/app.js` | `GET` | Reactive Client Application Controller | **PASS (200 OK)** |

---

## 3. Serverless API Endpoint Tests
| Endpoint | Method | Input / Test Case | Result | Status |
|:---|:---:|:---|:---|:---:|
| `/api/status` | `GET` | Health & Accelerator check | `200 OK` (Operational, Vercel Serverless Edge) | **PASS** |
| `/api/presets` | `GET` | Preset catalog retrieval | `200 OK` (3 presets with ground-truth metadata) | **PASS** |
| `/api/analyze/preset` | `POST` | `preset_id: palisades` | `200 OK` (107.25 km², Level 4, USGS dNBR) | **PASS** |
| `/api/analyze/preset` | `POST` | `preset_id: copernicus_encinedo` | `200 OK` (6.5 km², CEMS damage vector) | **PASS** |
| `/api/chat` | `POST` | Query: *"What is the burned area?"* | `200 OK` (Accurate area from SitRep) | **PASS** |
| `/api/sitrep/markdown` | `GET` | SitRep markdown export | `200 OK` (2,999 bytes Markdown SitRep) | **PASS** |

---

## 4. File Upload & Scientific Data Structure Validation
| Test Case | Input | Expected Result | Actual Result | Status |
|:---|:---|:---|:---|:---:|
| **Unsupported Extension** | Plain text file (`test.txt`) | HTTP 400 Error | `400 Bad Request` ("Unsupported image format or satellite data structure.") | **PASS** |
| **Missing Optical Bands** | 1-channel or invalid shape raster | HTTP 400 Error | `400 Bad Request` ("Invalid satellite raster: Missing required spectral bands.") | **PASS** |
| **Calibrated Patch Ingest** | Multi-band NumPy array (`.npz`) | Delineation + SitRep | `200 OK` (Delineated burn scar, Ground truth unverified) | **PASS** |
| **Geospatial Resolution Missing** | Unreferenced image | No fabricated area | `affected_area_display: "Geospatial area estimate unavailable"` | **PASS** |

---

## 5. Chatbot Reasoning & LLM Tests
- **Intent Routing:** Tested queries for *burned area*, *debris flow hazard*, *sensor arbitration*, and *tactical prescriptions*.
- **Factual Grounding:** All responses grounded in active SitRep without fabricating metrics.
- **Graceful Fallback:** Tested when `GEMINI_API_KEY` is not set; system falls back to rule-based tactical intent routing without throwing exceptions or crashing.

---

## 6. Known Limitations
1. **Model Weights Size on Vercel Edge:** The 8 trained PyTorch model checkpoints total ~1.32 GB and cannot be bundled inside a single Vercel Serverless Function (250 MB AWS Lambda uncompressed limit). Full deep neural network inference on raw 16-band GeoTIFFs runs locally or offloads securely via `MODEL_API_URL`.
2. **Upload Size Limit:** Direct Vercel serverless HTTP requests are limited to 4.5 MB. Full multi-gigabyte satellite scenes must be processed locally or via dedicated direct object storage.
3. **Ground Truth Disclosure:** For arbitrary user-uploaded imagery, ground-truth labels do not exist in the wild; the system honestly marks them as `Ground Truth: Not Available | Validation Status: Unverified`.
