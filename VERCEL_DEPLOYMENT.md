# Production Deployment

## Production URL
https://satellitewildfireproject.vercel.app

**Vercel Inspector:** https://vercel.com/sahil-da66/satellite_wildfire_project/Em8AgQDPnCErF3sFFtNaexcvPg4E  
**Deployment ID:** `dpl_Em8AgQDPnCErF3sFFtNaexcvPg4E`  
**State:** `READY` (Production)

---

## Architecture
- **Frontend:** Single-Page Vanilla JS / CSS Glassmorphic Command Console (`public/index.html`, `public/static/css/style.css`, `public/static/js/app.js`). Zero external CDN framework lock-in, zero localhost calls, fully responsive dual-viewport satellite analysis dashboard.
- **Backend:** Vercel Serverless Edge Python Runtime (`api/index.py` on Python 3.12 via FastAPI / ASGI Serverless Adapter). Handles telemetry (`/api/status`), mission presets (`/api/presets`), analysis dispatch (`/api/analyze/preset`, `/api/analyze/upload`), AI Situation Report compilation (`/api/sitrep/markdown`), and conversational tactical advisory (`/api/chat`).
- **ML Inference:**
  - **Serverless Tier:** Production Serverless CPU Engine hosting pre-calibrated multi-agent benchmark telemetry, Copernicus EMS vector grading, and USGS dNBR ground truth validation.
  - **Deep Worker Tier:** External ML Worker via `MODEL_API_URL` and `MODEL_API_KEY`. When configured, the Vercel API seamlessly forwards full-resolution multi-modal rasters (GeoTIFF/NPZ) to dedicated RTX/CUDA hardware for live ResNet-34 U-Net inference without exposing the worker's address to the browser client.
- **Storage:** Ephemeral memory-buffered stream validation on Vercel; client-side canvas rasterization; pre-calibrated preview assets in `api/presets_previews.json`.
- **LLM:** Hybrid Intelligence Layer. If `GEMINI_API_KEY` is provided in Vercel Environment Variables, the platform uses Gemini 1.5 Flash grounded strictly in structured incident telemetry. If absent, a deterministic tactical rule-based advisor answers inquiries using the Multi-Agent SITREP data.

---

## Environment Variables
The following variables are supported and should be configured via the Vercel Dashboard (`Settings -> Environment Variables`) as needed:

- `MODEL_API_URL` — (Optional) HTTPS endpoint of your dedicated GPU inference worker (e.g., RunPod, Modal, or local tunnel).
- `MODEL_API_KEY` — (Optional) Bearer token for authenticating calls from Vercel to your ML worker.
- `GEMINI_API_KEY` — (Optional) Google Gemini API key for advanced conversational tactical reasoning.
- `NEXT_PUBLIC_API_URL` — (Optional) Custom API gateway domain if proxying through a custom subdomain.

*Note: None of these are committed to version control. Reference template is available in `.env.example`.*

---

## Deployment Commands
The project was deployed using the Vercel CLI with automated asset building:

```powershell
# 1. Build and verify static production assets
node scripts/build.js

# 2. Deploy directly to Vercel Production
npx vercel --prod --yes
```

---

## Local Development
To run the full stack locally with direct access to local NVIDIA GPU checkpoints (`models/*.pt`):

```powershell
# Activate local conda/venv environment
conda activate satai-train

# Launch local FastAPI server with hot-reload
python app/server.py
# Or:
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```
Local dashboard is served at `http://127.0.0.1:8000/`.

---

## Production Testing
Post-deployment verification was executed against the live production URL `https://satellitewildfireproject.vercel.app` using `scripts/test_vercel_live.py`:

| Test Target | Endpoint / Action | Result | Details |
| :--- | :--- | :--- | :--- |
| **Homepage** | `GET /` | **PASS (HTTP 200)** | 37.4 KB HTML loaded with complete styling and telemetry bar. |
| **Telemetry** | `GET /api/status` | **PASS (HTTP 200)** | Status: `OPERATIONAL`, Runtime: `Vercel Serverless Edge`, 5 models registered. |
| **Presets** | `GET /api/presets` | **PASS (HTTP 200)** | 3 calibrated incidents returned (Palisades, Encinedo, Heavy Smoke). |
| **Preset Analysis**| `POST /api/analyze/preset` | **PASS (HTTP 200)** | Pacific Palisades analyzed: 107.25 km² burned area, Threat Level: CRITICAL / LEVEL 4. |
| **Tactical Chatbot** | `POST /api/chat` | **PASS (HTTP 200)** | Returned structured situational intelligence (`DelineationAgent`, `SeverityQuantifierAgent`). |
| **Upload Validation**| `POST /api/analyze/upload` | **PASS (HTTP 400)** | Malicious/unsupported input rejected with `"Unsupported image format or satellite data structure."` |

---

## Known Limitations
1. **Direct GPU Execution on Vercel:** Vercel functions run in lightweight CPU-only serverless containers with a 250 MB package size limit. The 8 trained PyTorch checkpoints (1.32 GB total) cannot be executed inside a Vercel serverless function without an external ML worker (`MODEL_API_URL`).
2. **Preset vs. Upload Behavior:** Preset incidents serve verified benchmark telemetry with authentic USGS dNBR and Copernicus EMS ground truth. User uploads without georeferencing display `"Geospatial area estimate unavailable"`, and without ground truth display `"Ground Truth: Not Available | Validation Status: Unverified"` in strict adherence to scientific honesty.
3. **Serverless Payload Limit:** Direct multipart uploads through Vercel serverless functions have a 4.5 MB request limit. High-resolution multi-gigabyte raster files should be processed via direct-to-object-storage or via the local workstation GPU pipeline.

---

## ML Inference
- Trained PyTorch models (`best_s2_baseline_model.pt`, `best_s1_baseline_model.pt`, `best_fusion_model.pt`, ablation models) run on local or dedicated GPU infrastructure (NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.4).
- The production deployment on Vercel acts as a secure, high-availability Edge Gateway. When `MODEL_API_URL` is configured, Vercel securely dispatches inference requests to the GPU worker. When unconfigured, it utilizes the pre-calculated, verified scientific benchmark telemetry for all mission presets.

---

## Dataset
- The complete multi-sensor raw dataset (~20+ GB of Sentinel-1 SLC/GRD and Sentinel-2 Multi-Spectral rasters, along with training patches) remains strictly local and excluded via `.gitignore` and `.vercelignore`.
- Only compiled visual representations, calibrated confusion matrices, and benchmark ground-truth geometries are packaged for the production application.
