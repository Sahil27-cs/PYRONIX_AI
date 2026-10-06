"""
FastAPI Backend Server for Pyronix AI Wildfire Intelligence Application
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System
"""

import os
import sys
import io
import time
import json
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import WildfireOrchestrator

app = FastAPI(
    title="Pyronix AI — Satellite Wildfire Intelligence",
    description="Autonomous Multi-Agent Satellite Wildfire Delineation, Damage Assessment & Tactical Response System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# App directory paths
app_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_dir, "static")
templates_dir = os.path.join(app_dir, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Models and data paths
models_dir = os.path.join(project_root, "models")
stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
s1_stats_json = os.path.join(project_root, "data", "inspection", "s1_statistics.json")
reports_dir = os.path.join(project_root, "results", "reports")
palisades_dir = os.path.join(project_root, "data", "extracted", "palisades")
zenodo_test_dir = os.path.join(project_root, "data", "processed_fusion", "test")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Master multi-agent orchestrator instance
orchestrator = WildfireOrchestrator(
    models_dir=models_dir,
    stats_json=stats_json,
    s1_stats_json=s1_stats_json,
    device=device
)

# Active incident state for real-time chatbot interaction
active_incident_state = {
    "incident_name": "Pacific Palisades Wildfire",
    "sitrep": None,
    "context": None,
    "last_analyzed": None
}

def array_to_base64_png(arr: np.ndarray, cmap: str = None, vmin=None, vmax=None) -> str:
    """Converts 2D or 3D NumPy array into a base64 encoded PNG string."""
    buf = io.BytesIO()
    if arr.ndim == 2:
        if cmap is None:
            cmap = "gray"
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    elif arr.ndim == 3:
        # Check if C, H, W or H, W, C
        if arr.shape[0] in [1, 3, 4] and arr.shape[0] < arr.shape[1]:
            # Channel first -> Convert to H, W, C
            if arr.shape[0] >= 3:
                rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
            else:
                rgb = arr[0]
        else:
            rgb = arr
        # Normalize to [0, 255] uint8
        rgb_norm = np.clip(rgb / (np.percentile(rgb, 98) + 1e-6), 0.0, 1.0)
        img_uint8 = (rgb_norm * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Pyronix AI System: templates/index.html is being prepared...</h1>"

@app.get("/api/status")
async def get_system_status():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0

    models_list = [f for f in os.listdir(models_dir) if f.endswith(".pt")] if os.path.exists(models_dir) else []

    return {
        "status": "OPERATIONAL",
        "system": "Pyronix AI Wildfire Multi-Agent Platform",
        "device": str(device),
        "device_name": device_name,
        "vram_gb": vram_gb,
        "cuda_available": cuda_avail,
        "models_available": models_list,
        "total_models": len(models_list),
        "active_incident": active_incident_state["incident_name"],
        "has_analyzed_incident": active_incident_state["sitrep"] is not None
    }

@app.get("/api/presets")
async def get_incident_presets():
    return [
        {
            "id": "palisades",
            "name": "Pacific Palisades Wildfire (California, USA)",
            "description": "May 2021 Southern California wildfire across Topanga Canyon & Pacific Palisades WUI.",
            "area_km2": 235.9,
            "ground_truth": "USGS Differenced Normalized Burn Ratio (dNBR)",
            "sensors": ["Sentinel-2 Optical (RGB+NIR)", "Sentinel-1A SAR (Dual-Pol VV+VH)"]
        },
        {
            "id": "copernicus_encinedo",
            "name": "Copernicus EMS Encinedo Fire (Spain)",
            "description": "Copernicus EMS EMSR227 Rapid Mapping activation in Mediterranean pine forest.",
            "area_km2": 6.5,
            "ground_truth": "CEMS Grading Map Damage Vector",
            "sensors": ["Sentinel-2 Optical (6 Bands)", "Sentinel-1 SAR (3 Bands)"]
        },
        {
            "id": "smoke_occluded",
            "name": "Heavy Smoke & Cloud Occluded Wildfire (SAR Penetration)",
            "description": "Synthetic multi-sensor scene simulating 95% cloud and dense pyrocumulus smoke occlusion.",
            "area_km2": 6.5,
            "ground_truth": "Sub-canopy Burned Perimeter",
            "sensors": ["Sentinel-1 SAR (All-Weather Radar Delineation)"]
        }
    ]

@app.post("/api/analyze/preset")
async def analyze_preset(preset_id: str = Form(...), threshold: float = Form(0.5)):
    global active_incident_state

    if preset_id == "palisades":
        # Load Palisades Scene from data/extracted/palisades
        if not os.path.exists(palisades_dir):
            raise HTTPException(status_code=404, detail="Palisades data not found on disk.")

        from scripts.run_multiagent_analysis import load_palisades_scene
        ctx = load_palisades_scene(palisades_dir)
        ctx["threshold"] = threshold

    elif preset_id == "copernicus_encinedo":
        if not os.path.exists(zenodo_test_dir):
            raise HTTPException(status_code=404, detail="Zenodo test data not found.")

        from scripts.run_multiagent_analysis import load_zenodo_test_sample
        ctx = load_zenodo_test_sample(zenodo_test_dir)
        ctx["threshold"] = threshold

    elif preset_id == "smoke_occluded":
        # Generate smoke-occluded scenario from Zenodo test patch
        from scripts.run_multiagent_analysis import load_zenodo_test_sample
        ctx = load_zenodo_test_sample(zenodo_test_dir)
        ctx["incident_name"] = "Smoke Occluded Wildfire (SAR Bypass Demonstration)"
        # Inject severe cloud/smoke mask
        C, H, W = ctx["optical_data"].shape
        ctx["cloud_mask"] = np.ones((H, W), dtype=np.uint8) * 255
        ctx["threshold"] = threshold
    else:
        raise HTTPException(status_code=400, detail=f"Unknown preset ID: {preset_id}")

    # Run multi-agent orchestrator
    sitrep = orchestrator.run_analysis(ctx, output_dir=reports_dir)

    active_incident_state["incident_name"] = ctx["incident_name"]
    active_incident_state["sitrep"] = sitrep
    active_incident_state["context"] = ctx
    active_incident_state["last_analyzed"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate image previews
    optical = ctx.get("optical_data")
    sar = ctx.get("sar_data")
    dnbr = ctx.get("dnbr_data")
    delin = ctx["delineation_results"]
    sev = ctx["severity_results"]

    previews = {
        "probability_map": array_to_base64_png(delin["probability_map"], cmap="inferno", vmin=0.0, vmax=1.0),
        "binary_mask": array_to_base64_png(delin["binary_mask"], cmap="Reds", vmin=0, vmax=1),
        "uncertainty_map": array_to_base64_png(delin["uncertainty_map"], cmap="viridis", vmin=0, vmax=1),
    }

    if optical is not None and optical.shape[0] >= 3:
        # True color: B4=2, B3=1, B2=0
        rgb = np.stack([optical[2], optical[1], optical[0]], axis=-1)
        previews["optical_rgb"] = array_to_base64_png(rgb)
        if optical.shape[0] >= 4:
            # False color: NIR=3, Red=2, Green=1
            fcolor = np.stack([optical[3], optical[2], optical[1]], axis=-1)
            previews["optical_fcolor"] = array_to_base64_png(fcolor)

    if sar is not None and sar.shape[0] >= 2:
        vh = sar[1]
        previews["sar_vh"] = array_to_base64_png(vh, cmap="gray")

    if dnbr is not None:
        previews["dnbr"] = array_to_base64_png(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=0.8)

    if "severity_map" in sev:
        from matplotlib.colors import ListedColormap
        sev_cmap = ListedColormap(["#1a1a24", "#fed976", "#fd8d3c", "#bd0026"])
        previews["severity_map"] = array_to_base64_png(sev["severity_map"], cmap=sev_cmap, vmin=0, vmax=3)

    return {
        "status": "SUCCESS",
        "incident_name": ctx["incident_name"],
        "sitrep": orchestrator._make_serializable(sitrep),
        "previews": previews,
        "threat_level": sitrep["threat_level"],
        "burned_area_km2": sitrep["burned_area_km2"],
        "burned_area_acres": sitrep["burned_area_acres"],
        "perimeter_km": sitrep["perimeter_km"],
        "arbitration_mode": sitrep["arbitration"]["mode"],
        "recommended_model": sitrep["arbitration"]["recommended_model"]
    }

@app.post("/api/analyze/upload")
async def analyze_uploaded_file(
    optical_file: UploadFile = File(None),
    sar_file: UploadFile = File(None),
    incident_name: str = Form("User Uploaded Wildfire"),
    threshold: float = Form(0.5)
):
    global active_incident_state

    if not optical_file and not sar_file:
        raise HTTPException(status_code=400, detail="Please upload at least one satellite image file (Optical or SAR).")

    optical_data = None
    sar_data = None

    # Helper to parse uploaded file
    def parse_file(upload: UploadFile):
        content = upload.file.read()
        fname = upload.filename.lower()
        if fname.endswith(".npz"):
            with np.load(io.BytesIO(content)) as d:
                # Check for 'image' or first array
                key = "image" if "image" in d else list(d.keys())[0]
                return d[key].astype(np.float32)
        elif fname.endswith(".npy"):
            return np.load(io.BytesIO(content)).astype(np.float32)
        elif fname.endswith((".tif", ".tiff")):
            import rasterio
            with rasterio.open(io.BytesIO(content)) as src:
                raw = src.read().astype(np.float32)
                return raw
        elif fname.endswith((".png", ".jpg", ".jpeg")):
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            # Convert H, W, C to C, H, W (RGB) -> Invert order to B, G, R for Sentinel format
            arr_c = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=0)
            return arr_c
        else:
            raise ValueError(f"Unsupported file format: {fname}")

    try:
        if optical_file:
            optical_data = parse_file(optical_file)
        if sar_file:
            sar_data = parse_file(sar_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded satellite image: {str(e)}")

    ctx = {
        "incident_name": incident_name,
        "optical_data": optical_data,
        "sar_data": sar_data,
        "threshold": threshold,
        "metadata": {
            "source": "User Web Upload",
            "optical_filename": optical_file.filename if optical_file else None,
            "sar_filename": sar_file.filename if sar_file else None
        }
    }

    sitrep = orchestrator.run_analysis(ctx, output_dir=reports_dir)

    active_incident_state["incident_name"] = incident_name
    active_incident_state["sitrep"] = sitrep
    active_incident_state["context"] = ctx
    active_incident_state["last_analyzed"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate previews
    delin = ctx["delineation_results"]
    sev = ctx["severity_results"]

    previews = {
        "probability_map": array_to_base64_png(delin["probability_map"], cmap="inferno", vmin=0.0, vmax=1.0),
        "binary_mask": array_to_base64_png(delin["binary_mask"], cmap="Reds", vmin=0, vmax=1),
        "uncertainty_map": array_to_base64_png(delin["uncertainty_map"], cmap="viridis", vmin=0, vmax=1),
    }

    if optical_data is not None and optical_data.shape[0] >= 3:
        rgb = np.stack([optical_data[2], optical_data[1], optical_data[0]], axis=-1)
        previews["optical_rgb"] = array_to_base64_png(rgb)

    if sar_data is not None:
        sar_preview = sar_data[1] if sar_data.shape[0] >= 2 else sar_data[0]
        previews["sar_vh"] = array_to_base64_png(sar_preview, cmap="gray")

    if "severity_map" in sev:
        from matplotlib.colors import ListedColormap
        sev_cmap = ListedColormap(["#1a1a24", "#fed976", "#fd8d3c", "#bd0026"])
        previews["severity_map"] = array_to_base64_png(sev["severity_map"], cmap=sev_cmap, vmin=0, vmax=3)

    return {
        "status": "SUCCESS",
        "incident_name": incident_name,
        "sitrep": orchestrator._make_serializable(sitrep),
        "previews": previews,
        "threat_level": sitrep["threat_level"],
        "burned_area_km2": sitrep["burned_area_km2"],
        "burned_area_acres": sitrep["burned_area_acres"],
        "perimeter_km": sitrep["perimeter_km"],
        "arbitration_mode": sitrep["arbitration"]["mode"],
        "recommended_model": sitrep["arbitration"]["recommended_model"]
    }

@app.post("/api/chat")
async def tactical_chatbot_query(query: str = Form(...)):
    """
    Tactical Multi-Agent Chatbot Assistant.
    Answers natural language queries about the active wildfire incident.
    """
    sitrep = active_incident_state.get("sitrep")
    incident_name = active_incident_state.get("incident_name", "None")

    if sitrep is None:
        return {
            "reply": "No wildfire incident has been analyzed yet. Please select an incident preset (e.g. Pacific Palisades Wildfire) or upload satellite imagery to initiate multi-agent tactical analysis.",
            "sources": []
        }

    q = query.lower()
    arb = sitrep["arbitration"]
    delin = sitrep["delineation"]
    sev = sitrep["severity"]
    risk = sitrep["risk"]
    tiers = sev.get("severity_tiers", {})

    # Intent routing
    if any(k in q for k in ["area", "size", "acre", "hectare", "how big", "extent"]):
        reply = (
            f"**Total Burned Area for {incident_name}:**\n"
            f"• Delineated Burned Area: **{delin['burned_area_km2']:.2f} km²** ({delin['burned_area_acres']:.1f} acres)\n"
            f"• Scene Coverage: **{delin['burn_percentage']:.2f}%** of the monitored area ({delin['total_scene_km2']:.1f} km² total)\n"
            f"• Perimeter Boundary: **{sev['fire_perimeter_km']:.2f} km**\n"
            f"• Separate Fire Clusters: **{sev['num_fire_clusters']}** clusters"
        )
        sources = ["DelineationAgent", "SeverityQuantifierAgent"]

    elif any(k in q for k in ["threat", "risk", "level", "danger", "critical"]):
        reply = (
            f"**Threat Assessment for {incident_name}:**\n"
            f"• Overall Threat Rating: **{risk['overall_threat_level']}** (Score: {risk['threat_score']}/100)\n"
            f"• Post-Fire Debris Flow Hazard: **{risk['debris_flow_hazard']['level']}**\n"
            f"• Assessment: {risk['debris_flow_hazard']['assessment']}\n"
            f"• Containment Complexity: **{risk['containment_complexity']['level']}** (Irregularity ratio: {risk['containment_complexity']['shape_irregularity_ratio']})\n"
            f"• Soil Hydrophobicity: **{risk['soil_hydrophobicity_risk']}**"
        )
        sources = ["RiskAssessmentAgent"]

    elif any(k in q for k in ["debris", "mudslide", "landslide", "flood", "soil"]):
        reply = (
            f"**Debris Flow & Hydrological Hazard Report:**\n"
            f"• Hazard Level: **{risk['debris_flow_hazard']['level']}** (Hazard Score: {risk['debris_flow_hazard']['score']})\n"
            f"• Technical Assessment: {risk['debris_flow_hazard']['assessment']}\n"
            f"• Soil Hydrophobicity: **{risk['soil_hydrophobicity_risk']}** risk due to heat-induced vaporized organic compounds sealing soil pores.\n"
            f"• High Severity Burn Extent: **{tiers.get('high_severity', {}).get('km2', 0.0)} km²** ({tiers.get('high_severity', {}).get('pct_of_fire', 0.0)}% of total burn)."
        )
        sources = ["RiskAssessmentAgent", "SeverityQuantifierAgent"]

    elif any(k in q for k in ["arbitrat", "sensor", "optical", "sar", "cloud", "model", "why"]):
        reply = (
            f"**Sensor Arbitration Report:**\n"
            f"• Selected Mode: `{arb['mode']}`\n"
            f"• Chosen Deep Learning Backbone: `{arb['recommended_model']}`\n"
            f"• Optical Quality Score: **{arb['optical_quality_score']}** (Cloud Cover: {arb['cloud_cover_pct']:.1f}%)\n"
            f"• SAR Radar Quality Score: **{arb['sar_quality_score']}**\n"
            f"• Dynamic Sensor Weights: Optical: {arb['sensor_weights']['optical']*100:.0f}%, SAR: {arb['sensor_weights']['sar']*100:.0f}%\n"
            f"• Operational Reasoning: {arb['reasoning']}"
        )
        sources = ["SensorArbitratorAgent"]

    elif any(k in q for k in ["severity", "damage", "copernicus", "grade", "destruction"]):
        reply = (
            f"**Copernicus EMS Damage Stratification:**\n"
            f"• **Grade 1 (Low Severity)**: {tiers.get('low_severity', {}).get('km2', 0.0)} km² ({tiers.get('low_severity', {}).get('pct_of_fire', 0.0)}%)\n"
            f"• **Grade 2 (Moderate Severity)**: {tiers.get('moderate_severity', {}).get('km2', 0.0)} km² ({tiers.get('moderate_severity', {}).get('pct_of_fire', 0.0)}%)\n"
            f"• **Grade 3 (High Severity)**: {tiers.get('high_severity', {}).get('km2', 0.0)} km² ({tiers.get('high_severity', {}).get('pct_of_fire', 0.0)}%)\n"
            f"• High severity ratio indicates severe tree canopy mortality and hydrophobic ash bedding."
        )
        sources = ["SeverityQuantifierAgent"]

    elif any(k in q for k in ["action", "recommend", "baer", "tactical", "evacuat", "what should"]):
        recs = "\n".join([f"{i+1}. {r}" for i, r in enumerate(risk['tactical_recommendations'])])
        reply = f"**Tactical Incident & BAER Prescriptions:**\n{recs}"
        sources = ["RiskAssessmentAgent", "WildfireOrchestrator"]

    else:
        # General summary reply
        reply = (
            f"**Situation Summary for {incident_name}:**\n"
            f"• Status: {risk['overall_threat_level']}\n"
            f"• Burned Extent: {delin['burned_area_km2']:.2f} km² ({delin['burned_area_acres']:.1f} acres)\n"
            f"• Perimeter: {sev['fire_perimeter_km']:.2f} km across {sev['num_fire_clusters']} clusters\n"
            f"• Sensor Mode: {arb['mode']} using {arb['recommended_model']}\n"
            f"• Debris Flow Hazard: {risk['debris_flow_hazard']['level']}\n\n"
            f"Ask me about: *burned area metrics*, *debris flow hazard*, *sensor arbitration*, *damage tiers*, or *recommended actions*."
        )
        sources = ["WildfireOrchestrator"]

    return {
        "reply": reply,
        "incident_name": incident_name,
        "sources": sources
    }

@app.get("/api/sitrep/markdown")
async def get_sitrep_markdown():
    sitrep = active_incident_state.get("sitrep")
    if not sitrep:
        raise HTTPException(status_code=404, detail="No active Situation Report available.")
    return Response(content=sitrep["markdown_report"], media_type="text/markdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
