"""
Stage 8: Autonomous Multi-Agent Wildfire Intelligence Execution
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Executes end-to-end multi-agent wildfire intelligence on real incidents:
1. Incident A: Pacific Palisades Wildfire (Los Angeles County, CA, USA)
2. Incident B: European Copernicus EMS Wildfire Activation (Shared Test Partition)

Generates official Situation Reports (SitRep) and publication multi-agent overview graphics.
"""

import os
import sys
import glob
import numpy as np
import torch
import rasterio

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import WildfireOrchestrator

def load_palisades_scene(palisades_dir, col_off=5120, row_off=2048, width=1536, height=1536):
    """Loads and reprojects multi-sensor arrays over the Palisades Wildfire."""
    import rasterio.warp
    from rasterio.enums import Resampling

    win = rasterio.windows.Window(col_off, row_off, width, height)
    dnbr_p = os.path.join(palisades_dir, "DNBR.tif")
    rgbn_p = os.path.join(palisades_dir, "rgbn.tif")
    s1_p = os.path.join(palisades_dir, "S1A_asc_TC_32611.tif")

    with rasterio.open(dnbr_p) as s_d:
        dnbr = s_d.read(1, window=win).astype(np.float32)
        win_tr = rasterio.windows.transform(win, s_d.transform)
        win_crs = s_d.crs

    with rasterio.open(rgbn_p) as s_r:
        rgbn_raw = s_r.read(window=win).astype(np.float32)
        rgbn = np.clip(rgbn_raw / 10000.0, 0.0, 1.0)

    with rasterio.open(s1_p) as s_s:
        vv_raw = np.zeros((height, width), dtype=np.float32)
        vh_raw = np.zeros((height, width), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(s_s, 12),
            destination=vv_raw,
            src_transform=s_s.transform,
            src_crs=s_s.crs,
            dst_transform=win_tr,
            dst_crs=win_crs,
            resampling=Resampling.bilinear
        )
        rasterio.warp.reproject(
            source=rasterio.band(s_s, 11),
            destination=vh_raw,
            src_transform=s_s.transform,
            src_crs=s_s.crs,
            dst_transform=win_tr,
            dst_crs=win_crs,
            resampling=Resampling.bilinear
        )

    vv = np.nan_to_num(vv_raw, nan=0.0, posinf=10.0, neginf=0.0)
    vh = np.nan_to_num(vh_raw, nan=0.0, posinf=10.0, neginf=0.0)
    ratio = np.clip((vv / (vh + 1e-6)) / 50.0, 0.0, 5.0)
    s1 = np.stack([vv, vh, ratio], axis=0)

    return {
        "incident_name": "Pacific Palisades Wildfire",
        "optical_data": rgbn,
        "sar_data": s1,
        "dnbr_data": dnbr,
        "metadata": {
            "location": "Topanga Canyon & Pacific Palisades, Los Angeles County, CA",
            "country": "United States",
            "date": "May 2021",
            "resolution_m": 10.0,
            "crs": str(win_crs)
        }
    }

def load_zenodo_test_sample(test_dir):
    """Loads a prominent multi-modal test patch from the Zenodo dataset."""
    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    # Pick a patch with active fire
    chosen_f = files[0]
    for f in files:
        with np.load(f) as d:
            if d["mask"].sum() > 3000:
                chosen_f = f
                break

    fname = os.path.basename(chosen_f)
    event_name = "_".join(fname.split("_")[4:]).replace(".npz", "").replace("_", " ").title()

    with np.load(chosen_f) as d:
        img_9 = d["image"].astype(np.float32)  # (9, 256, 256)
        mask = d["mask"][0].astype(np.uint8)   # (256, 256)

    # In Zenodo 9-channel format:
    # 0..5: S2 Optical (B2, B3, B4, B8, B11, B12)
    # 6..8: S1 SAR (VV, VH, Ratio)
    optical_6 = img_9[:6]
    sar_3 = img_9[6:9]

    return {
        "incident_name": f"Copernicus EMS {event_name}",
        "optical_data": optical_6,
        "sar_data": sar_3,
        "gt_mask": mask,
        "metadata": {
            "source": "Copernicus Emergency Management Service Rapid Mapping",
            "patch_file": fname,
            "resolution_m": 10.0
        }
    }

def main():
    print("=" * 75)
    print("STAGE 8: MULTI-AGENT WILDFIRE ANALYSIS SYSTEM DEPLOYMENT")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    models_dir = os.path.join(project_root, "models")
    stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    s1_stats_json = os.path.join(project_root, "data", "inspection", "s1_statistics.json")
    reports_dir = os.path.join(project_root, "results", "reports")
    vis_dir = os.path.join(project_root, "results", "visualizations", "multiagent")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize Master Multi-Agent Orchestrator
    orchestrator = WildfireOrchestrator(
        models_dir=models_dir,
        stats_json=stats_json,
        s1_stats_json=s1_stats_json,
        device=device
    )

    # -------------------------------------------------------------
    # Scenario 1: Pacific Palisades Wildfire (External Benchmark)
    # -------------------------------------------------------------
    palisades_dir = os.path.join(project_root, "data", "extracted", "palisades")
    if os.path.exists(palisades_dir):
        print("\n" + "#" * 65)
        print("SCENARIO 1: PACIFIC PALISADES WILDFIRE (SOUTHERN CALIFORNIA, USA)")
        print("#" * 65)
        palisades_ctx = load_palisades_scene(palisades_dir)
        palisades_sitrep = orchestrator.run_analysis(palisades_ctx, output_dir=reports_dir)

        # Copy overview plot to visualizations directory
        src_fig = os.path.join(reports_dir, "pacific_palisades_wildfire_multiagent_overview.png")
        dst_fig = os.path.join(vis_dir, "pacific_palisades_wildfire_multiagent_overview.png")
        if os.path.exists(src_fig):
            import shutil
            shutil.copy2(src_fig, dst_fig)

    # -------------------------------------------------------------
    # Scenario 2: Copernicus EMS European Test Wildfire
    # -------------------------------------------------------------
    zenodo_test_dir = os.path.join(project_root, "data", "processed_fusion", "test")
    if os.path.exists(zenodo_test_dir):
        print("\n" + "#" * 65)
        print("SCENARIO 2: COPERNICUS EMS TEST SCENE (SHARED BENCHMARK)")
        print("#" * 65)
        zenodo_ctx = load_zenodo_test_sample(zenodo_test_dir)
        zenodo_sitrep = orchestrator.run_analysis(zenodo_ctx, output_dir=reports_dir)

        src_fig2 = os.path.join(reports_dir, f"{zenodo_ctx['incident_name'].lower().replace(' ', '_')}_multiagent_overview.png")
        dst_fig2 = os.path.join(vis_dir, "zenodo_test_scene_multiagent_overview.png")
        if os.path.exists(src_fig2):
            import shutil
            shutil.copy2(src_fig2, dst_fig2)

    print("\n" + "=" * 75)
    print("STAGE 8 MULTI-AGENT EXECUTION COMPLETE!")
    print(f"Situation Reports saved to: {reports_dir}")
    print(f"Overview Visualizations saved to: {vis_dir}")
    print("=" * 75)

if __name__ == "__main__":
    main()
