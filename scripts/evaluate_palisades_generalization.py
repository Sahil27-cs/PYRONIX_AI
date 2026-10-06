"""
Stage 7: External Wildfire Generalization Benchmark — Pacific Palisades Wildfire
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Evaluates zero-shot domain transfer and generalization performance of trained models
(Sentinel-1 SAR, Sentinel-2 Optical, and Multimodal Fusion) on an unseen external wildfire:
the May 2021 Pacific Palisades / Topanga Wildfire (Los Angeles County, California, USA).

Reference:
- Ground Truth: dNBR (differenced Normalized Burn Ratio) derived from pre/post Landsat/Sentinel-2
- Sensors:
    * Sentinel-1A Terrain-Corrected Dual-Pol SAR (VV, VH, VV/VH ratio)
    * Sentinel-2 Multi-Spectral Optical (Blue, Green, Red, NIR)
- Geographic Region: Southern California Coastal Mountains (Chaparral & Wildland-Urban Interface)
- Training Distribution: European Wildfires (Copernicus EMS Zenodo Dataset)
"""

import os
import sys
import time
import json
import torch
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import ResNet34UNet
from src.evaluation.metrics import MetricTracker

def prepare_palisades_data(palisades_dir, col_off=5120, row_off=2048, width=1536, height=1536):
    """
    Extracts and aligns multi-sensor arrays over the Palisades Wildfire Region of Interest.
    Returns:
        dnbr: (H, W) float32
        rgbn: (4, H, W) float32 in [0, 1] (B2_Blue, B3_Green, B4_Red, B8_NIR)
        s1: (3, H, W) float32 (VV, VH, VV/VH ratio)
        meta: dict of spatial metadata
    """
    print(f"\nExtracting Palisades ROI: col={col_off}, row={row_off}, size={width}x{height} (10m res, ~{width*height*0.0001:.1f} km^2)...")
    win = rasterio.windows.Window(col_off, row_off, width, height)

    dnbr_path = os.path.join(palisades_dir, "DNBR.tif")
    rgbn_path = os.path.join(palisades_dir, "rgbn.tif")
    s1_path = os.path.join(palisades_dir, "S1A_asc_TC_32611.tif")

    with rasterio.open(dnbr_path) as src_d:
        dnbr = src_d.read(1, window=win).astype(np.float32)
        win_transform = rasterio.windows.transform(win, src_d.transform)
        win_crs = src_d.crs

    with rasterio.open(rgbn_path) as src_r:
        # Bands 1..4: Blue, Green, Red, NIR
        rgbn_raw = src_r.read(window=win).astype(np.float32)
        rgbn = np.clip(rgbn_raw / 10000.0, 0.0, 1.0)

    # Reproject S1 SAR to match the exact same grid
    with rasterio.open(s1_path) as src_s:
        # Band 12 is post-fire VV, Band 11 is post-fire VH
        vv_raw = np.zeros((height, width), dtype=np.float32)
        vh_raw = np.zeros((height, width), dtype=np.float32)

        rasterio.warp.reproject(
            source=rasterio.band(src_s, 12),
            destination=vv_raw,
            src_transform=src_s.transform,
            src_crs=src_s.crs,
            dst_transform=win_transform,
            dst_crs=win_crs,
            resampling=Resampling.bilinear
        )
        rasterio.warp.reproject(
            source=rasterio.band(src_s, 11),
            destination=vh_raw,
            src_transform=src_s.transform,
            src_crs=src_s.crs,
            dst_transform=win_transform,
            dst_crs=win_crs,
            resampling=Resampling.bilinear
        )

    # Clean SAR channels
    vv = np.nan_to_num(vv_raw, nan=0.0, posinf=10.0, neginf=0.0)
    vh = np.nan_to_num(vh_raw, nan=0.0, posinf=10.0, neginf=0.0)
    # Ratio = (VV / (VH + 1e-6)) / 50.0 clamped to [0, 5.0]
    ratio = np.clip((vv / (vh + 1e-6)) / 50.0, 0.0, 5.0)

    s1 = np.stack([vv, vh, ratio], axis=0)

    meta = {
        "col_off": col_off,
        "row_off": row_off,
        "width": width,
        "height": height,
        "crs": str(win_crs),
        "area_km2": float(width * height * 0.0001)
    }

    return dnbr, rgbn, s1, meta

def tile_into_patches(image_array, patch_size=256, stride=128):
    """
    Extracts sliding-window patches from (C, H, W) array.
    Returns: patches (N, C, patch_size, patch_size), positions list [(y, x)]
    """
    C, H, W = image_array.shape
    patches = []
    positions = []

    y_steps = list(range(0, H - patch_size + 1, stride))
    if (H - patch_size) % stride != 0:
        y_steps.append(H - patch_size)

    x_steps = list(range(0, W - patch_size + 1, stride))
    if (W - patch_size) % stride != 0:
        x_steps.append(W - patch_size)

    for y in y_steps:
        for x in x_steps:
            patch = image_array[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches, dtype=np.float32), positions

def reconstruct_from_patches(patch_preds, positions, H, W, patch_size=256):
    """
    Reconstructs full-scene probability map by averaging overlapping predictions.
    """
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for pred, (y, x) in zip(patch_preds, positions):
        prob_map[y:y+patch_size, x:x+patch_size] += pred
        weight_map[y:y+patch_size, x:x+patch_size] += 1.0

    weight_map = np.maximum(weight_map, 1.0)
    return prob_map / weight_map

def compute_binary_metrics(pred_binary, gt_binary):
    """
    Computes full pixel-level segmentation metrics.
    """
    tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
    tn = np.logical_and(pred_binary == 0, gt_binary == 0).sum()

    total = tp + fp + fn + tn
    iou = float(tp / (tp + fp + fn + 1e-8))
    dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    accuracy = float((tp + tn) / (total + 1e-8))
    specificity = float(tn / (tn + fp + 1e-8))

    gt_km2 = float(gt_binary.sum() * 0.0001)
    pred_km2 = float(pred_binary.sum() * 0.0001)
    bias_km2 = pred_km2 - gt_km2
    bias_pct = float((bias_km2 / (gt_km2 + 1e-6)) * 100.0)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "gt_km2": gt_km2,
        "pred_km2": pred_km2,
        "bias_km2": bias_km2,
        "bias_pct": bias_pct,
        "abs_error_km2": abs(bias_km2)
    }

def main():
    print("=" * 75)
    print("STAGE 7: EXTERNAL WILDFIRE GENERALIZATION TEST (PACIFIC PALISADES)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    palisades_dir = os.path.join(project_root, "data", "extracted", "palisades")
    models_dir = os.path.join(project_root, "models")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    vis_dir = os.path.join(project_root, "results", "visualizations", "palisades")
    stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    s1_stats_json = os.path.join(project_root, "data", "inspection", "s1_statistics.json")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Preprocessed Palisades Data
    dnbr, rgbn, s1, meta = prepare_palisades_data(palisades_dir)
    H, W = dnbr.shape

    # Binary Ground Truth Masks at standard severity thresholds
    # Standard USGS: dNBR >= 0.15 represents burned area (low to high severity)
    gt_masks = {
        "dnbr_010": (dnbr >= 0.10).astype(np.uint8),
        "dnbr_015": (dnbr >= 0.15).astype(np.uint8),
        "dnbr_020": (dnbr >= 0.20).astype(np.uint8),
        "dnbr_027": (dnbr >= 0.27).astype(np.uint8)
    }
    primary_gt = gt_masks["dnbr_015"]
    print(f"Ground Truth Burned Area (dNBR >= 0.15): {primary_gt.sum()*0.0001:.2f} km^2 ({primary_gt.mean()*100:.2f}% of scene)")

    # Load Normalization Statistics
    with open(stats_json) as f:
        fusion_st = json.load(f)
    with open(s1_stats_json) as f:
        s1_st = json.load(f)

    # Prepare normalized model input rasters
    # Optical B2, B3, B4 (RGB)
    rgb_mean = np.array(fusion_st["mean"][:3], dtype=np.float32).reshape(3, 1, 1)
    rgb_std = np.array(fusion_st["std"][:3], dtype=np.float32).reshape(3, 1, 1)
    norm_rgb = (rgbn[:3] - rgb_mean) / rgb_std

    # Optical RGB + NIR (B2, B3, B4, B8)
    rgbn_mean = np.array(fusion_st["mean"][:4], dtype=np.float32).reshape(4, 1, 1)
    rgbn_std = np.array(fusion_st["std"][:4], dtype=np.float32).reshape(4, 1, 1)
    norm_rgbn = (rgbn[:4] - rgbn_mean) / rgbn_std

    # SAR VV, VH, Ratio
    s1_mean = np.array(s1_st["mean"], dtype=np.float32).reshape(3, 1, 1)
    s1_std = np.array(s1_st["std"], dtype=np.float32).reshape(3, 1, 1)
    norm_s1 = (s1 - s1_mean) / s1_std

    # 2. Extract Patches for Inference (256x256 with 128 stride)
    patch_size = 256
    stride = 128

    rgb_patches, positions = tile_into_patches(norm_rgb, patch_size, stride)
    s1_patches, _ = tile_into_patches(norm_s1, patch_size, stride)

    print(f"Extracted {len(positions)} evaluation patches ({patch_size}x{patch_size}, stride={stride})")

    # 3. Models to Evaluate
    models_to_test = [
        {
            "name": "S2_RGB_only",
            "desc": "Sentinel-2 Optical (RGB Bands B2, B3, B4)",
            "path": os.path.join(models_dir, "ablation_S2_RGB_only.pt"),
            "in_channels": 3,
            "inputs": rgb_patches,
            "channels_used": "B2, B3, B4"
        },
        {
            "name": "S1_VV_only",
            "desc": "Sentinel-1 SAR (Single-Pol VV)",
            "path": os.path.join(models_dir, "ablation_S1_VV_only.pt"),
            "in_channels": 1,
            "inputs": s1_patches[:, 0:1, :, :],
            "channels_used": "VV"
        },
        {
            "name": "S1_VH_only",
            "desc": "Sentinel-1 SAR (Cross-Pol VH)",
            "path": os.path.join(models_dir, "ablation_S1_VH_only.pt"),
            "in_channels": 1,
            "inputs": s1_patches[:, 1:2, :, :],
            "channels_used": "VH"
        },
        {
            "name": "S1_VV_VH",
            "desc": "Sentinel-1 SAR (Dual-Pol VV+VH)",
            "path": os.path.join(models_dir, "ablation_S1_VV_VH.pt"),
            "in_channels": 2,
            "inputs": s1_patches[:, 0:2, :, :],
            "channels_used": "VV, VH"
        },
        {
            "name": "S1_Full",
            "desc": "Sentinel-1 SAR Baseline (VV, VH, VV/VH Ratio)",
            "path": os.path.join(models_dir, "best_s1_baseline_model.pt"),
            "in_channels": 3,
            "inputs": s1_patches,
            "channels_used": "VV, VH, Ratio"
        }
    ]

    benchmark_results = {}
    predicted_maps = {}

    print("\n" + "=" * 60)
    print("RUNNING ZERO-SHOT GENERALIZATION INFERENCE ON PALISADES...")
    print("=" * 60)

    batch_size = 16

    for m_info in models_to_test:
        name = m_info["name"]
        print(f"\nEvaluating Model: {name} ({m_info['desc']})...")
        ckpt = torch.load(m_info["path"], map_location=device, weights_only=False)
        model = ResNet34UNet(in_channels=m_info["in_channels"], num_classes=1, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        patch_tensor = torch.from_numpy(m_info["inputs"])
        loader = DataLoader(TensorDataset(patch_tensor), batch_size=batch_size, shuffle=False)

        pred_patches_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().squeeze(1).numpy()
                pred_patches_list.append(probs)

        pred_patches = np.concatenate(pred_patches_list, axis=0)

        # Reconstruct full probability map
        prob_map = reconstruct_from_patches(pred_patches, positions, H, W, patch_size)
        pred_binary = (prob_map >= 0.5).astype(np.uint8)

        predicted_maps[name] = {
            "prob": prob_map,
            "binary": pred_binary
        }

        # Compute metrics across multiple dNBR thresholds
        metrics_primary = compute_binary_metrics(pred_binary, primary_gt)
        metrics_010 = compute_binary_metrics(pred_binary, gt_masks["dnbr_010"])
        metrics_020 = compute_binary_metrics(pred_binary, gt_masks["dnbr_020"])
        metrics_027 = compute_binary_metrics(pred_binary, gt_masks["dnbr_027"])

        benchmark_results[name] = {
            "model_name": name,
            "description": m_info["desc"],
            "channels_used": m_info["channels_used"],
            "metrics_primary_dnbr015": metrics_primary,
            "metrics_dnbr010": metrics_010,
            "metrics_dnbr020": metrics_020,
            "metrics_dnbr027": metrics_027
        }

        print(f"  Test IoU:       {metrics_primary['iou']*100:.2f}%")
        print(f"  Test Dice:      {metrics_primary['dice']*100:.2f}%")
        print(f"  Precision:      {metrics_primary['precision']*100:.2f}%")
        print(f"  Recall:         {metrics_primary['recall']*100:.2f}%")
        print(f"  Accuracy:       {metrics_primary['accuracy']*100:.2f}%")
        print(f"  Area Predicted: {metrics_primary['pred_km2']:.2f} km^2 vs GT: {metrics_primary['gt_km2']:.2f} km^2")
        print(f"  Area Bias:      {metrics_primary['bias_pct']:+.2f}% (Error: {metrics_primary['bias_km2']:+.2f} km^2)")

    # 4. Burn Severity Stratification Analysis
    print("\n" + "=" * 60)
    print("BURN SEVERITY RECALL STRATIFICATION (USGS SEVERITY CLASSES)...")
    print("=" * 60)

    severity_classes = {
        "Unburned (dNBR < 0.10)": (dnbr < 0.10),
        "Low Severity (0.10 <= dNBR < 0.27)": (dnbr >= 0.10) & (dnbr < 0.27),
        "Moderate Severity (0.27 <= dNBR < 0.66)": (dnbr >= 0.27) & (dnbr < 0.66),
        "High Severity (dNBR >= 0.66)": (dnbr >= 0.66)
    }

    stratification_rows = []
    for s_name, s_mask in severity_classes.items():
        s_count = int(s_mask.sum())
        row = {"Severity Class": s_name, "Pixels": s_count, "Area (km2)": round(s_count * 0.0001, 2)}
        for m_info in models_to_test:
            m_name = m_info["name"]
            p_bin = predicted_maps[m_name]["binary"]
            if "Unburned" in s_name:
                # False positive rate in unburned zones
                fpr = float((p_bin[s_mask] == 1).mean() * 100.0)
                row[f"{m_name} (FPR %)"] = round(fpr, 2)
            else:
                # Recall in burned zones
                rec = float((p_bin[s_mask] == 1).mean() * 100.0)
                row[f"{m_name} (Recall %)"] = round(rec, 2)
        stratification_rows.append(row)

    strat_df = pd.DataFrame(stratification_rows)
    strat_csv_path = os.path.join(metrics_dir, "palisades_severity_stratification.csv")
    strat_df.to_csv(strat_csv_path, index=False)
    print(strat_df.to_string(index=False))
    print(f"\nSaved severity breakdown to {strat_csv_path}")

    # 5. Build Summary Tables & Save Metrics
    summary_rows = []
    for m_info in models_to_test:
        name = m_info["name"]
        res = benchmark_results[name]["metrics_primary_dnbr015"]
        summary_rows.append({
            "Model": name,
            "Modality": "Optical" if "S2" in name else "SAR",
            "Channels": m_info["channels_used"],
            "IoU (%)": round(res["iou"] * 100.0, 2),
            "Dice (%)": round(res["dice"] * 100.0, 2),
            "Precision (%)": round(res["precision"] * 100.0, 2),
            "Recall (%)": round(res["recall"] * 100.0, 2),
            "Accuracy (%)": round(res["accuracy"] * 100.0, 2),
            "Pred Area (km2)": round(res["pred_km2"], 2),
            "GT Area (km2)": round(res["gt_km2"], 2),
            "Area Bias (%)": round(res["bias_pct"], 2),
            "Abs Area Err (km2)": round(res["abs_error_km2"], 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(metrics_dir, "palisades_generalization_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved generalization summary table to {summary_csv_path}")

    # Full JSON output
    final_output = {
        "stage": "STAGE 7: EXTERNAL WILDFIRE GENERALIZATION BENCHMARK (PALISADES)",
        "fire_name": "Pacific Palisades & Topanga Wildfire (May 2021)",
        "location": "Los Angeles County, California, USA",
        "roi_metadata": meta,
        "models_evaluated": benchmark_results,
        "severity_stratification": stratification_rows,
        "summary_table": summary_rows
    }
    json_path = os.path.join(metrics_dir, "palisades_generalization.json")
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved complete study metrics to {json_path}")

    # 6. Generate Publication-Quality Visualizations
    print("\n" + "=" * 60)
    print("GENERATING PALISADES PUBLICATION VISUALIZATIONS...")
    print("=" * 60)

    # Plot 1: Full-Scene Multi-Sensor & Model Prediction Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # S2 RGB composite (B4=idx 2, B3=idx 1, B2=idx 0)
    rgb_comp = np.stack([rgbn[2], rgbn[1], rgbn[0]], axis=-1)
    rgb_comp = np.clip(rgb_comp / np.percentile(rgb_comp, 98), 0, 1)

    # False Color NIR composite (B8=idx 3, B4=idx 2, B3=idx 1)
    fcolor_comp = np.stack([rgbn[3], rgbn[2], rgbn[1]], axis=-1)
    fcolor_comp = np.clip(fcolor_comp / np.percentile(fcolor_comp, 98), 0, 1)

    # SAR VH backscatter (idx 1)
    sar_vh = np.clip(s1[1] / np.percentile(s1[1], 98), 0, 1)

    # Row 1: Sensors and Ground Truth
    axes[0, 0].imshow(rgb_comp)
    axes[0, 0].set_title("Sentinel-2 Optical (True Color RGB)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(fcolor_comp)
    axes[0, 1].set_title("Sentinel-2 False Color (NIR-R-G)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sar_vh, cmap="gray")
    axes[0, 2].set_title("Sentinel-1 SAR (VH Backscatter)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    im_d = axes[0, 3].imshow(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=0.8)
    axes[0, 3].set_title("Ground Truth dNBR Burn Severity", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    # Row 2: Model Predictions
    axes[1, 0].imshow(primary_gt, cmap="Reds")
    axes[1, 0].set_title(f"Ground Truth Binary Mask (dNBR >= 0.15)\nArea: {primary_gt.sum()*0.0001:.1f} km²", fontsize=11, fontweight="bold", color="darkred")
    axes[1, 0].axis("off")

    # Model: S2_RGB_only
    m_rgb_res = benchmark_results["S2_RGB_only"]["metrics_primary_dnbr015"]
    axes[1, 1].imshow(predicted_maps["S2_RGB_only"]["binary"], cmap="Purples")
    axes[1, 1].set_title(f"S2 RGB Only (Optical Zero-Shot)\nIoU: {m_rgb_res['iou']*100:.1f}% | Dice: {m_rgb_res['dice']*100:.1f}%\nPred Area: {m_rgb_res['pred_km2']:.1f} km²", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Model: S1_VV_VH
    m_vvvh_res = benchmark_results["S1_VV_VH"]["metrics_primary_dnbr015"]
    axes[1, 2].imshow(predicted_maps["S1_VV_VH"]["binary"], cmap="copper")
    axes[1, 2].set_title(f"S1 Dual-Pol VV+VH (SAR Zero-Shot)\nIoU: {m_vvvh_res['iou']*100:.1f}% | Dice: {m_vvvh_res['dice']*100:.1f}%\nPred Area: {m_vvvh_res['pred_km2']:.1f} km²", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    # Model: S1_Full
    m_s1_res = benchmark_results["S1_Full"]["metrics_primary_dnbr015"]
    axes[1, 3].imshow(predicted_maps["S1_Full"]["binary"], cmap="Blues")
    axes[1, 3].set_title(f"S1 Full Baseline (VV+VH+Ratio Zero-Shot)\nIoU: {m_s1_res['iou']*100:.1f}% | Dice: {m_s1_res['dice']*100:.1f}%\nPred Area: {m_s1_res['pred_km2']:.1f} km²", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle("External Domain Transfer: Model Delineation on Palisades Wildfire (Los Angeles, USA)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    chart1_path = os.path.join(vis_dir, "palisades_spatial_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path}")

    # Plot 2: Quantitative Performance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_rows))
    width = 0.20

    ious = [r["IoU (%)"] for r in summary_rows]
    dices = [r["Dice (%)"] for r in summary_rows]
    precs = [r["Precision (%)"] for r in summary_rows]
    recs = [r["Recall (%)"] for r in summary_rows]

    rects1 = ax.bar(x - 1.5*width, ious, width, label="IoU (%)", color="#2b5c8f")
    rects2 = ax.bar(x - 0.5*width, dices, width, label="Dice (F1) (%)", color="#2ca02c")
    rects3 = ax.bar(x + 0.5*width, precs, width, label="Precision (%)", color="#ff7f0e")
    rects4 = ax.bar(x + 1.5*width, recs, width, label="Recall (%)", color="#d62728")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Zero-Shot Generalization on Palisades Wildfire Across Models", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    labels = [r["Model"] for r in summary_rows]
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 100)

    for bar_group in [rects1, rects2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart2_path = os.path.join(vis_dir, "palisades_metrics_chart.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path}")

    # Plot 3: Error Classification Overlays (TP, FP, FN)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models_for_error = ["S2_RGB_only", "S1_VV_VH", "S1_Full"]

    for i, m_name in enumerate(models_for_error):
        pred_b = predicted_maps[m_name]["binary"]
        error_rgb = np.ones((H, W, 3), dtype=np.float32)  # White background

        # Unburned true negative = Light gray
        error_rgb[(pred_b == 0) & (primary_gt == 0)] = [0.92, 0.92, 0.92]
        # True Positive (Hit) = Green
        error_rgb[(pred_b == 1) & (primary_gt == 1)] = [0.15, 0.68, 0.25]
        # False Positive (False Alarm) = Orange
        error_rgb[(pred_b == 1) & (primary_gt == 0)] = [0.95, 0.55, 0.15]
        # False Negative (Miss) = Red
        error_rgb[(pred_b == 0) & (primary_gt == 1)] = [0.85, 0.15, 0.15]

        axes[i].imshow(error_rgb)
        axes[i].set_title(f"{m_name}\nIoU: {benchmark_results[m_name]['metrics_primary_dnbr015']['iou']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    # Legend handles
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=[0.15, 0.68, 0.25], label="True Positive (Burned Detected)"),
        mpatches.Patch(color=[0.95, 0.55, 0.15], label="False Positive (Commission Error)"),
        mpatches.Patch(color=[0.85, 0.15, 0.15], label="False Negative (Omission Error)"),
        mpatches.Patch(color=[0.92, 0.92, 0.92], label="True Negative (Unburned Background)")
    ]
    axes[1].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.95, fontsize=10)

    plt.suptitle("Palisades Wildfire Error Breakdown: True Positives vs False Alarms vs Misses", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart3_path = os.path.join(vis_dir, "palisades_error_maps.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart3_path}")

    print("\n" + "=" * 60)
    print("STAGE 7 GENERALIZATION BENCHMARK COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
