"""
Satellite Wildfire Dataset - Ground Truth Mask & Label Inspection Script
Phase 4: Inspects ground truth masks, computes class distributions,
generates label_distribution.csv, and produces visual overlays of masks.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    # Paths configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_zenodo = os.path.abspath(os.path.join(project_root, "..", "dataset", "Zenodo"))
    output_csv = os.path.join(project_root, "data", "inspection", "label_distribution.csv")
    vis_dir = os.path.join(project_root, "results", "visualizations", "ground_truth")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_zenodo}")
    print(f"Scanning for ground-truth masks...")

    # Find all mask PNG files and TIFF files
    mask_files = []
    for root, dirs, files in os.walk(dataset_zenodo):
        for f in files:
            if f.endswith("_mask.png"):
                mask_files.append(os.path.join(root, f))

    mask_files = sorted(mask_files)
    print(f"Found {len(mask_files)} ground truth mask files.")

    rows = []
    event_details = []

    for idx, mask_path in enumerate(mask_files):
        filename = os.path.basename(mask_path)
        event_id = filename.replace("_mask.png", "")
        parent_dir = os.path.dirname(mask_path)

        # Also find the corresponding TIFF to verify dtype, crs, nodata
        tiff_path = mask_path.replace(".png", ".tiff")
        has_tiff = os.path.exists(tiff_path)
        tiff_nodata = None
        tiff_shape = None
        tiff_dtype = None

        if has_tiff:
            try:
                with rasterio.open(tiff_path) as src:
                    tiff_nodata = src.nodata
                    tiff_shape = (src.height, src.width)
                    tiff_dtype = src.dtypes[0]
            except Exception as e:
                pass

        # Load PNG mask
        img = Image.open(mask_path)
        arr = np.array(img)
        total_pixels = arr.size
        unique_vals, counts = np.unique(arr, return_counts=True)

        event_details.append({
            "event_id": event_id,
            "mask_path": mask_path,
            "parent_dir": parent_dir,
            "shape": arr.shape,
            "total_pixels": total_pixels,
            "classes": unique_vals.tolist(),
            "has_tiff": has_tiff,
            "tiff_nodata": tiff_nodata,
            "tiff_dtype": tiff_dtype
        })

        for val, count in zip(unique_vals, counts):
            pct = (count / total_pixels) * 100.0
            rows.append({
                "event_id": event_id,
                "file": filename,
                "class_value": int(val),
                "pixel_count": int(count),
                "percentage": round(pct, 6)
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated label distribution CSV: {output_csv}")
    print(f"Total rows in CSV: {len(df)}")

    # Summary statistics
    print("\n" + "="*50)
    print("GLOBAL CLASS DISTRIBUTION SUMMARY")
    print("="*50)
    class_summary = df.groupby("class_value")["pixel_count"].sum().reset_index()
    grand_total = class_summary["pixel_count"].sum()
    class_summary["global_percentage"] = (class_summary["pixel_count"] / grand_total) * 100.0
    for _, row in class_summary.iterrows():
        cv = int(row['class_value'])
        cnt = int(row['pixel_count'])
        gp = row['global_percentage']
        print(f"Class {cv:3d}: {cnt:12,d} pixels ({gp:6.2f}%)")

    # Generate Visualizations for randomly selected events
    # We select diverse events: some with 0/64/128/192, some with 32, some with 255
    random.seed(42)
    sample_events = random.sample(event_details, min(8, len(event_details)))

    # Color mapping for classes:
    # 0: Dark gray (unburned)
    # 32: Light gray / muted green (unburned/slight)
    # 64: Yellow (moderate damage)
    # 128: Orange (high damage)
    # 192: Red (completely destroyed)
    # 255: Dark red / Crimson (burned delineation)
    color_map = {
        0: [30, 30, 30],
        32: [70, 90, 70],
        64: [230, 200, 20],
        128: [240, 130, 30],
        192: [220, 30, 30],
        255: [140, 10, 40]
    }
    class_labels = {
        0: "0: Unburned / Background",
        32: "32: Negligible / Unburned AOI",
        64: "64: Moderate Damage",
        128: "128: High Damage",
        192: "192: Completely Destroyed",
        255: "255: Severe / Delineated Burned"
    }

    print("\nGenerating ground truth visualizations...")
    for ev in sample_events:
        event_id = ev["event_id"]
        parent_dir = ev["parent_dir"]
        mask_arr = np.array(Image.open(ev["mask_path"]))

        # Look for Sentinel-2 preview RGB images in parent_dir
        s2_pngs = sorted(glob.glob(os.path.join(parent_dir, "sentinel2_*.png")))
        s2_pre = None
        s2_post = None

        # Filter out _coverage and _cloud_coverage
        s2_main_pngs = [p for p in s2_pngs if not ("_coverage" in p or "_cloud" in p)]
        if len(s2_main_pngs) >= 2:
            s2_pre = np.array(Image.open(s2_main_pngs[0]))
            s2_post = np.array(Image.open(s2_main_pngs[-1]))
        elif len(s2_main_pngs) == 1:
            s2_post = np.array(Image.open(s2_main_pngs[0]))

        # Look for Sentinel-1 preview RGB
        s1_pngs = sorted([p for p in glob.glob(os.path.join(parent_dir, "sentinel1_*.png")) if "_coverage" not in p])
        s1_img = np.array(Image.open(s1_pngs[-1])) if s1_pngs else None

        # Build RGB colormap representation of mask
        h, w = mask_arr.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for val, col in color_map.items():
            color_mask[mask_arr == val] = col

        # Plot multi-panel figure
        num_cols = 4 if (s2_pre is not None and s1_img is not None) else 3
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), dpi=150)

        col_idx = 0
        if s2_pre is not None:
            axes[col_idx].imshow(s2_pre)
            axes[col_idx].set_title(f"S2 Pre-Fire ({os.path.basename(s2_main_pngs[0])[10:20]})", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

        if s2_post is not None:
            axes[col_idx].imshow(s2_post)
            post_label = os.path.basename(s2_main_pngs[-1])[10:20] if s2_main_pngs else "Post-Fire"
            axes[col_idx].set_title(f"S2 Post-Fire ({post_label})", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

        if s1_img is not None and col_idx < num_cols - 1:
            axes[col_idx].imshow(s1_img)
            axes[col_idx].set_title(f"S1 Post-Fire SAR", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

        # Ground Truth Mask Panel
        axes[col_idx].imshow(color_mask)
        axes[col_idx].set_title(f"Ground Truth Mask ({event_id[:18]}...)", fontsize=10)
        axes[col_idx].axis("off")

        # Add legend for classes present in this mask
        unique_here = np.unique(mask_arr)
        patches = [
            mpatches.Patch(color=np.array(color_map[v]) / 255.0, label=f"{class_labels.get(v, str(v))} ({np.sum(mask_arr==v)/mask_arr.size*100:.1f}%)")
            for v in unique_here if v in color_map
        ]
        axes[col_idx].legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.8)

        plt.suptitle(f"Event: {event_id}\nShape: {h}x{w} | Classes present: {list(unique_here)}", fontsize=11, fontweight="bold")
        plt.tight_layout()

        out_img_path = os.path.join(vis_dir, f"{event_id}_inspection.png")
        plt.savefig(out_img_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization: {os.path.basename(out_img_path)}")

    print(f"\nAll visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()
