"""
Sentinel-1 SAR Patch Extraction and Training Normalization Script
Extracts geospatially consistent 256x256 patches (stride 128) across Train, Val, and Test folds.
Inputs: 3 channels (VV, VH, VV/VH ratio).
Filters by validity (>=95% using Band 4).
Samples training patches (~70% burned >=1%, ~30% background <1%).
Calculates robust per-band statistics strictly from TRAINING DATA ONLY.
"""

import os
import glob
import json
import random
import argparse
import numpy as np
import pandas as pd
import rasterio

def find_event_dirs(zenodo_root):
    event_dirs = {}
    for part in ['Satellite_burned_area_dataset_part1', 'Satellite_burned_area_dataset_part3', 'Satellite_burned_area_dataset_part4', 'Satellite_burned_area_dataset_part5']:
        inner = os.path.join(zenodo_root, part, part)
        if os.path.exists(inner):
            for e in os.listdir(inner):
                ep = os.path.join(inner, e)
                if os.path.isdir(ep):
                    event_dirs[e] = ep
    return event_dirs

def extract_s1_patches_for_event(event_dir, patch_size=256, stride=128):
    # 1. Find post-fire Sentinel-1 TIFF
    s1_tiffs = sorted([f for f in glob.glob(os.path.join(event_dir, "sentinel1_*.tiff")) if "_coverage" not in f])
    if not s1_tiffs:
        return []
    post_s1_tiff = s1_tiffs[-1]  # Latest acquisition is post-fire

    # 2. Identify ground-truth mask
    mask_tiffs = glob.glob(os.path.join(event_dir, "*_mask.tiff"))
    if not mask_tiffs:
        return []
    mask_tiff = mask_tiffs[0]

    # 3. Read data
    with rasterio.open(post_s1_tiff) as src_s1:
        s1_data = src_s1.read()  # (4, H, W)
        H, W = s1_data.shape[1], s1_data.shape[2]

    with rasterio.open(mask_tiff) as src_m:
        raw_mask = src_m.read(1)  # (H, W)

    # 3 channels: VV (idx 0), VH (idx 1), Ratio (idx 2)
    s1_channels = s1_data[:3, :, :].copy()  # (3, H, W)
    validity_mask = s1_data[3, :, :]        # Band 4 is validity mask

    # Clean and clamp ratio channel to avoid isolated infs
    s1_channels[0] = np.nan_to_num(s1_channels[0], nan=0.0, posinf=10.0, neginf=0.0)
    s1_channels[1] = np.nan_to_num(s1_channels[1], nan=0.0, posinf=10.0, neginf=0.0)
    s1_channels[2] = np.clip(np.nan_to_num(s1_channels[2], nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)

    # Convert ground-truth to binary:
    # 0, 32 -> 0 (Unburned)
    # 64, 128, 192, 255 -> 1 (Burned)
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    burned_pixels = np.isin(raw_mask, [64, 128, 192, 255])
    binary_mask[burned_pixels] = 1

    patches = []

    y_starts = list(range(0, H - patch_size + 1, stride))
    if (H - patch_size) % stride != 0 and H >= patch_size:
        y_starts.append(H - patch_size)

    x_starts = list(range(0, W - patch_size + 1, stride))
    if (W - patch_size) % stride != 0 and W >= patch_size:
        x_starts.append(W - patch_size)

    for y in y_starts:
        for x in x_starts:
            val_patch = validity_mask[y:y+patch_size, x:x+patch_size]
            val_fraction = np.mean(val_patch >= 0.5)
            if val_fraction < 0.95:
                continue

            img_patch = s1_channels[:, y:y+patch_size, x:x+patch_size]
            m_patch = binary_mask[y:y+patch_size, x:x+patch_size][np.newaxis, :, :]

            burned_ratio = float(np.mean(m_patch == 1))
            patches.append({
                "image": img_patch.astype(np.float32),
                "mask": m_patch.astype(np.uint8),
                "burned_ratio": burned_ratio,
                "event": os.path.basename(event_dir),
                "coord": (y, x)
            })

    return patches

def run_s1_extraction(project_root, dataset_root):
    csv_path = os.path.join(dataset_root, "satellite_data.csv")
    df = pd.read_csv(csv_path)
    event_dirs = find_event_dirs(dataset_root)

    test_events = df[df["fold"] == "cyan"]["folder"].tolist()
    val_events = df[df["fold"] == "purple"]["folder"].tolist()
    train_events = df[~df["fold"].isin(["cyan", "purple"])]["folder"].tolist()

    # Filter to events on disk that have S1 TIFFs
    train_events = [e for e in train_events if e in event_dirs and glob.glob(os.path.join(event_dirs[e], "sentinel1_*.tiff"))]
    val_events = [e for e in val_events if e in event_dirs and glob.glob(os.path.join(event_dirs[e], "sentinel1_*.tiff"))]
    test_events = [e for e in test_events if e in event_dirs and glob.glob(os.path.join(event_dirs[e], "sentinel1_*.tiff"))]

    print("="*60)
    print("STAGE 3 — SENTINEL-1 SAR PATCH EXTRACTION")
    print(f"  Train events: {len(train_events)}")
    print(f"  Val events:   {len(val_events)}")
    print(f"  Test events:  {len(test_events)}")
    print("="*60)

    out_dirs = {
        "train": os.path.join(project_root, "data", "processed_s1", "train"),
        "val": os.path.join(project_root, "data", "processed_s1", "val"),
        "test": os.path.join(project_root, "data", "processed_s1", "test")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # 1. PROCESS TRAINING SET
    print("\nProcessing TRAINING SAR events...")
    train_patches_burned = []
    train_patches_background = []

    for idx, ev in enumerate(train_events):
        patches = extract_s1_patches_for_event(event_dirs[ev])
        for p in patches:
            if p["burned_ratio"] >= 0.01:
                train_patches_burned.append(p)
            else:
                train_patches_background.append(p)
        print(f"[{idx+1:2d}/{len(train_events)}] {ev[:25]}: total={len(patches)} (burned={len([p for p in patches if p['burned_ratio']>=0.01])})")

    num_burned = len(train_patches_burned)
    target_bg = int(num_burned * (30.0 / 70.0))
    selected_bg = random.sample(train_patches_background, min(len(train_patches_background), target_bg))
    final_train_patches = train_patches_burned + selected_bg
    random.shuffle(final_train_patches)

    print(f"\nFinal Balanced Training Set: {len(final_train_patches)} patches ({len(train_patches_burned)} burned, {len(selected_bg)} background)")

    # 2. COMPUTE NORMALIZATION STATISTICS FROM TRAINING DATA ONLY
    print("\nCalculating robust per-band normalization statistics from TRAINING set only...")
    sample_size = min(len(final_train_patches), 1000)
    stat_samples = random.sample(final_train_patches, sample_size)
    stacked_images = np.stack([p["image"] for p in stat_samples], axis=0)  # (N, 3, 256, 256)

    channel_names = ["VV", "VH", "VV_VH_ratio"]
    stats_dict = {
        "channel_names": channel_names,
        "mean": [float(stacked_images[:, c].mean()) for c in range(3)],
        "std": [float(stacked_images[:, c].std()) for c in range(3)],
        "min": [float(stacked_images[:, c].min()) for c in range(3)],
        "max": [float(stacked_images[:, c].max()) for c in range(3)],
        "p1": [float(np.percentile(stacked_images[:, c], 1)) for c in range(3)],
        "p99": [float(np.percentile(stacked_images[:, c], 99)) for c in range(3)]
    }

    stats_file = os.path.join(project_root, "data", "inspection", "s1_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved normalization statistics to: {stats_file}")
    for c in range(3):
        print(f"  {channel_names[c]:12s}: mean={stats_dict['mean'][c]:.4f}, std={stats_dict['std'][c]:.4f}, min={stats_dict['min'][c]:.4f}, max={stats_dict['max'][c]:.4f}")

    # Save training patches
    print("\nSaving training SAR patches...")
    for idx, p in enumerate(final_train_patches):
        p_name = f"train_s1_patch_{idx:05d}_{p['event']}.npz"
        np.savez_compressed(os.path.join(out_dirs["train"], p_name), image=p["image"], mask=p["mask"])

    # 3. PROCESS VALIDATION SET (natural distribution)
    print("\nProcessing VALIDATION SAR events (natural distribution)...")
    val_patch_count = 0
    for idx, ev in enumerate(val_events):
        patches = extract_s1_patches_for_event(event_dirs[ev])
        for p in patches:
            p_name = f"val_s1_patch_{val_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["val"], p_name), image=p["image"], mask=p["mask"])
            val_patch_count += 1
        print(f"[{idx+1:2d}/{len(val_events)}] {ev[:25]}: extracted {len(patches)} patches")

    # 4. PROCESS TEST SET (natural distribution)
    print("\nProcessing TEST SAR events (natural distribution)...")
    test_patch_count = 0
    for idx, ev in enumerate(test_events):
        patches = extract_s1_patches_for_event(event_dirs[ev])
        for p in patches:
            p_name = f"test_s1_patch_{test_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["test"], p_name), image=p["image"], mask=p["mask"])
            test_patch_count += 1
        print(f"[{idx+1:2d}/{len(test_events)}] {ev[:25]}: extracted {len(patches)} patches")

    print("\n" + "="*60)
    print("SAR PATCH EXTRACTION COMPLETED!")
    print(f"  Train S1 Patches: {len(final_train_patches)}")
    print(f"  Val S1 Patches:   {val_patch_count}")
    print(f"  Test S1 Patches:  {test_patch_count}")
    print("="*60)

if __name__ == "__main__":
    proj = r"C:\Users\Sahil\OneDrive\Desktop\AI project\satellite_wildfire_project"
    zenodo = r"C:\Users\Sahil\OneDrive\Desktop\AI project\dataset\Zenodo"
    run_s1_extraction(proj, zenodo)
