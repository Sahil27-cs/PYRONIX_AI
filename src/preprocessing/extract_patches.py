"""
Sentinel-2 Patch Extraction and Robust Training Normalization Script
Extracts geospatially consistent 256x256 patches (stride 128) across Train, Val, and Test folds.
Filters by validity (>=95%) and cloud coverage (<=20% if mask present).
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
from PIL import Image
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

def extract_patches_for_event(event_dir, patch_size=256, stride=128):
    """
    Extracts valid patches from a single wildfire event folder.
    Returns: list of dicts with 'image': [6, 256, 256], 'mask': [1, 256, 256], 'burned_ratio': float
    """
    # 1. Identify post-fire Sentinel-2 TIFF
    s2_tiffs = sorted([f for f in glob.glob(os.path.join(event_dir, "sentinel2_*.tiff")) if "_cloud" not in f])
    if not s2_tiffs:
        return []
    post_s2_tiff = s2_tiffs[-1]  # Latest acquisition is post-fire

    # 2. Check for optional cloud mask matching the post-fire acquisition
    post_date_str = os.path.basename(post_s2_tiff).replace("sentinel2_", "").replace(".tiff", "")
    cloud_candidates = glob.glob(os.path.join(event_dir, f"sentinel2_{post_date_str}_cloud_coverage.tiff"))
    cloud_tiff = cloud_candidates[0] if cloud_candidates else None

    # 3. Identify ground-truth mask
    mask_tiffs = glob.glob(os.path.join(event_dir, "*_mask.tiff"))
    if not mask_tiffs:
        return []
    mask_tiff = mask_tiffs[0]

    # 4. Read data using rasterio
    with rasterio.open(post_s2_tiff) as src_s2:
        s2_data = src_s2.read()  # (13, H, W)
        H, W = s2_data.shape[1], s2_data.shape[2]

    with rasterio.open(mask_tiff) as src_m:
        raw_mask = src_m.read(1)  # (H, W)

    cloud_data = None
    if cloud_tiff and os.path.exists(cloud_tiff):
        with rasterio.open(cloud_tiff) as src_c:
            cloud_data = src_c.read(1)

    # Selected 6 channels: B2 (idx 1), B3 (idx 2), B4 (idx 3), B8 (idx 7), B11 (idx 10), B12 (idx 11)
    s2_channels = s2_data[[1, 2, 3, 7, 10, 11], :, :]  # (6, H, W)
    validity_mask = s2_data[12, :, :]  # Band 13 is validity mask

    # Convert ground-truth to binary:
    # 0, 32 -> 0 (Unburned)
    # 64, 128, 192, 255 -> 1 (Burned)
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    burned_pixels = np.isin(raw_mask, [64, 128, 192, 255])
    binary_mask[burned_pixels] = 1

    patches = []

    # Slide window across (H, W)
    y_starts = list(range(0, H - patch_size + 1, stride))
    if (H - patch_size) % stride != 0 and H >= patch_size:
        y_starts.append(H - patch_size)

    x_starts = list(range(0, W - patch_size + 1, stride))
    if (W - patch_size) % stride != 0 and W >= patch_size:
        x_starts.append(W - patch_size)

    for y in y_starts:
        for x in x_starts:
            # Check validity fraction in patch
            val_patch = validity_mask[y:y+patch_size, x:x+patch_size]
            val_fraction = np.mean(val_patch >= 0.5)
            if val_fraction < 0.95:
                continue

            # Check cloud fraction if cloud mask present
            if cloud_data is not None:
                c_patch = cloud_data[y:y+patch_size, x:x+patch_size]
                # cloud values are typically > 0 for cloud
                cloud_fraction = np.mean(c_patch > 0.1)
                if cloud_fraction > 0.20:
                    continue

            img_patch = s2_channels[:, y:y+patch_size, x:x+patch_size]  # (6, 256, 256)
            m_patch = binary_mask[y:y+patch_size, x:x+patch_size][np.newaxis, :, :]  # (1, 256, 256)

            # Check for non-finite values (NaN / Inf)
            if not np.all(np.isfinite(img_patch)):
                # Clean NaNs if isolated
                img_patch = np.nan_to_num(img_patch, nan=0.0, posinf=1.0, neginf=0.0)

            burned_ratio = float(np.mean(m_patch == 1))
            patches.append({
                "image": img_patch.astype(np.float32),
                "mask": m_patch.astype(np.uint8),
                "burned_ratio": burned_ratio,
                "event": os.path.basename(event_dir),
                "coord": (y, x)
            })

    return patches

def run_extraction(project_root, dataset_root, sanity_check=False):
    csv_path = os.path.join(dataset_root, "satellite_data.csv")
    df = pd.read_csv(csv_path)

    event_dirs = find_event_dirs(dataset_root)
    print(f"Total extracted event directories available: {len(event_dirs)}")

    # Split definitions
    # TEST: fold = cyan (11 events)
    # VAL: fold = purple (8 events)
    # TRAIN: remaining available folds (coral, grey, lime, magenta, pink)
    test_events = df[df["fold"] == "cyan"]["folder"].tolist()
    val_events = df[df["fold"] == "purple"]["folder"].tolist()
    train_events = df[~df["fold"].isin(["cyan", "purple"])]["folder"].tolist()

    # Filter to events available on disk
    test_events = [e for e in test_events if e in event_dirs]
    val_events = [e for e in val_events if e in event_dirs]
    train_events = [e for e in train_events if e in event_dirs]

    print(f"Dataset Split Plan:")
    print(f"  Test events (cyan): {len(test_events)}")
    print(f"  Val events (purple): {len(val_events)}")
    print(f"  Train events: {len(train_events)}")

    out_dirs = {
        "train": os.path.join(project_root, "data", "processed", "train"),
        "val": os.path.join(project_root, "data", "processed", "val"),
        "test": os.path.join(project_root, "data", "processed", "test")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    if sanity_check:
        print("\n" + "="*50)
        print("RUNNING SANITY CHECK ON 2 TRAINING EVENTS")
        print("="*50)
        sample_events = train_events[:2]
        all_sample_patches = []
        for ev in sample_events:
            edir = event_dirs[ev]
            p = extract_patches_for_event(edir)
            print(f"Extracted {len(p)} candidate patches from {ev}")
            all_sample_patches.extend(p)

        print(f"Total candidate patches from 2 events: {len(all_sample_patches)}")
        # Verify 1st patch properties
        sample = all_sample_patches[0]
        img, m = sample["image"], sample["mask"]
        print(f"Image tensor shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask tensor shape: {m.shape}, dtype: {m.dtype}")
        print(f"Finite check: image is finite = {np.all(np.isfinite(img))}, mask is finite = {np.all(np.isfinite(m))}")
        print(f"Per-band min: {img.min(axis=(1,2))}")
        print(f"Per-band max: {img.max(axis=(1,2))}")
        print(f"Per-band mean: {img.mean(axis=(1,2))}")
        print(f"Mask unique values: {np.unique(m)}")
        return

    # FULL EXTRACTION
    random.seed(42)
    np.random.seed(42)

    # 1. PROCESS TRAINING SET
    print("\nProcessing TRAINING events...")
    train_patches_burned = []
    train_patches_background = []

    for idx, ev in enumerate(train_events):
        edir = event_dirs[ev]
        patches = extract_patches_for_event(edir)
        for p in patches:
            if p["burned_ratio"] >= 0.01:
                train_patches_burned.append(p)
            else:
                train_patches_background.append(p)
        print(f"[{idx+1}/{len(train_events)}] {ev[:25]}: total={len(patches)} (burned={len([p for p in patches if p['burned_ratio']>=0.01])})")

    print(f"\nRaw Training Patches: {len(train_patches_burned)} burned (>=1%), {len(train_patches_background)} background (<1%)")

    # Balanced Sampling: ~70% burned (>=1%), ~30% background (<1%)
    # If N burned patches, target total = N / 0.7, so background = N * 0.3 / 0.7
    num_burned = len(train_patches_burned)
    target_bg = int(num_burned * (30.0 / 70.0))
    if len(train_patches_background) > target_bg:
        selected_bg = random.sample(train_patches_background, target_bg)
    else:
        selected_bg = train_patches_background

    final_train_patches = train_patches_burned + selected_bg
    random.shuffle(final_train_patches)
    print(f"Final Balanced Training Set: {len(final_train_patches)} patches ({len(train_patches_burned)} burned, {len(selected_bg)} background)")

    # 2. COMPUTE NORMALIZATION STATISTICS STRICTLY FROM TRAINING DATA
    print("\nCalculating robust per-band normalization statistics from TRAINING set only...")
    # Sample up to 1000 patches for computing percentiles and accurate mean/std
    sample_size = min(len(final_train_patches), 1000)
    stat_samples = random.sample(final_train_patches, sample_size)
    stacked_images = np.stack([p["image"] for p in stat_samples], axis=0)  # (N, 6, 256, 256)

    # Channels: 0:B2, 1:B3, 2:B4, 3:B8, 4:B11, 5:B12
    channel_names = ["B2_Blue", "B3_Green", "B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2"]
    stats_dict = {
        "channel_names": channel_names,
        "mean": [float(stacked_images[:, c].mean()) for c in range(6)],
        "std": [float(stacked_images[:, c].std()) for c in range(6)],
        "min": [float(stacked_images[:, c].min()) for c in range(6)],
        "max": [float(stacked_images[:, c].max()) for c in range(6)],
        "p1": [float(np.percentile(stacked_images[:, c], 1)) for c in range(6)],
        "p99": [float(np.percentile(stacked_images[:, c], 99)) for c in range(6)]
    }

    stats_file = os.path.join(project_root, "data", "inspection", "s2_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved normalization statistics to: {stats_file}")
    for c in range(6):
        print(f"  {channel_names[c]:10s}: mean={stats_dict['mean'][c]:.4f}, std={stats_dict['std'][c]:.4f}, min={stats_dict['min'][c]:.4f}, max={stats_dict['max'][c]:.4f}")

    # Save training patches to disk
    print("\nSaving training patches to disk...")
    for idx, p in enumerate(final_train_patches):
        p_name = f"train_patch_{idx:05d}_{p['event']}.npz"
        np.savez_compressed(os.path.join(out_dirs["train"], p_name), image=p["image"], mask=p["mask"])

    # 3. PROCESS VALIDATION SET (Natural Distribution - No artificial balancing)
    print("\nProcessing VALIDATION events (natural distribution)...")
    val_patch_count = 0
    for idx, ev in enumerate(val_events):
        edir = event_dirs[ev]
        patches = extract_patches_for_event(edir)
        for p in patches:
            p_name = f"val_patch_{val_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["val"], p_name), image=p["image"], mask=p["mask"])
            val_patch_count += 1
        print(f"[{idx+1}/{len(val_events)}] {ev[:25]}: extracted {len(patches)} patches")
    print(f"Total Validation Patches: {val_patch_count}")

    # 4. PROCESS TEST SET (Natural Distribution - No artificial balancing)
    print("\nProcessing TEST events (natural distribution)...")
    test_patch_count = 0
    for idx, ev in enumerate(test_events):
        edir = event_dirs[ev]
        patches = extract_patches_for_event(edir)
        for p in patches:
            p_name = f"test_patch_{test_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["test"], p_name), image=p["image"], mask=p["mask"])
            test_patch_count += 1
        print(f"[{idx+1}/{len(test_events)}] {ev[:25]}: extracted {len(patches)} patches")
    print(f"Total Test Patches: {test_patch_count}")

    print("\n" + "="*50)
    print("PATCH EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"  Train Patches: {len(final_train_patches)}")
    print(f"  Val Patches:   {val_patch_count}")
    print(f"  Test Patches:  {test_patch_count}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_check", action="store_true", help="Run sanity check on 2 events")
    args = parser.parse_args()

    project_root = r"C:\Users\Sahil\OneDrive\Desktop\AI project\satellite_wildfire_project"
    dataset_root = r"C:\Users\Sahil\OneDrive\Desktop\AI project\dataset\Zenodo"
    run_extraction(project_root, dataset_root, sanity_check=args.sanity_check)
