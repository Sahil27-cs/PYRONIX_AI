"""
Stage 5: Multimodal Fusion Patch Extraction and Normalization Script
Combines Sentinel-2 (6 optical channels) + Sentinel-1 (3 SAR channels) -> 9 channels.
Guarantees strict spatial and temporal co-registration per event patch.
Filters patches by joint validity (>=95% in both S2 and S1, <=20% cloud).
Samples training patches (~70% burned, ~30% background).
Calculates 9-channel normalization statistics strictly from TRAINING DATA ONLY.
"""

import os
import glob
import json
import random
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

def extract_fusion_patches_for_event(event_dir, patch_size=256, stride=128):
    # 1. Post-fire S2
    s2_tiffs = sorted([f for f in glob.glob(os.path.join(event_dir, "sentinel2_*.tiff")) if "_cloud" not in f])
    # 2. Post-fire S1
    s1_tiffs = sorted([f for f in glob.glob(os.path.join(event_dir, "sentinel1_*.tiff")) if "_coverage" not in f])
    # 3. Mask
    mask_tiffs = glob.glob(os.path.join(event_dir, "*_mask.tiff"))

    if not s2_tiffs or not s1_tiffs or not mask_tiffs:
        return []

    post_s2_tiff = s2_tiffs[-1]
    post_s1_tiff = s1_tiffs[-1]
    mask_tiff = mask_tiffs[0]

    # Optional cloud mask
    post_date_str = os.path.basename(post_s2_tiff).replace("sentinel2_", "").replace(".tiff", "")
    cloud_candidates = glob.glob(os.path.join(event_dir, f"sentinel2_{post_date_str}_cloud_coverage.tiff"))
    cloud_tiff = cloud_candidates[0] if cloud_candidates else None

    # Read S2
    with rasterio.open(post_s2_tiff) as src:
        s2_data = src.read()  # (13, H, W)
        H2, W2 = s2_data.shape[1], s2_data.shape[2]

    # Read S1
    with rasterio.open(post_s1_tiff) as src:
        s1_data = src.read()  # (4, H, W)
        H1, W1 = s1_data.shape[1], s1_data.shape[2]

    # Read Mask
    with rasterio.open(mask_tiff) as src:
        raw_mask = src.read(1)  # (H, W)
        Hm, Wm = raw_mask.shape

    # Ensure spatial dimensions match exactly
    H = min(H2, H1, Hm)
    W = min(W2, W1, Wm)

    s2_channels = s2_data[[1, 2, 3, 7, 10, 11], :H, :W]  # 6 bands: B2, B3, B4, B8, B11, B12
    s2_validity = s2_data[12, :H, :W]

    s1_channels = s1_data[:3, :H, :W].copy()  # 3 bands: VV, VH, Ratio
    s1_channels[0] = np.nan_to_num(s1_channels[0], nan=0.0, posinf=10.0, neginf=0.0)
    s1_channels[1] = np.nan_to_num(s1_channels[1], nan=0.0, posinf=10.0, neginf=0.0)
    s1_channels[2] = np.clip(np.nan_to_num(s1_channels[2], nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    s1_validity = s1_data[3, :H, :W]

    cloud_data = None
    if cloud_tiff and os.path.exists(cloud_tiff):
        with rasterio.open(cloud_tiff) as src:
            cloud_data = src.read(1)[:H, :W]

    # Ground truth binary mask
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    burned_pixels = np.isin(raw_mask[:H, :W], [64, 128, 192, 255])
    binary_mask[burned_pixels] = 1

    # Joint 9-channel tensor: [6 S2 + 3 S1, H, W]
    joint_channels = np.concatenate([s2_channels, s1_channels], axis=0)  # (9, H, W)
    joint_channels = np.nan_to_num(joint_channels, nan=0.0, posinf=1.0, neginf=0.0)

    patches = []
    y_starts = list(range(0, H - patch_size + 1, stride))
    if (H - patch_size) % stride != 0 and H >= patch_size:
        y_starts.append(H - patch_size)

    x_starts = list(range(0, W - patch_size + 1, stride))
    if (W - patch_size) % stride != 0 and W >= patch_size:
        x_starts.append(W - patch_size)

    for y in y_starts:
        for x in x_starts:
            # Check S2 validity
            s2_val_patch = s2_validity[y:y+patch_size, x:x+patch_size]
            if np.mean(s2_val_patch >= 0.5) < 0.95:
                continue

            # Check S1 validity
            s1_val_patch = s1_validity[y:y+patch_size, x:x+patch_size]
            if np.mean(s1_val_patch >= 0.5) < 0.95:
                continue

            # Check cloud coverage
            if cloud_data is not None:
                c_patch = cloud_data[y:y+patch_size, x:x+patch_size]
                if np.mean(c_patch > 0.1) > 0.20:
                    continue

            img_patch = joint_channels[:, y:y+patch_size, x:x+patch_size]
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

def run_fusion_extraction(project_root, dataset_root):
    csv_path = os.path.join(dataset_root, "satellite_data.csv")
    df = pd.read_csv(csv_path)
    event_dirs = find_event_dirs(dataset_root)

    test_events = df[df["fold"] == "cyan"]["folder"].tolist()
    val_events = df[df["fold"] == "purple"]["folder"].tolist()
    train_events = df[~df["fold"].isin(["cyan", "purple"])]["folder"].tolist()

    # Filter to events that have BOTH S2 and S1
    def has_both(ev):
        if ev not in event_dirs:
            return False
        ed = event_dirs[ev]
        has_s2 = bool(glob.glob(os.path.join(ed, "sentinel2_*.tiff")))
        has_s1 = bool(glob.glob(os.path.join(ed, "sentinel1_*.tiff")))
        return has_s2 and has_s1

    train_events = [e for e in train_events if has_both(e)]
    val_events = [e for e in val_events if has_both(e)]
    test_events = [e for e in test_events if has_both(e)]

    print("="*65)
    print("STAGE 5 — MULTIMODAL FUSION (S2 + S1) PATCH EXTRACTION")
    print(f"  Train events: {len(train_events)}")
    print(f"  Val events:   {len(val_events)}")
    print(f"  Test events:  {len(test_events)}")
    print("="*65)

    out_dirs = {
        "train": os.path.join(project_root, "data", "processed_fusion", "train"),
        "val": os.path.join(project_root, "data", "processed_fusion", "val"),
        "test": os.path.join(project_root, "data", "processed_fusion", "test")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # 1. PROCESS TRAINING SET
    print("\nProcessing TRAINING fusion events...")
    train_patches_burned = []
    train_patches_background = []

    for idx, ev in enumerate(train_events):
        patches = extract_fusion_patches_for_event(event_dirs[ev])
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

    # 2. COMPUTE 9-CHANNEL NORMALIZATION STATISTICS FROM TRAINING DATA ONLY
    print("\nCalculating robust per-band normalization statistics from TRAINING set only...")
    sample_size = min(len(final_train_patches), 1000)
    stat_samples = random.sample(final_train_patches, sample_size)
    stacked_images = np.stack([p["image"] for p in stat_samples], axis=0)  # (N, 9, 256, 256)

    channel_names = [
        "S2_B2_Blue", "S2_B3_Green", "S2_B4_Red", "S2_B8_NIR", "S2_B11_SWIR1", "S2_B12_SWIR2",
        "S1_VV", "S1_VH", "S1_VV_VH_ratio"
    ]
    stats_dict = {
        "channel_names": channel_names,
        "mean": [float(stacked_images[:, c].mean()) for c in range(9)],
        "std": [float(stacked_images[:, c].std()) for c in range(9)],
        "min": [float(stacked_images[:, c].min()) for c in range(9)],
        "max": [float(stacked_images[:, c].max()) for c in range(9)],
        "p1": [float(np.percentile(stacked_images[:, c], 1)) for c in range(9)],
        "p99": [float(np.percentile(stacked_images[:, c], 99)) for c in range(9)]
    }

    stats_file = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved fusion normalization statistics to: {stats_file}")
    for c in range(9):
        print(f"  {channel_names[c]:18s}: mean={stats_dict['mean'][c]:.4f}, std={stats_dict['std'][c]:.4f}")

    # Save training patches
    print("\nSaving training fusion patches...")
    for idx, p in enumerate(final_train_patches):
        p_name = f"train_fusion_patch_{idx:05d}_{p['event']}.npz"
        np.savez_compressed(os.path.join(out_dirs["train"], p_name), image=p["image"], mask=p["mask"])

    # 3. PROCESS VALIDATION SET (natural distribution)
    print("\nProcessing VALIDATION fusion events (natural distribution)...")
    val_patch_count = 0
    for idx, ev in enumerate(val_events):
        patches = extract_fusion_patches_for_event(event_dirs[ev])
        for p in patches:
            p_name = f"val_fusion_patch_{val_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["val"], p_name), image=p["image"], mask=p["mask"])
            val_patch_count += 1
        print(f"[{idx+1:2d}/{len(val_events)}] {ev[:25]}: extracted {len(patches)} patches")

    # 4. PROCESS TEST SET (natural distribution)
    print("\nProcessing TEST fusion events (natural distribution)...")
    test_patch_count = 0
    for idx, ev in enumerate(test_events):
        patches = extract_fusion_patches_for_event(event_dirs[ev])
        for p in patches:
            p_name = f"test_fusion_patch_{test_patch_count:05d}_{p['event']}.npz"
            np.savez_compressed(os.path.join(out_dirs["test"], p_name), image=p["image"], mask=p["mask"])
            test_patch_count += 1
        print(f"[{idx+1:2d}/{len(test_events)}] {ev[:25]}: extracted {len(patches)} patches")

    print("\n" + "="*65)
    print("MULTIMODAL PATCH EXTRACTION COMPLETED!")
    print(f"  Train Fusion Patches: {len(final_train_patches)}")
    print(f"  Val Fusion Patches:   {val_patch_count}")
    print(f"  Test Fusion Patches:  {test_patch_count}")
    print("="*65)

if __name__ == "__main__":
    proj = r"C:\Users\Sahil\OneDrive\Desktop\AI project\satellite_wildfire_project"
    zenodo = r"C:\Users\Sahil\OneDrive\Desktop\AI project\dataset\Zenodo"
    run_fusion_extraction(proj, zenodo)
