"""
Full Dataset Statistics Collector for Phase 3 and Phase 5 Reports
"""

import os
import glob
import json
import rasterio
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def analyze():
    project_root = r"C:\Users\Sahil\OneDrive\Desktop\AI project\satellite_wildfire_project"
    dataset_zenodo = r"C:\Users\Sahil\OneDrive\Desktop\AI project\dataset\Zenodo"
    csv_path = os.path.join(dataset_zenodo, "satellite_data.csv")

    df_csv = pd.read_csv(csv_path)

    # 1. File breakdown
    total_files = 0
    total_bytes = 0
    ext_counter = Counter()
    file_type_counter = Counter()

    event_dirs = {}
    for part in ['Satellite_burned_area_dataset_part1', 'Satellite_burned_area_dataset_part3', 'Satellite_burned_area_dataset_part4', 'Satellite_burned_area_dataset_part5']:
        inner = os.path.join(dataset_zenodo, part, part)
        if os.path.exists(inner):
            for e in os.listdir(inner):
                ep = os.path.join(inner, e)
                if os.path.isdir(ep):
                    event_dirs[e] = ep

    for root, dirs, files in os.walk(dataset_zenodo):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_bytes += sz
            total_files += 1
            ext = os.path.splitext(f)[1].lower()
            ext_counter[ext] += 1

            # Categorize
            if f.endswith('_mask.tiff'):
                file_type_counter['mask_tiff'] += 1
            elif f.endswith('_mask.png'):
                file_type_counter['mask_png'] += 1
            elif f.startswith('sentinel1_') and f.endswith('_coverage.png'):
                file_type_counter['s1_coverage_png'] += 1
            elif f.startswith('sentinel1_') and f.endswith('.tiff'):
                file_type_counter['s1_tiff'] += 1
            elif f.startswith('sentinel1_') and f.endswith('.png'):
                file_type_counter['s1_png'] += 1
            elif f.startswith('sentinel2_') and f.endswith('_cloud_coverage.tiff'):
                file_type_counter['s2_cloud_coverage_tiff'] += 1
            elif f.startswith('sentinel2_') and f.endswith('_cloud_coverage.png'):
                file_type_counter['s2_cloud_coverage_png'] += 1
            elif f.startswith('sentinel2_') and f.endswith('_coverage.png'):
                file_type_counter['s2_coverage_png'] += 1
            elif f.startswith('sentinel2_') and f.endswith('.tiff'):
                file_type_counter['s2_tiff'] += 1
            elif f.startswith('sentinel2_') and f.endswith('.png'):
                file_type_counter['s2_png'] += 1
            elif f == 'satellite_data.csv':
                file_type_counter['csv'] += 1
            elif f.endswith('.crdownload'):
                file_type_counter['incomplete_download'] += 1
            else:
                file_type_counter['other'] += 1

    # 2. Dimensions and resolution analysis
    dimensions = []
    resolutions = []
    crss = set()
    s1_counts_per_event = []
    s2_counts_per_event = []

    s1_min_vals = []
    s1_max_vals = []
    s2_min_vals = []
    s2_max_vals = []

    for ev_id, ev_path in event_dirs.items():
        s1_tiffs = sorted(glob.glob(os.path.join(ev_path, 'sentinel1_*.tiff')))
        s2_tiffs = sorted([f for f in glob.glob(os.path.join(ev_path, 'sentinel2_*.tiff')) if '_cloud' not in f])
        mask_tiffs = glob.glob(os.path.join(ev_path, '*_mask.tiff'))

        s1_counts_per_event.append(len(s1_tiffs))
        s2_counts_per_event.append(len(s2_tiffs))

        if mask_tiffs:
            with rasterio.open(mask_tiffs[0]) as src:
                dimensions.append((src.height, src.width))

        if s2_tiffs:
            with rasterio.open(s2_tiffs[0]) as src:
                resolutions.append(src.res)
                crss.add(str(src.crs))

    # Min/max dimensions
    heights = [d[0] for d in dimensions]
    widths = [d[1] for d in dimensions]

    # Folds analysis
    df_csv['present_on_disk'] = df_csv['folder'].apply(lambda x: x in event_dirs)
    fold_summary = df_csv.groupby(['fold', 'present_on_disk']).size().unstack(fill_value=0)

    stats = {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024**3),
        "ext_counter": dict(ext_counter),
        "file_type_counter": dict(file_type_counter),
        "total_events_csv": len(df_csv),
        "total_events_on_disk": len(event_dirs),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": float(np.mean(heights)),
        "median_height": float(np.median(heights)),
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "common_dimensions": Counter(dimensions).most_common(5),
        "crs": list(crss),
        "sample_res": resolutions[0] if resolutions else None,
        "s1_per_event_dist": dict(Counter(s1_counts_per_event)),
        "s2_per_event_dist": dict(Counter(s2_counts_per_event)),
        "fold_summary": fold_summary.to_dict()
    }

    print(json.dumps(stats, indent=2))
    with open(os.path.join(project_root, "data", "inspection", "stats_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    analyze()
