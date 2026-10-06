"""
Generate Test Sample Visualizations from Best Trained Checkpoint
Produces 5-panel figures:
1. Sentinel-2 RGB composite (B4/B3/B2)
2. Ground-truth binary mask
3. Predicted probability map (sigmoid)
4. Predicted binary mask (threshold 0.5)
5. Overlay (Yellow=TP, Red=FP, Green=FN)
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import WildfirePatchDataset
from src.models.unet import ResNet34UNet
from src.training.trainer import generate_visualizations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = os.path.join(project_root, "models", "best_s2_baseline_model.pt")
    stats_file = os.path.join(project_root, "data", "inspection", "s2_statistics.json")
    test_dir = os.path.join(project_root, "data", "processed", "test")
    vis_dir = os.path.join(project_root, "results", "visualizations", "baseline_s2")

    print(f"Loading best model checkpoint from {model_path}...")
    ckpt = torch.load(model_path, weights_only=False)
    model = ResNet34UNet(in_channels=6, num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = WildfirePatchDataset(test_dir, stats_json=stats_file, augment=False)
    print(f"Total test patches: {len(test_dataset)}")

    generate_visualizations(model, test_dataset, stats_file, vis_dir, device, num_samples=10)
    print(f"\nSuccessfully generated 10 test prediction visualizations in: {vis_dir}")

if __name__ == "__main__":
    main()
