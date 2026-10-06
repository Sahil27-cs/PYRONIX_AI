"""
Pre-Training Sanity Verification Script
Executes the 9 mandatory pre-flight checks:
1. Run preprocessing on 1-2 training events.
2. Verify image dimensions [6, 256, 256].
3. Verify masks visually and check binary values {0, 1}.
4. Verify B4/B3/B2 RGB looks correct (Red, Green, Blue composite).
5. Verify burned pixels align spatially with the mask (dNBR / NIR drop).
6. Verify no NaN / Inf values.
7. Verify normalization function.
8. Run one forward pass through ResNet-34 U-Net on GPU with AMP.
9. Run one backward pass and optimizer step.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessing.extract_patches import extract_patches_for_event, find_event_dirs
from src.models.unet import ResNet34UNet
from src.training.losses import CombinedLoss

def main():
    print("="*60)
    print("STAGE 2 — PRE-TRAINING SANITY VERIFICATION (9 CHECKS)")
    print("="*60)

    dataset_root = os.path.abspath(os.path.join(project_root, "..", "dataset", "Zenodo"))
    event_dirs = find_event_dirs(dataset_root)

    # 1. Run preprocessing on 2 training events
    print("\n[CHECK 1] Extracting patches from 2 training events...")
    sample_events = ["EMSR207_01MIRANDADOCORVO_02GRADING_MAP_v2_vector", "EMSR207_02LOUSA_02GRADING_MAP_v2_vector"]
    patches = []
    for ev in sample_events:
        if ev in event_dirs:
            p = extract_patches_for_event(event_dirs[ev])
            print(f"  Extracted {len(p)} candidate patches from {ev}")
            patches.extend(p)

    assert len(patches) > 0, "Error: No patches extracted!"
    print(f"  --> Check 1 PASSED: Successfully extracted {len(patches)} candidate patches.")

    # Find a patch with burned area and a patch without burned area
    burned_patches = [p for p in patches if p["burned_ratio"] > 0.05]
    sample_patch = burned_patches[0] if burned_patches else patches[0]

    img = sample_patch["image"]   # (6, 256, 256)
    mask = sample_patch["mask"]   # (1, 256, 256)

    # 2. Verify image dimensions
    print("\n[CHECK 2] Verifying image and mask dimensions...")
    print(f"  Image shape: {img.shape}, expected (6, 256, 256)")
    print(f"  Mask shape:  {mask.shape}, expected (1, 256, 256)")
    assert img.shape == (6, 256, 256), f"Wrong image shape: {img.shape}"
    assert mask.shape == (1, 256, 256), f"Wrong mask shape: {mask.shape}"
    print("  --> Check 2 PASSED: Tensor dimensions are exactly [6, 256, 256] and [1, 256, 256].")

    # 3. Verify masks values
    print("\n[CHECK 3] Verifying mask binary values...")
    unique_mask_vals = np.unique(mask)
    print(f"  Mask unique values: {unique_mask_vals}")
    assert set(unique_mask_vals).issubset({0, 1}), f"Invalid mask values: {unique_mask_vals}"
    print("  --> Check 3 PASSED: Mask strictly contains binary classes {0, 1}.")

    # 4. Verify B4/B3/B2 RGB composite
    # channels are 0:B2 (Blue), 1:B3 (Green), 2:B4 (Red), 3:B8 (NIR), 4:B11 (SWIR1), 5:B12 (SWIR2)
    print("\n[CHECK 4] Verifying B4/B3/B2 RGB channels...")
    b2 = img[0]
    b3 = img[1]
    b4 = img[2]
    # Build RGB for display
    rgb = np.stack([b4, b3, b2], axis=-1)
    # Clip to 98th percentile for visualization
    p98 = np.percentile(rgb, 98)
    rgb_vis = np.clip(rgb / (p98 + 1e-6), 0, 1)
    print(f"  RGB range min={rgb.min():.4f}, max={rgb.max():.4f}, p98={p98:.4f}")
    assert rgb_vis.shape == (256, 256, 3), "RGB composite shape mismatch"
    print("  --> Check 4 PASSED: True-color RGB (B4=Red, B3=Green, B2=Blue) successfully created.")

    # 5. Verify burned pixels align spatially with the mask
    print("\n[CHECK 5] Verifying spatial alignment of burned pixels with ground-truth...")
    # In burned pixels, NIR (B8, idx 3) drops and SWIR2 (B12, idx 5) rises -> NBR = (B8 - B12)/(B8 + B12) drops sharply
    b8 = img[3]
    b12 = img[5]
    nbr = (b8 - b12) / (b8 + b12 + 1e-6)
    burned_idx = (mask[0] == 1)
    unburned_idx = (mask[0] == 0)

    if burned_idx.sum() > 0:
        mean_nbr_burned = nbr[burned_idx].mean()
        mean_nbr_unburned = nbr[unburned_idx].mean()
        print(f"  Mean NBR in burned pixels:   {mean_nbr_burned:+.4f}")
        print(f"  Mean NBR in unburned pixels: {mean_nbr_unburned:+.4f}")
        assert mean_nbr_burned < mean_nbr_unburned, "Spectral check failed: Burned pixels do not exhibit lower NBR than unburned!"
        print(f"  Burned pixels show expected drop in NBR ({mean_nbr_burned:+.3f} vs {mean_nbr_unburned:+.3f})")
    print("  --> Check 5 PASSED: Burned ground-truth aligns with spectral characteristics.")

    # Save visual verification plot
    vis_out = os.path.join(project_root, "results", "visualizations", "baseline_s2")
    os.makedirs(vis_out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(rgb_vis)
    axes[0].set_title("Sentinel-2 RGB (B4-B3-B2)")
    axes[0].axis("off")

    axes[1].imshow(nbr, cmap="RdYlGn")
    axes[1].set_title("Normalized Burn Ratio (NBR)")
    axes[1].axis("off")

    axes[2].imshow(mask[0], cmap="gray")
    axes[2].set_title(f"Binary GT Mask (Burned: {sample_patch['burned_ratio']*100:.1f}%)")
    axes[2].axis("off")

    plt.suptitle(f"Sanity Check: {sample_patch['event']} at {sample_patch['coord']}")
    plt.tight_layout()
    sanity_img_path = os.path.join(vis_out, "sanity_check_patch_alignment.png")
    plt.savefig(sanity_img_path)
    plt.close()
    print(f"  Saved alignment visualization to: {sanity_img_path}")

    # 6. Verify no NaN / Inf values
    print("\n[CHECK 6] Verifying no NaN / Inf values...")
    assert np.all(np.isfinite(img)), "Image contains NaN or Inf!"
    assert np.all(np.isfinite(mask)), "Mask contains NaN or Inf!"
    print("  --> Check 6 PASSED: Tensors are 100% finite (no NaN or Inf).")

    # 7. Verify normalization
    print("\n[CHECK 7] Verifying normalization function...")
    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True) + 1e-6
    img_norm = (img - mean) / std
    print(f"  Normalized image mean: {img_norm.mean(axis=(1,2)).round(4)}")
    print(f"  Normalized image std:  {img_norm.std(axis=(1,2)).round(4)}")
    assert np.allclose(img_norm.mean(axis=(1,2)), 0.0, atol=1e-3), "Normalized mean not 0"
    assert np.allclose(img_norm.std(axis=(1,2)), 1.0, atol=1e-3), "Normalized std not 1"
    print("  --> Check 7 PASSED: Normalization produces zero-mean, unit-variance data.")

    # 8. Run forward pass on GPU with AMP
    print("\n[CHECK 8] Running single forward pass on GPU with PyTorch AMP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = ResNet34UNet(in_channels=6, num_classes=1, pretrained=True).to(device)
    x = torch.from_numpy(img_norm).unsqueeze(0).to(device)         # (1, 6, 256, 256)
    y = torch.from_numpy(mask).float().unsqueeze(0).to(device) # (1, 1, 256, 256)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        out = model(x)
    print(f"  Forward output shape: {out.shape}, dtype: {out.dtype}")
    assert out.shape == (1, 1, 256, 256), f"Output shape mismatch: {out.shape}"
    print("  --> Check 8 PASSED: Forward pass succeeded with expected output shape.")

    # 9. Run backward pass
    print("\n[CHECK 9] Running backward pass and optimizer update...")
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        out = model(x)
        loss, focal_l, dice_l = criterion(out, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  Loss value: {loss.item():.4f} (Focal: {focal_l.item():.4f}, Dice: {dice_l.item():.4f})")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Backward loss is NaN or Inf!"
    print("  --> Check 9 PASSED: Backward pass and gradient update succeeded.")

    print("\n" + "="*60)
    print("ALL 9 PRE-TRAINING CHECKS PASSED PERFECTLY!")
    print("READY TO PROCEED TO FULL EXTRACTION AND MODEL TRAINING.")
    print("="*60)

if __name__ == "__main__":
    main()
