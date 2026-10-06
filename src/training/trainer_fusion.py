"""
Stage 5: Multimodal Fusion (Sentinel-1 SAR + Sentinel-2 Optical) Training Engine
9 Channels: [B2, B3, B4, B8, B11, B12, VV, VH, Ratio] -> ResNet-34 U-Net -> Binary Burned Area
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import WildfirePatchDataset
from src.models.unet import ResNet34UNet
from src.training.losses import CombinedLoss
from src.evaluation.metrics import MetricTracker

def test_fusion_batch_size(model, device, patch_dir, stats_file, candidate_batch_sizes=[4, 8, 16]):
    print("\n" + "="*50)
    print("TESTING FUSION BATCH SIZES FOR 6 GB VRAM...")
    print("="*50)
    best_bs = 4
    sample_dataset = WildfirePatchDataset(patch_dir, stats_json=stats_file, augment=False)

    for bs in candidate_batch_sizes:
        if bs > len(sample_dataset):
            continue
        try:
            torch.cuda.empty_cache()
            loader = DataLoader(sample_dataset, batch_size=bs, shuffle=True, num_workers=0)
            scaler = torch.amp.GradScaler('cuda')
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = CombinedLoss()

            step_count = 0
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_count += 1
                if step_count >= 2:
                    break

            vram_used = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"  Batch size {bs:2d}: STABLE! (Peak VRAM: {vram_used:.2f} GB)")
            best_bs = bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  Batch size {bs:2d}: CUDA OOM! Reverting to previous stable batch size.")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    print(f"--> Selected Largest Stable Batch Size: {best_bs}")
    return best_bs

def train_one_epoch_fusion(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_focal, total_dice = 0.0, 0.0, 0.0
    metric_tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss, focal_l, dice_l = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_focal += focal_l.item() * images.size(0)
        total_dice += dice_l.item() * images.size(0)
        metric_tracker.update(outputs, masks)

    n = len(loader.dataset)
    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / n
    metrics["focal_loss"] = total_focal / n
    metrics["dice_loss"] = total_dice / n
    return metrics

def evaluate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss, total_focal, total_dice = 0.0, 0.0, 0.0
    metric_tracker = MetricTracker()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss, focal_l, dice_l = criterion(outputs, masks)

            total_loss += loss.item() * images.size(0)
            total_focal += focal_l.item() * images.size(0)
            total_dice += dice_l.item() * images.size(0)
            metric_tracker.update(outputs, masks)

    n = len(loader.dataset)
    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / n
    metrics["focal_loss"] = total_focal / n
    metrics["dice_loss"] = total_dice / n
    return metrics

def generate_fusion_visualizations(model, test_dataset, stats_file, out_dir, device, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with open(stats_file, "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"]).reshape(9, 1, 1)
    std = np.array(stats["std"]).reshape(9, 1, 1)

    sample_indices = []
    for idx in range(len(test_dataset)):
        file_path = test_dataset.patch_files[idx]
        with np.load(file_path) as d:
            m = d["mask"]
            if np.mean(m == 1) > 0.01:
                sample_indices.append(idx)

    if len(sample_indices) < num_samples:
        remaining = [i for i in range(len(test_dataset)) if i not in sample_indices]
        sample_indices.extend(remaining[:num_samples - len(sample_indices)])
    else:
        sample_indices = sample_indices[:num_samples]

    print(f"\nGenerating {len(sample_indices)} test Multimodal Fusion prediction visualizations...")

    for i, idx in enumerate(sample_indices):
        file_path = test_dataset.patch_files[idx]
        with np.load(file_path) as d:
            raw_img = d["image"].astype(np.float32)  # (9, 256, 256): 6 S2 + 3 S1
            raw_mask = d["mask"].astype(np.uint8)[0]

        norm_img = ((raw_img - mean) / std).astype(np.float32)
        input_t = torch.from_numpy(norm_img).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logit = model(input_t)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                pred_binary = (prob >= 0.5).astype(np.uint8)

        # S2 RGB (B4, B3, B2)
        s2_rgb = np.stack([raw_img[2], raw_img[1], raw_img[0]], axis=-1)
        p98_rgb = np.percentile(s2_rgb, 98) + 1e-6
        s2_vis = np.clip(s2_rgb / p98_rgb, 0, 1)

        # S1 SAR Composite (VV, VH, Ratio)
        s1_sar = np.stack([raw_img[6], raw_img[7], raw_img[8]], axis=-1)
        s1_vis = np.zeros_like(s1_sar)
        for c in range(3):
            p98_s = np.percentile(s1_sar[:, :, c], 98) + 1e-6
            s1_vis[:, :, c] = np.clip(s1_sar[:, :, c] / p98_s, 0, 1)

        # Overlay: Yellow=TP, Red=FP, Green=FN
        overlay = s2_vis.copy()
        tp_mask = (pred_binary == 1) & (raw_mask == 1)
        fp_mask = (pred_binary == 1) & (raw_mask == 0)
        fn_mask = (pred_binary == 0) & (raw_mask == 1)

        overlay[tp_mask] = [1.0, 0.85, 0.0]
        overlay[fp_mask] = [1.0, 0.2, 0.2]
        overlay[fn_mask] = [0.2, 0.9, 0.2]

        fig, axes = plt.subplots(1, 6, figsize=(24, 4), dpi=150)
        axes[0].imshow(s2_vis)
        axes[0].set_title("1. S2 Optical RGB (B4-B3-B2)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(s1_vis)
        axes[1].set_title("2. S1 SAR (VV-VH-Ratio)", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(raw_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"3. Ground Truth ({np.mean(raw_mask)*100:.1f}% Burned)", fontsize=10)
        axes[2].axis("off")

        axes[3].imshow(prob, cmap="inferno", vmin=0, vmax=1)
        axes[3].set_title("4. Fusion Probability Map", fontsize=10)
        axes[3].axis("off")

        axes[4].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title(f"5. Fusion Prediction ({np.mean(pred_binary)*100:.1f}%)", fontsize=10)
        axes[4].axis("off")

        axes[5].imshow(overlay)
        axes[5].set_title("6. Overlay (Yellow=TP, Red=FP, Green=FN)", fontsize=10)
        axes[5].axis("off")

        patch_base = os.path.basename(file_path).replace(".npz", "")
        plt.suptitle(f"Multimodal Fusion Sample #{i+1:02d}: {patch_base}", fontsize=11, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"test_pred_fusion_{i+1:02d}_{patch_base}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.basename(save_path)}")

def run_fusion_training_experiment():
    print("="*65)
    print("STAGE 5 — MULTIMODAL FUSION TRAINING & EVALUATION PIPELINE")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = os.path.join(project_root, "data", "processed_fusion", "train")
    val_dir = os.path.join(project_root, "data", "processed_fusion", "val")
    test_dir = os.path.join(project_root, "data", "processed_fusion", "test")
    stats_file = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    model_save_dir = os.path.join(project_root, "models")
    metrics_save_dir = os.path.join(project_root, "results", "metrics")
    vis_save_dir = os.path.join(project_root, "results", "visualizations", "fusion")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    train_dataset = WildfirePatchDataset(train_dir, stats_json=stats_file, augment=True)
    val_dataset = WildfirePatchDataset(val_dir, stats_json=stats_file, augment=False)
    test_dataset = WildfirePatchDataset(test_dir, stats_json=stats_file, augment=False)

    print(f"Multimodal Fusion Dataset Summary:")
    print(f"  Train Fusion patches: {len(train_dataset)}")
    print(f"  Val Fusion patches:   {len(val_dataset)}")
    print(f"  Test Fusion patches:  {len(test_dataset)}")

    # 9-channel ResNet-34 U-Net
    model = ResNet34UNet(in_channels=9, num_classes=1, pretrained=True).to(device)

    # Discover optimal batch size
    batch_size = test_fusion_batch_size(model, device, train_dir, stats_file, candidate_batch_sizes=[4, 8, 16])

    # Re-initialize clean model
    model = ResNet34UNet(in_channels=9, num_classes=1, pretrained=True).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    criterion = CombinedLoss(alpha=0.75, gamma=2.0, smooth=1.0, focal_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 30
    patience = 7
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    history = []
    start_time = time.time()

    print("\n" + "="*65)
    print(f"STARTING FUSION TRAINING: {epochs} EPOCHS (Early Stopping Patience: {patience})")
    print(f"Batch Size: {batch_size} | Optimizer: AdamW (lr=1e-4, wd=1e-4)")
    print("="*65)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_m = train_one_epoch_fusion(model, train_loader, optimizer, criterion, scaler, device)
        val_m = evaluate_fusion(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_start

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_m["dice"])

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_m["loss"],
            "train_iou": train_m["iou"],
            "train_dice": train_m["dice"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "val_loss": val_m["loss"],
            "val_iou": val_m["iou"],
            "val_dice": val_m["dice"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_accuracy": val_m["accuracy"],
            "time_sec": ep_duration
        }
        history.append(record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:4.1f}s) | "
              f"Train Loss: {train_m['loss']:.4f} (Dice: {train_m['dice']:.4f}, IoU: {train_m['iou']:.4f}) | "
              f"Val Loss: {val_m['loss']:.4f} (Dice: {val_m['dice']:.4f}, IoU: {val_m['iou']:.4f})")

        if val_m["dice"] > best_val_dice:
            best_val_dice = val_m["dice"]
            best_epoch = epoch
            patience_counter = 0

            with open(stats_file, "r") as f:
                norm_stats = json.load(f)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": val_m,
                "normalization_statistics": norm_stats,
                "config": {
                    "architecture": "ResNet34UNet",
                    "in_channels": 9,
                    "channels": [
                        "S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12",
                        "S1_VV", "S1_VH", "S1_VV_VH_ratio"
                    ],
                    "patch_size": 256,
                    "stride": 128,
                    "batch_size": batch_size,
                    "initial_lr": 1e-4,
                    "weight_decay": 1e-4,
                    "loss": "0.5*Focal + 0.5*Dice"
                }
            }
            ckpt_path = os.path.join(model_save_dir, "best_fusion_model.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"  ==> Saved new best Fusion model checkpoint! Val Dice: {best_val_dice:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] No validation Dice improvement for {patience} epochs.")
                break

    total_training_time = time.time() - start_time
    print(f"\nFusion Training completed in {total_training_time/60:.2f} minutes.")

    # Save History CSV
    df_history = pd.DataFrame(history)
    history_csv = os.path.join(metrics_save_dir, "fusion_history.csv")
    df_history.to_csv(history_csv, index=False)
    print(f"Saved Fusion training history to: {history_csv}")

    # Load Best Model for Final Test Evaluation
    print("\n" + "="*65)
    print("EVALUATING BEST FUSION CHECKPOINT ON UNSEEN TEST SET (fold = cyan, 10 events)")
    print("="*65)
    best_ckpt = torch.load(os.path.join(model_save_dir, "best_fusion_model.pt"), weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_fusion(model, test_loader, criterion, device)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    print("\nFINAL MULTIMODAL FUSION TEST PERFORMANCE METRICS (Globally Aggregated):")
    print(f"  Test IoU:              {test_metrics['iou']*100:.2f}%")
    print(f"  Test Dice / F1:        {test_metrics['dice']*100:.2f}%")
    print(f"  Test Precision:        {test_metrics['precision']*100:.2f}%")
    print(f"  Test Recall:           {test_metrics['recall']*100:.2f}%")
    print(f"  Test Pixel Accuracy:   {test_metrics['accuracy']*100:.2f}%")
    print(f"  True Positives (TP):   {test_metrics['tp']:,}")
    print(f"  False Positives (FP):  {test_metrics['fp']:,}")
    print(f"  True Negatives (TN):   {test_metrics['tn']:,}")
    print(f"  False Negatives (FN):  {test_metrics['fn']:,}")
    print(f"  Total Evaluated Pixels:{test_metrics['total_pixels']:,}")
    print(f"  Burned Pixels in GT:   {test_metrics['gt_burned_percentage']:.2f}%")
    print(f"  Predicted Burned:      {test_metrics['pred_burned_percentage']:.2f}%")
    print(f"  Peak GPU VRAM:         {peak_vram:.2f} GB")

    final_results = {
        "dataset_split": {
            "test_fold": "cyan",
            "val_fold": "purple",
            "train_folds": ["coral", "grey", "lime", "magenta", "pink"],
            "train_events": 43,
            "val_events": 8,
            "test_events": 10,
            "train_patches": len(train_dataset),
            "val_patches": len(val_dataset),
            "test_patches": len(test_dataset)
        },
        "model_configuration": {
            "model": "ResNet34UNet",
            "input_channels": [
                "S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12",
                "S1_VV", "S1_VH", "S1_VV_VH_ratio"
            ],
            "patch_size": 256,
            "stride": 128,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "loss_function": "0.5*Focal + 0.5*Dice"
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "total_epochs_trained": len(history),
            "best_val_dice": float(best_val_dice),
            "training_time_minutes": total_training_time / 60.0,
            "peak_vram_gb": float(peak_vram)
        },
        "test_metrics": test_metrics
    }
    metrics_json = os.path.join(metrics_save_dir, "fusion_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved complete Fusion metrics report to: {metrics_json}")

    # Generate 10 test visualizations
    generate_fusion_visualizations(model, test_dataset, stats_file, vis_save_dir, device, num_samples=10)

    print("\n" + "="*65)
    print("STAGE 5 MULTIMODAL FUSION EXPERIMENT COMPLETE!")
    print("="*65)

if __name__ == "__main__":
    run_fusion_training_experiment()
