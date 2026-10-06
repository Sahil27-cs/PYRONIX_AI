"""
Stage 6: Comprehensive Multimodal Fusion Evaluation & Systematic Ablation Study
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Features:
- True epoch-level checkpointing for all ablation variants (saved after EVERY completed epoch).
- True resume-from-last-saved-state (restores model, optimizer, scheduler, AMP scaler, epoch).
- Strict separation of LATEST checkpoint (models/checkpoints/stage6/) vs BEST checkpoint (models/).
- Explicit training completion flag (no skipping based solely on file existence).
- Resumable pipeline state tracking across training, evaluation, sensitivity, statistics, and plots.
- Atomic file saves preventing partial writes or corruption upon sudden interruptions.
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import ResNet34UNet
from src.data.ablation_dataset import AblationPatchDataset
from src.training.losses import CombinedLoss
from src.evaluation.metrics import MetricTracker
from src.training.stage6_state import Stage6StateManager, atomic_save_torch, atomic_save_json, load_json

# Random seeds for strict reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_ablation_model(
    model_name,
    channel_indices,
    in_channels,
    train_dir,
    val_dir,
    stats_json,
    models_dir,
    checkpoints_dir,
    state_manager,
    epochs=12,
    batch_size=16,
    device="cuda",
    seed=42
):
    """
    Trains an ablation model variant with true epoch-level checkpointing and safe resumption.
    Saves:
      - Latest checkpoint (after every epoch): models/checkpoints/stage6/{model_name}_latest.pt
      - Best model weights (on val dice improvement): models/ablation_{model_name}.pt
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_latest.pt")
    best_model_path = os.path.join(models_dir, f"ablation_{model_name}.pt")

    # 1. Check if training is already 100% complete
    if os.path.exists(latest_ckpt_path):
        try:
            latest_ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
            curr_ep = latest_ckpt.get("current_epoch", 0)
            tot_ep = latest_ckpt.get("total_epochs", epochs)
            is_complete = bool(latest_ckpt.get("training_complete", False)) and (curr_ep >= tot_ep)
            if is_complete:
                print(f"[{model_name}] Training is ALREADY COMPLETE ({curr_ep}/{tot_ep} epochs). Skipping retraining.")
                state_manager.update_model_state(model_name, "complete", curr_ep)
                return best_model_path
        except Exception as e:
            print(f"[{model_name}] Warning reading latest checkpoint: {e}")

    # Check best model if marked complete
    if os.path.exists(best_model_path):
        try:
            best_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
            if bool(best_ckpt.get("training_complete", False)) and best_ckpt.get("total_epochs", epochs) == epochs:
                print(f"[{model_name}] Best checkpoint marked training_complete = True. Skipping retraining.")
                state_manager.update_model_state(model_name, "complete", epochs)
                return best_model_path
        except Exception:
            pass

    # 2. Setup training components
    model = ResNet34UNet(in_channels=in_channels, num_classes=1, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedLoss(alpha=0.5, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda' if "cuda" in str(device) else 'cpu')

    start_epoch = 1
    best_val_dice = 0.0
    best_val_iou = 0.0

    # 3. Resume from last completed epoch if latest checkpoint exists
    if os.path.exists(latest_ckpt_path):
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            required_keys = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "current_epoch"]
            if all(k in ckpt for k in required_keys):
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                last_completed_epoch = ckpt["current_epoch"]
                best_val_dice = ckpt.get("best_val_dice", 0.0)
                best_val_iou = ckpt.get("best_val_iou", 0.0)
                start_epoch = last_completed_epoch + 1

                print("\n" + "="*60)
                print("RESUMING TRAINING")
                print(f"Model: {model_name}")
                print(f"Last completed epoch: {last_completed_epoch}")
                print(f"Resuming from epoch: {start_epoch}")
                print(f"Total epochs: {epochs}")
                print("="*60)
            else:
                print(f"[{model_name}] Warning: Latest checkpoint missing some required state keys. Starting fresh.")
        except Exception as e:
            print(f"[{model_name}] Error loading latest checkpoint: {e}. Starting fresh.")
    elif os.path.exists(best_model_path):
        # Found legacy checkpoint without latest.pt
        try:
            legacy_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
            best_val_dice = legacy_ckpt.get("best_val_dice", 0.0)
            best_val_iou = legacy_ckpt.get("best_val_iou", 0.0)
            print("\n" + "="*60)
            print(f"NOTICE: Found legacy checkpoint {best_model_path} (Best Val Dice: {best_val_dice*100:.2f}%).")
            print("It lacks epoch/optimizer/scheduler state required for mid-epoch resume.")
            print("Starting training from Epoch 1 with true epoch-level resume checkpointing enabled.")
            print("Existing best weights will be preserved unless exceeded by new validation score.")
            print("="*60)
        except Exception as e:
            print(f"[{model_name}] Warning reading legacy checkpoint: {e}")
    else:
        print(f"\n{'='*60}")
        print(f"STARTING NEW TRAINING: {model_name} (in_channels={in_channels}, indices={channel_indices})")
        print(f"Total epochs: {epochs}")
        print(f"{'='*60}")

    if start_epoch > epochs:
        print(f"[{model_name}] Already completed all {epochs} epochs. Returning.")
        state_manager.update_model_state(model_name, "complete", epochs)
        return best_model_path

    # Data loaders
    train_ds = AblationPatchDataset(train_dir, channel_indices, stats_json, augment=True)
    val_ds = AblationPatchDataset(val_dir, channel_indices, stats_json, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    state_manager.update_model_state(model_name, "running", start_epoch - 1)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        # Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                outputs = model(images)
                loss, _, _ = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_tracker = MetricTracker()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_tracker.update(outputs, masks)

        val_loss /= len(val_ds)
        val_metrics = val_tracker.compute()
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_metrics['iou']*100:.2f}% | Val Dice: {val_metrics['dice']*100:.2f}% | Time: {elapsed:.1f}s")

        is_complete = (epoch == epochs)

        # Save Best Checkpoint (only when validation Dice improves)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_ckpt = {
                "model_name": model_name,
                "in_channels": in_channels,
                "channel_indices": channel_indices,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "epoch_saved": epoch,
                "total_epochs": epochs,
                "training_complete": is_complete,
                "model_state_dict": model.state_dict(),
            }
            atomic_save_torch(best_ckpt, best_model_path)
            print(f"  --> Saved new best {model_name} (Val Dice: {best_val_dice*100:.2f}%)")

        # If final epoch and best_model_path exists, ensure it is stamped training_complete
        if is_complete:
            if os.path.exists(best_model_path):
                try:
                    b_data = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    b_data["training_complete"] = True
                    b_data["total_epochs"] = epochs
                    atomic_save_torch(b_data, best_model_path)
                except Exception:
                    pass
            else:
                # Save current weights as best if none was saved previously
                atomic_save_torch({
                    "model_name": model_name,
                    "in_channels": in_channels,
                    "channel_indices": channel_indices,
                    "best_val_dice": val_metrics["dice"],
                    "best_val_iou": val_metrics["iou"],
                    "epoch_saved": epoch,
                    "total_epochs": epochs,
                    "training_complete": True,
                    "model_state_dict": model.state_dict(),
                }, best_model_path)

        # Save Latest Checkpoint atomically after EVERY completed epoch
        latest_ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "current_epoch": epoch,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "model_name": model_name,
            "channel_indices": channel_indices,
            "in_channels": in_channels,
            "total_epochs": epochs,
            "random_seed": seed,
            "training_configuration": {
                "batch_size": batch_size,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "eta_min": 1e-6,
                "loss": "0.5*Focal + 0.5*Dice (alpha=0.5, gamma=2.0)",
                "architecture": "ResNet34UNet",
                "in_channels": in_channels,
                "channel_indices": channel_indices,
            },
            "training_complete": is_complete
        }
        atomic_save_torch(latest_ckpt, latest_ckpt_path)

        # Update pipeline state
        state_manager.update_model_state(
            model_name=model_name,
            status="complete" if is_complete else "running",
            last_completed_epoch=epoch
        )

    print(f"Completed training {model_name}. Best Val Dice: {best_val_dice*100:.2f}%")
    return best_model_path

def evaluate_model_on_test(model, channel_indices, test_dir, stats_json, device):
    """
    Evaluates a model patch by patch on the 424 test patches,
    returning overall pixel-level metrics, per-patch results, and per-event aggregations.
    """
    ds = AblationPatchDataset(test_dir, channel_indices, stats_json, augment=False)
    tracker = MetricTracker()
    patch_results = []

    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            image_t, mask_t = ds[i]
            img_in = image_t.unsqueeze(0).to(device)
            mask_in = mask_t.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                logits = model(img_in)

            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            tracker.update(logits, mask_in)

            # Burned area calculation (1 pixel = 100 m^2 = 0.0001 km^2)
            gt_pixels = mask_in.sum().item()
            pred_pixels = pred_mask.sum().item()
            gt_km2 = gt_pixels * 0.0001
            pred_km2 = pred_pixels * 0.0001

            # Event extraction from filename
            fname = os.path.basename(ds.patch_files[i])
            event_name = "_".join(fname.split("_")[4:]).replace(".npz", "")

            # Patch-level metrics
            intersection = ((pred_mask == 1) & (mask_in == 1)).sum().item()
            union = ((pred_mask == 1) | (mask_in == 1)).sum().item()
            patch_iou = (intersection / union) if union > 0 else (1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0)
            patch_dice = (2.0 * intersection / (gt_pixels + pred_pixels)) if (gt_pixels + pred_pixels) > 0 else (1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0)

            patch_results.append({
                "patch_file": fname,
                "event": event_name,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "gt_km2": gt_km2,
                "pred_km2": pred_km2,
                "error_km2": pred_km2 - gt_km2,
                "abs_error_km2": abs(pred_km2 - gt_km2),
                "patch_iou": patch_iou,
                "patch_dice": patch_dice
            })

    overall_metrics = tracker.compute()
    return overall_metrics, patch_results

def evaluate_masked_fusion(fusion_model, mask_channel_indices, test_dir, stats_json, device):
    """
    Evaluates the 9-channel Fusion model with specified channel indices zeroed out (masked).
    """
    ds = AblationPatchDataset(test_dir, list(range(9)), stats_json, augment=False)
    tracker = MetricTracker()
    patch_results = []

    fusion_model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            image_t, mask_t = ds[i]
            # Zero out the specified channels (0 is the normalized mean)
            image_masked = image_t.clone()
            for c in mask_channel_indices:
                image_masked[c, :, :] = 0.0

            img_in = image_masked.unsqueeze(0).to(device)
            mask_in = mask_t.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                logits = fusion_model(img_in)

            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            tracker.update(logits, mask_in)

            gt_pixels = mask_in.sum().item()
            pred_pixels = pred_mask.sum().item()
            gt_km2 = gt_pixels * 0.0001
            pred_km2 = pred_pixels * 0.0001

            fname = os.path.basename(ds.patch_files[i])
            event_name = "_".join(fname.split("_")[4:]).replace(".npz", "")

            intersection = ((pred_mask == 1) & (mask_in == 1)).sum().item()
            union = ((pred_mask == 1) | (mask_in == 1)).sum().item()
            patch_iou = (intersection / union) if union > 0 else (1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0)
            patch_dice = (2.0 * intersection / (gt_pixels + pred_pixels)) if (gt_pixels + pred_pixels) > 0 else (1.0 if gt_pixels == 0 and pred_pixels == 0 else 0.0)

            patch_results.append({
                "patch_file": fname,
                "event": event_name,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "gt_km2": gt_km2,
                "pred_km2": pred_km2,
                "error_km2": pred_km2 - gt_km2,
                "abs_error_km2": abs(pred_km2 - gt_km2),
                "patch_iou": patch_iou,
                "patch_dice": patch_dice
            })

    overall_metrics = tracker.compute()
    return overall_metrics, patch_results

def compute_burned_area_statistics(patch_results):
    df = pd.DataFrame(patch_results)
    event_df = df.groupby("event")[["gt_km2", "pred_km2"]].sum().reset_index()
    event_df["error_km2"] = event_df["pred_km2"] - event_df["gt_km2"]
    event_df["abs_error_km2"] = np.abs(event_df["error_km2"])
    event_df["pct_error"] = (event_df["abs_error_km2"] / (event_df["gt_km2"] + 1e-5)) * 100.0

    total_gt = event_df["gt_km2"].sum()
    total_pred = event_df["pred_km2"].sum()
    mae_event = event_df["abs_error_km2"].mean()
    mape_event = event_df["pct_error"].mean()

    # R^2 and Pearson r
    if len(event_df) > 1 and event_df["gt_km2"].std() > 0:
        r, _ = stats.pearsonr(event_df["gt_km2"], event_df["pred_km2"])
        slope, intercept, r_value, p_value, std_err = stats.linregress(event_df["gt_km2"], event_df["pred_km2"])
        r2 = r_value**2
    else:
        r = 0.0
        r2 = 0.0

    return {
        "total_gt_km2": float(total_gt),
        "total_pred_km2": float(total_pred),
        "total_bias_km2": float(total_pred - total_gt),
        "total_bias_pct": float(((total_pred - total_gt) / total_gt) * 100.0) if total_gt > 0 else 0.0,
        "mae_event_km2": float(mae_event),
        "mape_event_pct": float(mape_event),
        "pearson_r": float(r),
        "r2_score": float(r2),
        "event_breakdown": event_df.to_dict(orient="records")
    }

def print_status_report(state_manager):
    """Prints a clear inspection report of all Stage 6 checkpoints and state."""
    report = state_manager.inspect_checkpoints()
    print("\n" + "="*70)
    print("STAGE 6 CHECKPOINT & RESUME STATUS REPORT")
    print("="*70)
    print(f"{'Model':<16} {'Last Epoch':<14} {'Complete?':<12} {'Checkpoint Type':<25}")
    print("-" * 70)
    for m, info in report.items():
        comp_str = "YES" if info["complete"] else "NO"
        print(f"{m:<16} {str(info['last_epoch']):<14} {comp_str:<12} {info['checkpoint_type']:<25}")
    print("-" * 70)
    print(f"Current Stage 6 pipeline state: {state_manager.pipeline_state.get('overall_status', 'pending')}")
    print(f"Evaluation step:               {state_manager.pipeline_state.get('evaluation', 'pending')}")
    print(f"Feature sensitivity step:      {state_manager.pipeline_state.get('feature_sensitivity', 'pending')}")
    print(f"Statistics step:               {state_manager.pipeline_state.get('statistics', 'pending')}")
    print(f"Visualizations step:           {state_manager.pipeline_state.get('visualizations', 'pending')}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Stage 6 Ablation Study Pipeline")
    parser.add_argument("--status", action="store_true", help="Print current status and checkpoint report without training")
    args = parser.parse_args()

    print("="*75)
    print("STAGE 6 — MULTIMODAL FUSION EVALUATION & RIGOROUS ABLATION STUDY")
    print("="*75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = os.path.join(project_root, "data", "processed_fusion", "train")
    val_dir = os.path.join(project_root, "data", "processed_fusion", "val")
    test_dir = os.path.join(project_root, "data", "processed_fusion", "test")
    stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    models_dir = os.path.join(project_root, "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints", "stage6")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    vis_dir = os.path.join(project_root, "results", "visualizations", "ablation")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize State Manager
    state_manager = Stage6StateManager(
        metrics_dir=metrics_dir,
        checkpoints_dir=checkpoints_dir,
        models_dir=models_dir
    )

    if args.status:
        print_status_report(state_manager)
        return

    state_manager.set_overall_status("running")

    # 1. Define Ablation Variants to Train
    ablation_configs = [
        {"name": "S1_VV_only", "channels": [6], "in_channels": 1, "desc": "SAR Single-Pol (VV only)"},
        {"name": "S1_VH_only", "channels": [7], "in_channels": 1, "desc": "SAR Cross-Pol (VH only)"},
        {"name": "S1_VV_VH", "channels": [6, 7], "in_channels": 2, "desc": "SAR Dual-Pol without ratio (VV+VH)"},
        {"name": "S2_RGB_only", "channels": [0, 1, 2], "in_channels": 3, "desc": "Optical Visible Only (B2,B3,B4)"},
        {"name": "S2_NIR_SWIR", "channels": [3, 4, 5], "in_channels": 3, "desc": "Optical Infrared Only (B8,B11,B12)"}
    ]

    trained_models = {}

    # Train or load new ablation models
    for cfg in ablation_configs:
        ckpt_path = train_ablation_model(
            model_name=cfg["name"],
            channel_indices=cfg["channels"],
            in_channels=cfg["in_channels"],
            train_dir=train_dir,
            val_dir=val_dir,
            stats_json=stats_json,
            models_dir=models_dir,
            checkpoints_dir=checkpoints_dir,
            state_manager=state_manager,
            epochs=12,
            batch_size=16,
            device=device
        )
        trained_models[cfg["name"]] = {
            "path": ckpt_path,
            "channels": cfg["channels"],
            "in_channels": cfg["in_channels"],
            "desc": cfg["desc"]
        }

    # Add Full Baseline Models
    trained_models["S1_Full"] = {
        "path": os.path.join(models_dir, "best_s1_baseline_model.pt"),
        "channels": [6, 7, 8],
        "in_channels": 3,
        "desc": "SAR Full 3-channel Baseline (VV, VH, VV/VH ratio)"
    }
    trained_models["S2_Full"] = {
        "path": os.path.join(models_dir, "best_s2_baseline_model.pt"),
        "channels": [0, 1, 2, 3, 4, 5],
        "in_channels": 6,
        "desc": "Optical Full 6-channel Baseline (RGB + NIR + SWIR1 + SWIR2)"
    }
    trained_models["Multimodal_Fusion"] = {
        "path": os.path.join(models_dir, "best_fusion_model.pt"),
        "channels": list(range(9)),
        "in_channels": 9,
        "desc": "Multimodal Early Concatenation (S2 6-ch + S1 3-ch = 9-ch)"
    }

    # 2. Systematically Evaluate All 8 Models on Test Set (424 Patches)
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS ON THE 424 SHARED TEST PATCHES...")
    print("="*60)

    model_eval_results, patch_level_dict = state_manager.load_cached_evaluations()

    for name, info in trained_models.items():
        eval_status = state_manager.get_eval_status(name)
        if eval_status == "complete" and name in model_eval_results and name in patch_level_dict:
            print(f"[{name}] Evaluation already COMPLETE. Loaded from cache.")
            cached_res = model_eval_results[name]
            m = cached_res["metrics"]
            ba = cached_res["burned_area"]
            print(f"  IoU:       {m['iou']*100:.2f}%")
            print(f"  Dice:      {m['dice']*100:.2f}%")
            print(f"  Precision: {m['precision']*100:.2f}%")
            print(f"  Recall:    {m['recall']*100:.2f}%")
            print(f"  Area Bias: {ba['total_bias_pct']:+.2f}%")
            print(f"  Area R2:   {ba['r2_score']:.4f}")
            continue

        print(f"\nEvaluating: {name} ({info['desc']})...")
        ckpt = torch.load(info["path"], map_location=device, weights_only=False)
        m = ResNet34UNet(in_channels=info["in_channels"], num_classes=1, pretrained=False).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()

        overall_metrics, patch_res = evaluate_model_on_test(
            model=m,
            channel_indices=info["channels"],
            test_dir=test_dir,
            stats_json=stats_json,
            device=device
        )

        area_stats = compute_burned_area_statistics(patch_res)

        eval_data = {
            "description": info["desc"],
            "in_channels": info["in_channels"],
            "channel_indices": info["channels"],
            "metrics": overall_metrics,
            "burned_area": area_stats
        }

        # Atomically record evaluation
        state_manager.record_model_evaluation(name, eval_data, patch_res)
        model_eval_results[name] = eval_data
        patch_level_dict[name] = patch_res

        print(f"  IoU:       {overall_metrics['iou']*100:.2f}%")
        print(f"  Dice:      {overall_metrics['dice']*100:.2f}%")
        print(f"  Precision: {overall_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {overall_metrics['recall']*100:.2f}%")
        print(f"  Area Bias: {area_stats['total_bias_pct']:+.2f}% (Pred: {area_stats['total_pred_km2']:.1f} km2 vs GT: {area_stats['total_gt_km2']:.1f} km2)")
        print(f"  Area R2:   {area_stats['r2_score']:.4f}")

    state_manager.update_step_status("evaluation", "complete")

    # 3. Multimodal Fusion Feature Drop / Masking Sensitivity Analysis
    print("\n" + "="*60)
    print("PERFORMING FEATURE SENSITIVITY ANALYSIS ON FUSION MODEL...")
    print("="*60)

    fusion_ckpt = torch.load(trained_models["Multimodal_Fusion"]["path"], map_location=device, weights_only=False)
    fusion_model = ResNet34UNet(in_channels=9, num_classes=1, pretrained=False).to(device)
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fusion_model.eval()

    mask_experiments = [
        {"name": "Fusion_Baseline_All9", "mask_channels": [], "desc": "All 9 channels active (Baseline)"},
        {"name": "Mask_S1_SAR_All", "mask_channels": [6, 7, 8], "desc": "Masked S1 SAR (Bands 6,7,8 -> Optical Only)"},
        {"name": "Mask_S2_Optical_All", "mask_channels": [0, 1, 2, 3, 4, 5], "desc": "Masked S2 Optical (Bands 0..5 -> SAR Only)"},
        {"name": "Mask_VV_Only", "mask_channels": [6], "desc": "Masked VV Band"},
        {"name": "Mask_VH_Only", "mask_channels": [7], "desc": "Masked VH Band"},
        {"name": "Mask_Ratio_Only", "mask_channels": [8], "desc": "Masked VV/VH Ratio Band"},
        {"name": "Mask_RGB_Only", "mask_channels": [0, 1, 2], "desc": "Masked Optical RGB (B2,B3,B4)"},
        {"name": "Mask_NIR_Only", "mask_channels": [3], "desc": "Masked Optical NIR (B8)"},
        {"name": "Mask_SWIR_Only", "mask_channels": [4, 5], "desc": "Masked Optical SWIR (B11,B12)"}
    ]

    sensitivity_results = state_manager.load_cached_sensitivity()
    base_iou = model_eval_results["Multimodal_Fusion"]["metrics"]["iou"]
    base_dice = model_eval_results["Multimodal_Fusion"]["metrics"]["dice"]

    for me in mask_experiments:
        sens_status = state_manager.get_sens_status(me["name"])
        if sens_status == "complete" and me["name"] in sensitivity_results:
            print(f"[{me['name']}] Sensitivity test already COMPLETE. Loaded from cache.")
            cached_s = sensitivity_results[me["name"]]
            print(f"  IoU: {cached_s['iou']*100:.2f}% ({cached_s['delta_iou_pct']:+.2f}%) | Dice: {cached_s['dice']*100:.2f}% ({cached_s['delta_dice_pct']:+.2f}%)")
            continue

        print(f"Running sensitivity test: {me['name']} ({me['desc']})...")
        m_metrics, m_patch_res = evaluate_masked_fusion(
            fusion_model=fusion_model,
            mask_channel_indices=me["mask_channels"],
            test_dir=test_dir,
            stats_json=stats_json,
            device=device
        )
        delta_iou = (m_metrics["iou"] - base_iou) * 100.0
        delta_dice = (m_metrics["dice"] - base_dice) * 100.0
        res = {
            "description": me["desc"],
            "masked_channels": me["mask_channels"],
            "iou": m_metrics["iou"],
            "dice": m_metrics["dice"],
            "precision": m_metrics["precision"],
            "recall": m_metrics["recall"],
            "delta_iou_pct": delta_iou,
            "delta_dice_pct": delta_dice
        }
        state_manager.record_sensitivity_result(me["name"], res)
        sensitivity_results[me["name"]] = res
        print(f"  IoU: {m_metrics['iou']*100:.2f}% ({delta_iou:+.2f}%) | Dice: {m_metrics['dice']*100:.2f}% ({delta_dice:+.2f}%)")

    state_manager.update_step_status("feature_sensitivity", "complete")

    # 4. Statistical Significance Testing
    print("\n" + "="*60)
    print("PERFORMING STATISTICAL SIGNIFICANCE TESTS ACROSS EVENTS...")
    print("="*60)

    events_list = [e["event"] for e in model_eval_results["S2_Full"]["burned_area"]["event_breakdown"]]
    statistical_tests = {}

    for name in model_eval_results.keys():
        if name in ["S2_Full", "Multimodal_Fusion"]:
            continue
        s2_patch_iou = [p["patch_iou"] for p in patch_level_dict["S2_Full"]]
        m_patch_iou = [p["patch_iou"] for p in patch_level_dict[name]]

        try:
            stat_w, p_w = stats.wilcoxon(m_patch_iou, s2_patch_iou)
        except Exception:
            stat_w, p_w = 0.0, 1.0

        t_stat, p_t = stats.ttest_rel(m_patch_iou, s2_patch_iou)

        statistical_tests[name] = {
            "vs_model": "S2_Full",
            "wilcoxon_stat": float(stat_w),
            "wilcoxon_p_val": float(p_w),
            "ttest_stat": float(t_stat),
            "ttest_p_val": float(p_t),
            "significant_at_005": bool(p_w < 0.05),
            "mean_iou_diff_pct": float((np.mean(m_patch_iou) - np.mean(s2_patch_iou)) * 100.0)
        }
        print(f"{name} vs S2_Full: Wilcoxon p={p_w:.4e}, Significant={p_w < 0.05} (Delta IoU: {(np.mean(m_patch_iou) - np.mean(s2_patch_iou))*100:+.2f}%)")

    fusion_patch_iou = [p["patch_iou"] for p in patch_level_dict["Multimodal_Fusion"]]
    s2_patch_iou = [p["patch_iou"] for p in patch_level_dict["S2_Full"]]
    stat_w, p_w = stats.wilcoxon(fusion_patch_iou, s2_patch_iou)
    t_stat, p_t = stats.ttest_rel(fusion_patch_iou, s2_patch_iou)
    statistical_tests["Multimodal_Fusion_vs_S2"] = {
        "vs_model": "S2_Full",
        "wilcoxon_stat": float(stat_w),
        "wilcoxon_p_val": float(p_w),
        "ttest_stat": float(t_stat),
        "ttest_p_val": float(p_t),
        "significant_at_005": bool(p_w < 0.05),
        "mean_iou_diff_pct": float((np.mean(fusion_patch_iou) - np.mean(s2_patch_iou)) * 100.0)
    }

    # 5. Build Summary Tables
    summary_rows = []
    order = ["S1_VV_only", "S1_VH_only", "S1_VV_VH", "S1_Full", "S2_RGB_only", "S2_NIR_SWIR", "S2_Full", "Multimodal_Fusion"]
    for k in order:
        res = model_eval_results[k]
        m = res["metrics"]
        ba = res["burned_area"]
        summary_rows.append({
            "Model Variant": k,
            "Description": res["description"],
            "Channels": res["in_channels"],
            "Test IoU (%)": round(m["iou"] * 100.0, 2),
            "Test Dice (%)": round(m["dice"] * 100.0, 2),
            "Precision (%)": round(m["precision"] * 100.0, 2),
            "Recall (%)": round(m["recall"] * 100.0, 2),
            "Accuracy (%)": round(m["accuracy"] * 100.0, 2),
            "Pred Area (km2)": round(ba["total_pred_km2"], 1),
            "GT Area (km2)": round(ba["total_gt_km2"], 1),
            "Area Bias (%)": round(ba["total_bias_pct"], 2),
            "Area R2": round(ba["r2_score"], 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(metrics_dir, "ablation_study_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to {summary_csv_path}")

    sens_rows = []
    for k, v in sensitivity_results.items():
        sens_rows.append({
            "Experiment": k,
            "Description": v["description"],
            "Test IoU (%)": round(v["iou"] * 100.0, 2),
            "Test Dice (%)": round(v["dice"] * 100.0, 2),
            "Delta IoU (%)": round(v["delta_iou_pct"], 2),
            "Delta Dice (%)": round(v["delta_dice_pct"], 2),
            "Precision (%)": round(v["precision"] * 100.0, 2),
            "Recall (%)": round(v["recall"] * 100.0, 2)
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_csv_path = os.path.join(metrics_dir, "fusion_sensitivity_analysis.csv")
    sens_df.to_csv(sens_csv_path, index=False)
    print(f"Saved sensitivity table to {sens_csv_path}")

    event_rows = []
    for ev in events_list:
        row = {"Event": ev}
        for k in order:
            eb = next(e for e in model_eval_results[k]["burned_area"]["event_breakdown"] if e["event"] == ev)
            row["GT_km2"] = round(eb["gt_km2"], 2)
            row[f"{k}_Pred_km2"] = round(eb["pred_km2"], 2)
            row[f"{k}_Err_pct"] = round(eb["pct_error"], 1)
        event_rows.append(row)
    event_df = pd.DataFrame(event_rows)
    event_csv_path = os.path.join(metrics_dir, "ablation_per_event.csv")
    event_df.to_csv(event_csv_path, index=False)
    print(f"Saved per-event breakdown to {event_csv_path}")

    final_output = {
        "stage": "STAGE 6: MULTIMODAL FUSION EVALUATION & RIGOROUS ABLATION STUDY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": model_eval_results,
        "feature_sensitivity_analysis": sensitivity_results,
        "statistical_tests": statistical_tests,
        "summary_table": summary_rows
    }
    json_path = os.path.join(metrics_dir, "ablation_study.json")
    atomic_save_json(final_output, json_path)
    print(f"Saved complete study metrics to {json_path}")
    state_manager.update_step_status("statistics", "complete")

    # 6. Generate Publication-Quality Visualizations
    chart1_path = os.path.join(vis_dir, "ablation_metrics_comparison.png")
    chart2_path = os.path.join(vis_dir, "burned_area_scatter_plots.png")
    chart3_path = os.path.join(vis_dir, "fusion_channel_sensitivity.png")
    chart4_path = os.path.join(vis_dir, "scene_ablation_comparison.png")

    all_charts_exist = all(os.path.exists(p) for p in [chart1_path, chart2_path, chart3_path, chart4_path])
    if state_manager.pipeline_state.get("visualizations") == "complete" and all_charts_exist:
        print("\nVisualizations already complete and verified. Skipping plot generation.")
    else:
        print("\n" + "="*60)
        print("GENERATING ABLATION VISUALIZATIONS...")
        print("="*60)

        # Plot 1: Performance Bar Chart Across Models
        fig, ax = plt.subplots(figsize=(13, 6))
        x = np.arange(len(order))
        width = 0.20

        ious = [model_eval_results[k]["metrics"]["iou"] * 100 for k in order]
        dices = [model_eval_results[k]["metrics"]["dice"] * 100 for k in order]
        precs = [model_eval_results[k]["metrics"]["precision"] * 100 for k in order]
        recs = [model_eval_results[k]["metrics"]["recall"] * 100 for k in order]

        rects1 = ax.bar(x - 1.5*width, ious, width, label="IoU (%)", color="#2b5c8f")
        rects2 = ax.bar(x - 0.5*width, dices, width, label="Dice (F1) (%)", color="#2ca02c")
        rects3 = ax.bar(x + 0.5*width, precs, width, label="Precision (%)", color="#ff7f0e")
        rects4 = ax.bar(x + 1.5*width, recs, width, label="Recall (%)", color="#d62728")

        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("Ablation Study: Model Performance Across Sensor Modalities & Spectral Subsets", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        labels = ["S1 (VV)", "S1 (VH)", "S1 (VV+VH)", "S1 Full (3ch)", "S2 RGB (3ch)", "S2 NIR+SWIR (3ch)", "S2 Full (6ch)", "Fusion (9ch)"]
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 100)

        for bar_group in [rects1, rects2]:
            for bar in bar_group:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(chart1_path, dpi=300)
        plt.close()
        print(f"Saved: {chart1_path}")

        # Plot 2: Burned Area Scatter Plots (GT vs Pred) for Key Models
        key_models = ["S1_Full", "S2_RGB_only", "S2_NIR_SWIR", "S2_Full", "Multimodal_Fusion"]
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True, sharex=True)

        max_area = max(model_eval_results["S2_Full"]["burned_area"]["total_gt_km2"], 300)

        for i, km in enumerate(key_models):
            ax_i = axes[i]
            eb = model_eval_results[km]["burned_area"]["event_breakdown"]
            gt_vals = [e["gt_km2"] for e in eb]
            pred_vals = [e["pred_km2"] for e in eb]
            r2 = model_eval_results[km]["burned_area"]["r2_score"]

            ax_i.scatter(gt_vals, pred_vals, color="#e6550d" if "S1" in km else ("#3182bd" if "S2" in km else "#756bb1"), s=60, edgecolors="k", zorder=3)
            ax_i.plot([0, max_area], [0, max_area], "r--", linewidth=1.5, label="1:1 Perfect Fit", zorder=2)
            ax_i.set_title(f"{km}\n(R² = {r2:.3f})", fontsize=11, fontweight="bold")
            ax_i.set_xlabel("Ground Truth (km²)", fontsize=10)
            if i == 0:
                ax_i.set_ylabel("Predicted Burned Area (km²)", fontsize=10)
            ax_i.grid(True, linestyle=":", alpha=0.6)
            ax_i.legend(loc="upper left", fontsize=8)

        plt.suptitle("Burned Area Estimation Agreement Across Test Events (10 Wildfire Events)", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {chart2_path}")

        # Plot 3: Fusion Channel Sensitivity / Drop Chart
        fig, ax = plt.subplots(figsize=(11, 5.5))
        exp_names = [e for e in sensitivity_results.keys() if e != "Fusion_Baseline_All9"]
        d_ious = [sensitivity_results[e]["delta_iou_pct"] for e in exp_names]
        d_dices = [sensitivity_results[e]["delta_dice_pct"] for e in exp_names]
        clean_labels = [sensitivity_results[e]["description"].replace("Masked ", "") for e in exp_names]

        y = np.arange(len(exp_names))
        h = 0.35

        ax.barh(y - h/2, d_ious, h, label="Δ IoU (%)", color="#d95f02")
        ax.barh(y + h/2, d_dices, h, label="Δ Dice (%)", color="#7570b3")

        ax.set_yticks(y)
        ax.set_yticklabels(clean_labels, fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linestyle="-", linewidth=1)
        ax.set_xlabel("Performance Change Compared to Full Fusion Baseline (%)", fontsize=11, fontweight="bold")
        ax.set_title("Multimodal Network Sensitivity: Impact of Masking Modalities and Bands", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f"Saved: {chart3_path}")

        # Plot 4: Qualitative Multi-Model Comparison on Sample Test Patch
        high_fire_patches = [p for p in patch_level_dict["S2_Full"] if p["gt_pixels"] > 5000]
        sample_patch_name = high_fire_patches[0]["patch_file"] if high_fire_patches else patch_level_dict["S2_Full"][0]["patch_file"]
        sample_patch_path = os.path.join(test_dir, sample_patch_name)

        with np.load(sample_patch_path) as d:
            img_raw = d["image"].astype(np.float32)
            gt_mask = d["mask"][0].astype(np.float32)

        with open(stats_json) as f:
            st = json.load(f)
        mean_all = np.array(st["mean"], dtype=np.float32).reshape(9, 1, 1)
        std_all = np.array(st["std"], dtype=np.float32).reshape(9, 1, 1)
        norm_all = (img_raw - mean_all) / std_all

        # RGB composite (B4=2, B3=1, B2=0)
        rgb = np.stack([img_raw[2], img_raw[1], img_raw[0]], axis=-1)
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

        # False Color SWIR/NIR composite (B12=5, B8=3, B4=2)
        swir_comp = np.stack([img_raw[5], img_raw[3], img_raw[2]], axis=-1)
        swir_comp = np.clip(swir_comp / np.percentile(swir_comp, 98), 0, 1)

        # SAR VH
        sar_vh = img_raw[7]

        # Generate predictions for key models on this patch
        preds_map = {}
        eval_list = ["S1_Full", "S2_RGB_only", "S2_NIR_SWIR", "S2_Full", "Multimodal_Fusion"]
        for km in eval_list:
            info = trained_models[km]
            ckpt = torch.load(info["path"], map_location=device, weights_only=False)
            m = ResNet34UNet(in_channels=info["in_channels"], num_classes=1, pretrained=False).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()

            inp = norm_all[info["channels"], :, :]
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                    l = m(inp_t)
                p = (torch.sigmoid(l) > 0.5).float().cpu().squeeze().numpy()
            preds_map[km] = p

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("S2 True Color (RGB)", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(swir_comp)
        axes[0, 1].set_title("S2 False Color (SWIR/NIR)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(sar_vh, cmap="gray")
        axes[0, 2].set_title("S1 SAR (VH Backscatter)", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(gt_mask, cmap="Reds")
        axes[0, 3].set_title("Ground Truth Burned Area", fontsize=11, fontweight="bold", color="darkred")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(preds_map["S1_Full"], cmap="copper")
        axes[1, 0].set_title(f"S1 Full (SAR Only)\nIoU: {model_eval_results['S1_Full']['metrics']['iou']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(preds_map["S2_RGB_only"], cmap="Purples")
        axes[1, 1].set_title(f"S2 RGB Only\nIoU: {model_eval_results['S2_RGB_only']['metrics']['iou']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(preds_map["S2_Full"], cmap="Greens")
        axes[1, 2].set_title(f"S2 Full (Optical)\nIoU: {model_eval_results['S2_Full']['metrics']['iou']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(preds_map["Multimodal_Fusion"], cmap="Blues")
        axes[1, 3].set_title(f"Multimodal Fusion (S1+S2)\nIoU: {model_eval_results['Multimodal_Fusion']['metrics']['iou']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[1, 3].axis("off")

        plt.suptitle(f"Qualitative Ablation Comparison on Test Scene: {sample_patch_name[:40]}...", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(chart4_path, dpi=300)
        plt.close()
        print(f"Saved: {chart4_path}")

        state_manager.update_step_status("visualizations", "complete")

    state_manager.set_overall_status("complete")
    print("\n" + "="*60)
    print("STAGE 6 ABLATION STUDY COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
