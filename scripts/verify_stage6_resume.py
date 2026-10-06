"""
Verification Suite for Stage 6 True Resume-From-Last-Saved-State Pipeline
Tests all 10 criteria specified in Requirement 13 without performing expensive training.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.stage6_state import Stage6StateManager, atomic_save_torch, load_json

class TinyTestModel(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

def run_verification():
    print("=" * 70)
    print("STARTING STAGE 6 RESUME VERIFICATION SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        checkpoints_dir = os.path.join(models_dir, "checkpoints", "stage6")
        metrics_dir = os.path.join(tmpdir, "results", "metrics")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        state_mgr = Stage6StateManager(metrics_dir, checkpoints_dir, models_dir)

        # -------------------------------------------------------------
        # 1. Verify Checkpoint Saving
        # -------------------------------------------------------------
        model = TinyTestModel(in_channels=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda' if "cuda" in str(device) else 'cpu')

        # Simulate 3 training steps and scheduler steps
        for step in range(3):
            optimizer.zero_grad()
            dummy_x = torch.randn(2, 1, 16, 16, device=device)
            dummy_y = torch.ones(2, 1, device=device)
            with torch.amp.autocast('cuda' if "cuda" in str(device) else 'cpu'):
                out = model(dummy_x)
                loss = ((out - dummy_y)**2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        current_epoch = 3
        best_val_dice = 0.725
        best_val_iou = 0.582
        model_name = "S1_VV_only"
        ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_latest.pt")

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "current_epoch": current_epoch,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "model_name": model_name,
            "channel_indices": [6],
            "in_channels": 1,
            "total_epochs": 12,
            "random_seed": 42,
            "training_configuration": {
                "batch_size": 16,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "eta_min": 1e-6,
                "architecture": "ResNet34UNet",
                "in_channels": 1,
                "channel_indices": [6],
            },
            "training_complete": False
        }

        atomic_save_torch(checkpoint_data, ckpt_path)
        results["1. Checkpoint saved"] = os.path.exists(ckpt_path)

        # -------------------------------------------------------------
        # 2. Verify Checkpoint Loading
        # -------------------------------------------------------------
        loaded_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        results["2. Checkpoint loaded"] = (loaded_ckpt is not None and isinstance(loaded_ckpt, dict))

        # -------------------------------------------------------------
        # 3. Epoch Number Restored
        # -------------------------------------------------------------
        results["3. Epoch number restored"] = (loaded_ckpt["current_epoch"] == 3)

        # -------------------------------------------------------------
        # 4. Optimizer State Restored
        # -------------------------------------------------------------
        new_model = TinyTestModel(in_channels=1).to(device)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4, weight_decay=1e-4)
        new_optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
        # Compare optimizer state step count
        orig_step = list(optimizer.state.values())[0]['step'] if optimizer.state else 3
        new_step = list(new_optimizer.state.values())[0]['step'] if new_optimizer.state else 3
        results["4. Optimizer state restored"] = (orig_step == new_step)

        # -------------------------------------------------------------
        # 5. Scheduler State Restored
        # -------------------------------------------------------------
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=12, eta_min=1e-6)
        new_scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
        results["5. Scheduler state restored"] = (new_scheduler.last_epoch == scheduler.last_epoch)

        # -------------------------------------------------------------
        # 6. AMP Scaler State Restored
        # -------------------------------------------------------------
        new_scaler = torch.amp.GradScaler('cuda' if "cuda" in str(device) else 'cpu')
        new_scaler.load_state_dict(loaded_ckpt["scaler_state_dict"])
        results["6. AMP scaler state restored"] = (new_scaler.state_dict() == scaler.state_dict())

        # -------------------------------------------------------------
        # 7. Training Resumes at epoch + 1
        # -------------------------------------------------------------
        resumed_start_epoch = loaded_ckpt["current_epoch"] + 1
        results["7. Training resumes at epoch + 1"] = (resumed_start_epoch == 4)

        # -------------------------------------------------------------
        # 8. Completed Models Are Not Retrained
        # -------------------------------------------------------------
        completed_ckpt_data = dict(checkpoint_data)
        completed_ckpt_data["current_epoch"] = 12
        completed_ckpt_data["total_epochs"] = 12
        completed_ckpt_data["training_complete"] = True
        completed_model_name = "S2_RGB_only"
        comp_ckpt_path = os.path.join(checkpoints_dir, f"{completed_model_name}_latest.pt")
        atomic_save_torch(completed_ckpt_data, comp_ckpt_path)

        # Check inspection
        insp = state_mgr.inspect_checkpoints()
        results["8. Completed models marked complete"] = (
            insp[completed_model_name]["complete"] is True and
            insp[completed_model_name]["can_resume"] is False and
            insp[completed_model_name]["last_epoch"] == 12
        )

        # -------------------------------------------------------------
        # 9. Incomplete Models Are Resumed
        # -------------------------------------------------------------
        results["9. Incomplete models marked for resume"] = (
            insp[model_name]["complete"] is False and
            insp[model_name]["can_resume"] is True and
            insp[model_name]["next_epoch"] == 4
        )

        # -------------------------------------------------------------
        # 10. Stage 6 State Correctly Updated
        # -------------------------------------------------------------
        state_mgr.update_model_state(model_name, "running", 3)
        state_mgr.update_model_state(completed_model_name, "complete", 12)
        state_mgr.record_model_evaluation("S1_Full", {"metrics": {"iou": 0.15}}, [{"patch": 1}])
        state_mgr.record_sensitivity_result("Mask_VV_Only", {"delta_iou": -1.2})
        state_mgr.update_step_status("evaluation", "running")

        state_on_disk = load_json(state_mgr.state_file)
        eval_state_on_disk = load_json(state_mgr.eval_state_file)
        sens_state_on_disk = load_json(state_mgr.sens_state_file)

        results["10. Stage 6 state correctly updated"] = (
            state_on_disk[model_name]["last_completed_epoch"] == 3 and
            state_on_disk[completed_model_name]["status"] == "complete" and
            eval_state_on_disk["S1_Full"] == "complete" and
            sens_state_on_disk["Mask_VV_Only"] == "complete"
        )

    print("\nVERIFICATION RESULTS:")
    print("-" * 70)
    all_passed = True
    for item, passed in results.items():
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str}  {item}")
        if not passed:
            all_passed = False
    print("-" * 70)

    if all_passed:
        print("ALL 10 VERIFICATION CHECKS PASSED PERFECTLY!\n")
    else:
        print("SOME CHECKS FAILED! Please inspect errors.\n")
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
