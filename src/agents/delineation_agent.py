"""
Deep Learning Delineation Agent
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Executes deep semantic segmentation using trained PyTorch ResNet-34 U-Net checkpoints.
Performs tile extraction, mixed precision inference, smooth probability reconstruction,
and binary thresholding to output definitive burned area perimeters.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset

from .base_agent import BaseAgent, AgentResponse
from ..models.unet import ResNet34UNet

class DelineationAgent(BaseAgent):
    """
    Executes deep learning segmentation inference on multi-spectral / SAR satellite data.
    """
    def __init__(self, stats_json: str, s1_stats_json: Optional[str] = None, device: str = "cuda"):
        super().__init__(
            name="DelineationAgent",
            role="Deep Learning Burned-Area Semantic Segmentation",
            description="Runs deep ResNet-34 U-Net models to generate calibrated burned-area probability maps and binary masks."
        )
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.model_cache = {}

        # Load normalization parameters
        with open(stats_json, "r") as f:
            self.fusion_stats = json.load(f)

        if s1_stats_json and os.path.exists(s1_stats_json):
            with open(s1_stats_json, "r") as f:
                self.s1_stats = json.load(f)
        else:
            self.s1_stats = self.fusion_stats

    def _get_model(self, model_path: str, in_channels: int) -> ResNet34UNet:
        """Loads or retrieves cached model weights."""
        if model_path in self.model_cache:
            return self.model_cache[model_path]

        self.log(f"Loading checkpoint weights from {os.path.basename(model_path)} (in_channels={in_channels})...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model = ResNet34UNet(in_channels=in_channels, num_classes=1, pretrained=False).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model_cache[model_path] = model
        return model

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        self.log("Beginning deep learning segmentation delineation...")

        arbitration = context.get("arbitration_decision", {})
        model_path = context.get("model_path") or arbitration.get("model_path")
        mode = arbitration.get("mode", "AUTO")
        threshold = float(context.get("threshold", 0.5))

        optical = context.get("optical_data")  # (C, H, W)
        sar = context.get("sar_data")          # (C, H, W)

        if not model_path or not os.path.exists(model_path):
            return AgentResponse(
                agent_name=self.name,
                status="ERROR",
                data={},
                message=f"Model checkpoint not found: {model_path}"
            )

        # Determine input channels based on model filename / requested mode
        model_name = os.path.basename(model_path)
        if "fusion" in model_name.lower():
            in_channels = 9
            # Form 9-channel array: S2 (6 ch) + S1 (3 ch)
            if optical is None or sar is None:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="Fusion model requires both optical and SAR data.")
            # Pad or slice optical to 6 channels
            if optical.shape[0] < 6:
                pad = np.zeros((6 - optical.shape[0], optical.shape[1], optical.shape[2]), dtype=np.float32)
                opt_6 = np.concatenate([optical, pad], axis=0)
            else:
                opt_6 = optical[:6]
            s1_3 = sar[:3]
            raw_input = np.concatenate([opt_6, s1_3], axis=0)
            mean = np.array(self.fusion_stats["mean"], dtype=np.float32).reshape(9, 1, 1)
            std = np.array(self.fusion_stats["std"], dtype=np.float32).reshape(9, 1, 1)
            norm_input = (raw_input - mean) / std

        elif "s2_baseline" in model_name.lower():
            in_channels = 6
            if optical is None:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="S2 baseline requires optical data.")
            if optical.shape[0] < 6:
                pad = np.zeros((6 - optical.shape[0], optical.shape[1], optical.shape[2]), dtype=np.float32)
                opt_6 = np.concatenate([optical, pad], axis=0)
            else:
                opt_6 = optical[:6]
            mean = np.array(self.fusion_stats["mean"][:6], dtype=np.float32).reshape(6, 1, 1)
            std = np.array(self.fusion_stats["std"][:6], dtype=np.float32).reshape(6, 1, 1)
            norm_input = (opt_6 - mean) / std

        elif "rgb_only" in model_name.lower():
            in_channels = 3
            if optical is None or optical.shape[0] < 3:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="RGB model requires at least 3 optical bands.")
            rgb = optical[:3]
            mean = np.array(self.fusion_stats["mean"][:3], dtype=np.float32).reshape(3, 1, 1)
            std = np.array(self.fusion_stats["std"][:3], dtype=np.float32).reshape(3, 1, 1)
            norm_input = (rgb - mean) / std

        elif "s1_vv_vh" in model_name.lower():
            in_channels = 2
            if sar is None or sar.shape[0] < 2:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="Dual-pol SAR model requires 2 SAR bands.")
            vv_vh = sar[:2]
            mean = np.array(self.fusion_stats["mean"][6:8], dtype=np.float32).reshape(2, 1, 1)
            std = np.array(self.fusion_stats["std"][6:8], dtype=np.float32).reshape(2, 1, 1)
            norm_input = (vv_vh - mean) / std

        elif "vv_only" in model_name.lower():
            in_channels = 1
            if sar is None:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="VV model requires SAR data.")
            vv = sar[0:1]
            mean = np.array(self.fusion_stats["mean"][6:7], dtype=np.float32).reshape(1, 1, 1)
            std = np.array(self.fusion_stats["std"][6:7], dtype=np.float32).reshape(1, 1, 1)
            norm_input = (vv - mean) / std

        elif "vh_only" in model_name.lower():
            in_channels = 1
            if sar is None:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="VH model requires SAR data.")
            vh = sar[1:2] if sar.shape[0] >= 2 else sar[0:1]
            mean = np.array(self.fusion_stats["mean"][7:8], dtype=np.float32).reshape(1, 1, 1)
            std = np.array(self.fusion_stats["std"][7:8], dtype=np.float32).reshape(1, 1, 1)
            norm_input = (vh - mean) / std

        else:
            # Default: S1 Baseline (3 ch: VV, VH, ratio)
            in_channels = 3
            if sar is None or sar.shape[0] < 3:
                return AgentResponse(agent_name=self.name, status="ERROR", data={}, message="S1 baseline requires 3 SAR bands.")
            s1_3 = sar[:3]
            mean = np.array(self.s1_stats["mean"], dtype=np.float32).reshape(3, 1, 1)
            std = np.array(self.s1_stats["std"], dtype=np.float32).reshape(3, 1, 1)
            norm_input = (s1_3 - mean) / std

        # Tiled inference
        C, H, W = norm_input.shape
        patch_size = 256
        stride = 128

        patches, positions = self._tile(norm_input, patch_size, stride)
        self.log(f"Executing inference on {len(patches)} patches with model {model_name}...")

        model = self._get_model(model_path, in_channels)

        batch_size = 16
        patch_tensor = torch.from_numpy(patches)
        loader = DataLoader(TensorDataset(patch_tensor), batch_size=batch_size, shuffle=False)

        pred_list = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                with torch.amp.autocast('cuda' if "cuda" in str(self.device) else 'cpu'):
                    out = model(bx)
                    probs = torch.sigmoid(out).cpu().squeeze(1).numpy()
                pred_list.append(probs)

        all_preds = np.concatenate(pred_list, axis=0)

        # Smooth reconstruction
        prob_map = self._reconstruct(all_preds, positions, H, W, patch_size)
        binary_mask = (prob_map >= threshold).astype(np.uint8)

        # Epistemic uncertainty map (entropy-like margin: max at 0.5)
        uncertainty_map = 1.0 - 2.0 * np.abs(prob_map - 0.5)

        burned_pixels = int(binary_mask.sum())
        # 10m resolution pixel = 100 m^2 = 0.0001 km^2 = 0.0247105 acres
        burned_area_km2 = float(burned_pixels * 0.0001)
        burned_area_acres = float(burned_area_km2 * 247.105)
        total_scene_km2 = float(H * W * 0.0001)
        burn_pct = float((burned_pixels / (H * W)) * 100.0)

        self.log(f"Delineation complete: {burned_area_km2:.2f} km² ({burned_area_acres:.1f} acres, {burn_pct:.2f}% of scene)")

        results = {
            "model_used": model_name,
            "threshold": threshold,
            "burned_pixels": burned_pixels,
            "burned_area_km2": round(burned_area_km2, 2),
            "burned_area_acres": round(burned_area_acres, 1),
            "total_scene_km2": round(total_scene_km2, 2),
            "burn_percentage": round(burn_pct, 2),
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "uncertainty_map": uncertainty_map,
            "mean_confidence_burned": float(prob_map[binary_mask == 1].mean()) if burned_pixels > 0 else 0.0,
            "mean_confidence_unburned": float((1.0 - prob_map[binary_mask == 0]).mean()) if (H*W - burned_pixels) > 0 else 0.0
        }

        return AgentResponse(
            agent_name=self.name,
            status="SUCCESS",
            data=results,
            message=f"Delineated {burned_area_km2:.2f} km² ({burned_area_acres:.1f} acres) burned area with model {model_name}."
        )

    def _tile(self, img, patch_size=256, stride=128):
        C, H, W = img.shape
        patches = []
        positions = []
        y_steps = list(range(0, H - patch_size + 1, stride))
        if (H - patch_size) % stride != 0:
            y_steps.append(H - patch_size)
        x_steps = list(range(0, W - patch_size + 1, stride))
        if (W - patch_size) % stride != 0:
            x_steps.append(W - patch_size)

        for y in y_steps:
            for x in x_steps:
                patches.append(img[:, y:y+patch_size, x:x+patch_size])
                positions.append((y, x))
        return np.array(patches, dtype=np.float32), positions

    def _reconstruct(self, preds, positions, H, W, patch_size=256):
        prob = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        for p, (y, x) in zip(preds, positions):
            prob[y:y+patch_size, x:x+patch_size] += p
            weight[y:y+patch_size, x:x+patch_size] += 1.0
        weight = np.maximum(weight, 1.0)
        return prob / weight
