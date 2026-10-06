"""
Sensor Arbitrator Agent
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Evaluates incoming satellite data across optical and SAR modalities, assesses atmospheric
conditions (cloud cover, smoke, sun angle), detects missing bands, and arbitrates
the optimal inference mode and model checkpoint.
"""

import os
from typing import Dict, Any
import numpy as np
from .base_agent import BaseAgent, AgentResponse

class SensorArbitratorAgent(BaseAgent):
    """
    Arbitrates multi-sensor ingestion quality and selects the optimal deep learning model.
    """
    def __init__(self, models_dir: str):
        super().__init__(
            name="SensorArbitratorAgent",
            role="Multi-Modal Sensor Quality Assessment & Model Routing",
            description="Analyzes cloud cover, smoke occlusion, SAR backscatter integrity, and routes inference to the best model."
        )
        self.models_dir = models_dir

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        self.log("Evaluating multi-sensor ingestion package...")

        optical = context.get("optical_data")  # (C, H, W)
        sar = context.get("sar_data")          # (C, H, W)
        cloud_mask = context.get("cloud_mask")  # (H, W) or None
        meta = context.get("metadata", {})

        has_optical = optical is not None and isinstance(optical, np.ndarray) and optical.size > 0
        has_sar = sar is not None and isinstance(sar, np.ndarray) and sar.size > 0

        if not has_optical and not has_sar:
            return AgentResponse(
                agent_name=self.name,
                status="ERROR",
                data={},
                message="No valid satellite sensor data provided in context."
            )

        # 1. Analyze Optical Quality
        cloud_cover_pct = 0.0
        optical_quality_score = 0.0
        optical_channels = 0

        if has_optical:
            optical_channels = optical.shape[0]
            if cloud_mask is not None:
                cloud_cover_pct = float(np.mean(cloud_mask > 0) * 100.0)
            else:
                # Estimate cloud cover via high blue/brightness threshold if Blue channel exists
                # In normalized [0, 1] reflectance, clouds typically > 0.35 in all visible bands
                if optical_channels >= 3:
                    # In Sentinel-2 normalized reflectance or RGB images, thick clouds are near-white saturation (> 0.88)
                    bright_pixels = np.all(optical[:3] > 0.88, axis=0)
                    cloud_cover_pct = float(np.mean(bright_pixels) * 100.0)

            # Optical quality decays as cloud coverage increases
            optical_quality_score = max(0.0, 1.0 - (cloud_cover_pct / 100.0))
            self.log(f"Optical analysis: {optical_channels} bands, cloud cover={cloud_cover_pct:.1f}%, quality={optical_quality_score:.2f}")

        # 2. Analyze SAR Quality
        sar_quality_score = 0.0
        sar_channels = 0
        if has_sar:
            sar_channels = sar.shape[0]
            valid_ratio = float(np.mean(~np.isnan(sar) & (sar > 0)))
            sar_quality_score = min(1.0, valid_ratio)
            self.log(f"SAR analysis: {sar_channels} bands, valid pixel ratio={valid_ratio:.2f}, quality={sar_quality_score:.2f}")

        # 3. Model Routing Logic
        decision = {}
        if has_optical and has_sar:
            if cloud_cover_pct >= 60.0:
                # Severe cloud cover -> SAR dominates
                mode = "SAR_EXCLUSIVE"
                model_name = "best_s1_baseline_model.pt" if sar_channels >= 3 else "ablation_S1_VV_VH.pt"
                weights = {"optical": 0.05, "sar": 0.95}
                reason = f"Severe cloud cover ({cloud_cover_pct:.1f}%) renders optical bands unreliable. Routing to all-weather Sentinel-1 SAR."
            elif cloud_cover_pct >= 25.0:
                # Partial cloud cover -> SAR weighted fusion
                mode = "MULTIMODAL_FUSION"
                model_name = "best_fusion_model.pt" if (optical_channels >= 6 and sar_channels >= 3) else "ablation_S1_VV_VH.pt"
                weights = {"optical": 0.40, "sar": 0.60}
                reason = f"Moderate cloud cover ({cloud_cover_pct:.1f}%). Deploying Multimodal Fusion with SAR penetration emphasis."
            else:
                # Clear sky -> Full Multimodal Fusion
                mode = "MULTIMODAL_FUSION"
                if optical_channels >= 6 and sar_channels >= 3:
                    model_name = "best_fusion_model.pt"
                elif optical_channels >= 3:
                    model_name = "ablation_S2_RGB_only.pt"
                else:
                    model_name = "best_s1_baseline_model.pt"
                weights = {"optical": 0.65, "sar": 0.35}
                reason = f"Clear sky conditions ({cloud_cover_pct:.1f}% clouds). Deploying high-fidelity Multimodal Fusion network."
        elif has_optical and not has_sar:
            mode = "OPTICAL_ONLY"
            if optical_channels >= 6:
                model_name = "best_s2_baseline_model.pt"
            elif optical_channels >= 3:
                model_name = "ablation_S2_RGB_only.pt"
            else:
                model_name = "ablation_S2_RGB_only.pt"
            weights = {"optical": 1.0, "sar": 0.0}
            reason = f"Only optical data available ({optical_channels} bands). Routing to Sentinel-2 Optical baseline."
        else:
            mode = "SAR_ONLY"
            if sar_channels >= 3:
                model_name = "best_s1_baseline_model.pt"
            elif sar_channels == 2:
                model_name = "ablation_S1_VV_VH.pt"
            else:
                model_name = "ablation_S1_VV_only.pt"
            weights = {"optical": 0.0, "sar": 1.0}
            reason = f"Only SAR data available ({sar_channels} bands). Routing to all-weather Sentinel-1 SAR baseline."

        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            self.log(f"Warning: Preferred checkpoint {model_name} not found, checking fallback.", level="WARN")
            # Fallback to any existing model
            for alt in ["best_fusion_model.pt", "best_s2_baseline_model.pt", "best_s1_baseline_model.pt", "ablation_S2_RGB_only.pt"]:
                alt_path = os.path.join(self.models_dir, alt)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    model_name = alt
                    break

        decision = {
            "mode": mode,
            "recommended_model": model_name,
            "model_path": model_path,
            "cloud_cover_pct": round(cloud_cover_pct, 2),
            "optical_quality_score": round(optical_quality_score, 3),
            "sar_quality_score": round(sar_quality_score, 3),
            "sensor_weights": weights,
            "reasoning": reason,
            "has_optical": has_optical,
            "has_sar": has_sar,
            "optical_channels": optical_channels,
            "sar_channels": sar_channels
        }

        self.log(f"Arbitration decision: Mode={mode}, Model={model_name}")
        return AgentResponse(
            agent_name=self.name,
            status="SUCCESS",
            data=decision,
            message=reason
        )
