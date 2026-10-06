"""
Burn Severity and Damage Quantification Agent
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Quantifies multi-tier burn severity (USGS / Copernicus EMS grading),
vegetation consumption indices, perimeter morphology, and spatial footprint.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, AgentResponse

class SeverityQuantifierAgent(BaseAgent):
    """
    Analyzes burn severity tiers, fire morphology, perimeter complexity, and biophysical damage.
    """
    def __init__(self):
        super().__init__(
            name="SeverityQuantifierAgent",
            role="Burn Severity Stratification & Damage Assessment",
            description="Quantifies Copernicus EMS damage tiers, fire perimeter geometry, and biophysical spectral index degradation."
        )

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        self.log("Quantifying burn severity and spatial damage morphology...")

        delineation = context.get("delineation_results", {})
        binary_mask = delineation.get("binary_mask")
        prob_map = delineation.get("probability_map")

        if binary_mask is None:
            return AgentResponse(
                agent_name=self.name,
                status="ERROR",
                data={},
                message="Delineated binary mask not found in context."
            )

        H, W = binary_mask.shape
        burned_pixels = int(binary_mask.sum())
        total_pixels = H * W

        if burned_pixels == 0:
            return AgentResponse(
                agent_name=self.name,
                status="SUCCESS",
                data={
                    "total_burned_area_km2": 0.0,
                    "total_burned_area_acres": 0.0,
                    "severity_tiers": {
                        "low_severity": {"name": "Low Severity", "pixels": 0, "km2": 0.0, "pct_of_fire": 0.0},
                        "moderate_severity": {"name": "Moderate Severity", "pixels": 0, "km2": 0.0, "pct_of_fire": 0.0},
                        "high_severity": {"name": "High Severity", "pixels": 0, "km2": 0.0, "pct_of_fire": 0.0}
                    },
                    "severity_map": np.zeros((H, W), dtype=np.uint8),
                    "fire_perimeter_km": 0.0,
                    "num_fire_clusters": 0,
                    "top_cluster_sizes_km2": [],
                    "bounding_box": {"y_min": 0, "y_max": 0, "x_min": 0, "x_max": 0, "height_px": 0, "width_px": 0},
                    "centroid_pixel": {"y": 0.0, "x": 0.0},
                    "compactness_index": 0.0,
                    "high_severity_ratio": 0.0
                },
                message="No burned area detected in delineated mask."
            )

        # 1. Derive or Extract Burn Severity Map
        dnbr = context.get("dnbr_data")
        optical = context.get("optical_data")

        severity_map = np.zeros((H, W), dtype=np.uint8)

        if dnbr is not None and isinstance(dnbr, np.ndarray):
            # True dNBR available (USGS thresholds)
            # 0: Unburned (< 0.10)
            # 1: Low (0.10 <= dNBR < 0.27)
            # 2: Moderate (0.27 <= dNBR < 0.66)
            # 3: High (>= 0.66)
            severity_map[(dnbr >= 0.10) & (binary_mask == 1)] = 1
            severity_map[(dnbr >= 0.27) & (binary_mask == 1)] = 2
            severity_map[(dnbr >= 0.66) & (binary_mask == 1)] = 3
        elif optical is not None and optical.shape[0] >= 4:
            # Optical RGB + NIR available: calculate NDVI degradation
            # NDVI = (NIR - Red) / (NIR + Red)
            red = optical[2]
            nir = optical[3]
            denom = nir + red + 1e-6
            ndvi = (nir - red) / denom
            # Burned areas have negative or very low NDVI (< 0.15)
            severity_map[(ndvi < 0.25) & (binary_mask == 1)] = 1
            severity_map[(ndvi < 0.15) & (binary_mask == 1)] = 2
            severity_map[(ndvi < 0.05) & (binary_mask == 1)] = 3
        else:
            # Map based on model delineation confidence
            severity_map[(prob_map >= 0.50) & (binary_mask == 1)] = 1
            severity_map[(prob_map >= 0.70) & (binary_mask == 1)] = 2
            severity_map[(prob_map >= 0.90) & (binary_mask == 1)] = 3

        # 2. Stratify Severity Classes
        tiers = {
            "low_severity": {
                "name": "Low Severity (Grade 1 - Negligible to Slight Damage)",
                "pixels": int((severity_map == 1).sum()),
                "km2": round(float((severity_map == 1).sum() * 0.0001), 2),
                "pct_of_fire": round(float((severity_map == 1).sum() / (burned_pixels + 1e-8) * 100.0), 1)
            },
            "moderate_severity": {
                "name": "Moderate Severity (Grade 2 - Substantial Damage)",
                "pixels": int((severity_map == 2).sum()),
                "km2": round(float((severity_map == 2).sum() * 0.0001), 2),
                "pct_of_fire": round(float((severity_map == 2).sum() / (burned_pixels + 1e-8) * 100.0), 1)
            },
            "high_severity": {
                "name": "High Severity (Grade 3 - Complete Destruction)",
                "pixels": int((severity_map == 3).sum()),
                "km2": round(float((severity_map == 3).sum() * 0.0001), 2),
                "pct_of_fire": round(float((severity_map == 3).sum() / (burned_pixels + 1e-8) * 100.0), 1)
            }
        }

        # 3. Fire Morphology & Geometry
        ys, xs = np.where(binary_mask == 1)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        center_y, center_x = float(ys.mean()), float(xs.mean())

        # Perimeter extraction using morphological gradient
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(binary_mask, structure=struct)
        boundary = binary_mask - eroded.astype(np.uint8)
        perimeter_pixels = int(boundary.sum())
        # 1 pixel edge = 10m = 0.01 km
        perimeter_km = float(perimeter_pixels * 0.01)

        # Connected component cluster analysis
        labeled, num_clusters = ndimage.label(binary_mask)
        cluster_sizes = ndimage.sum(binary_mask, labeled, range(1, num_clusters + 1))
        major_clusters = [round(float(s * 0.0001), 2) for s in sorted(cluster_sizes, reverse=True)[:5]]

        # Isoperimetric quotient (Compactness = 4 * pi * Area / Perimeter^2)
        area_m2 = burned_pixels * 100.0
        perimeter_m = perimeter_pixels * 10.0
        compactness = float((4.0 * np.pi * area_m2) / (perimeter_m**2 + 1e-6))

        results = {
            "total_burned_area_km2": round(float(burned_pixels * 0.0001), 2),
            "total_burned_area_acres": round(float(burned_pixels * 0.0001 * 247.105), 1),
            "severity_tiers": tiers,
            "severity_map": severity_map,
            "fire_perimeter_km": round(perimeter_km, 2),
            "num_fire_clusters": int(num_clusters),
            "top_cluster_sizes_km2": major_clusters,
            "bounding_box": {
                "y_min": y_min, "y_max": y_max,
                "x_min": x_min, "x_max": x_max,
                "height_px": y_max - y_min + 1,
                "width_px": x_max - x_min + 1
            },
            "centroid_pixel": {"y": round(center_y, 1), "x": round(center_x, 1)},
            "compactness_index": round(compactness, 4),
            "high_severity_ratio": round(tiers["high_severity"]["pct_of_fire"], 1)
        }

        self.log(f"Severity analysis: Low={tiers['low_severity']['km2']} km², Mod={tiers['moderate_severity']['km2']} km², High={tiers['high_severity']['km2']} km²")
        self.log(f"Morphology: Perimeter={perimeter_km:.1f} km, Clusters={num_clusters}, Compactness={compactness:.3f}")

        return AgentResponse(
            agent_name=self.name,
            status="SUCCESS",
            data=results,
            message=f"Quantified {results['total_burned_area_km2']} km² across {num_clusters} clusters. High severity constitutes {tiers['high_severity']['pct_of_fire']}%."
        )
