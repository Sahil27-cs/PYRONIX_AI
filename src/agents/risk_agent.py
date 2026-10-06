"""
Environmental Risk Assessment and Tactical Hazard Agent
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Evaluates secondary cascading post-wildfire hazards:
1. Debris flow and post-fire landslide vulnerability
2. Hydrophobic soil water repellency and flash flood runoff potential
3. Fire containment complexity and flank instability
4. Wildland-Urban Interface (WUI) asset exposure rating
5. Tactical BAER (Burned Area Emergency Response) interventions
"""

import numpy as np
from typing import Dict, Any

from .base_agent import BaseAgent, AgentResponse

class RiskAssessmentAgent(BaseAgent):
    """
    Evaluates cascading environmental risks and generates operational response recommendations.
    """
    def __init__(self):
        super().__init__(
            name="RiskAssessmentAgent",
            role="Secondary Hazard & Tactical Emergency Risk Assessment",
            description="Evaluates debris flow hazard, hydrophobic soil runoff, WUI exposure, and prescribes BAER emergency interventions."
        )

    def process(self, context: Dict[str, Any]) -> AgentResponse:
        self.log("Assessing cascading post-fire hazards and environmental vulnerability...")

        severity = context.get("severity_results", {})
        delineation = context.get("delineation_results", {})
        meta = context.get("metadata", {})

        burned_km2 = severity.get("total_burned_area_km2", 0.0)
        perimeter_km = severity.get("fire_perimeter_km", 0.0)
        compactness = severity.get("compactness_index", 1.0)
        high_sev_pct = severity.get("high_severity_ratio", 0.0)
        num_clusters = severity.get("num_fire_clusters", 1)

        if burned_km2 == 0.0:
            return AgentResponse(
                agent_name=self.name,
                status="SUCCESS",
                data={
                    "overall_threat_level": "NEGLIGIBLE",
                    "threat_score": 0.0,
                    "debris_flow_hazard": {
                        "level": "NONE",
                        "score": 0.0,
                        "assessment": "No active fire damage detected in monitored region."
                    },
                    "containment_complexity": {
                        "level": "NONE",
                        "shape_irregularity_ratio": 0.0,
                        "num_separate_clusters": 0,
                        "assessment": "No active perimeter to contain."
                    },
                    "soil_hydrophobicity_risk": "NONE",
                    "tactical_recommendations": [
                        "Continue standard baseline satellite monitoring.",
                        "No emergency BAER interventions required."
                    ]
                },
                message="No fire threat detected."
            )

        # 1. Post-Fire Debris Flow / Mudslide Hazard Rating
        # Higher high-severity burn ratio + large area = elevated hydrophobic sediment release
        debris_score = (high_sev_pct * 0.5) + (min(burned_km2, 100.0) / 100.0 * 30.0) + (1.0 - min(compactness, 1.0)) * 20.0
        if debris_score >= 45.0 or (high_sev_pct >= 25.0 and burned_km2 >= 10.0):
            debris_level = "HIGH / EXTREME"
            debris_desc = "Severe risk of catastrophic post-fire debris flows and mudslides upon precipitation."
        elif debris_score >= 20.0 or high_sev_pct >= 10.0:
            debris_level = "MODERATE"
            debris_desc = "Substantial loss of ground vegetation; localized rill erosion and shallow gullying expected."
        else:
            debris_level = "LOW"
            debris_desc = "Primarily low-severity burn with intact root structures. Low sediment mobilization risk."

        # 2. Containment Complexity Index (Flank irregularity & spotting potential)
        # Ratio of perimeter to square root of area
        area_m2 = burned_km2 * 1e6
        perim_m = perimeter_km * 1e3
        shape_ratio = perim_m / (np.sqrt(area_m2) + 1e-5)

        if shape_ratio >= 15.0 or num_clusters >= 5:
            containment_level = "COMPLEX / FRAGMENTED"
            containment_desc = "Highly convoluted fire perimeter with potential multiple spot fires and uneven flanks."
        elif shape_ratio >= 8.0:
            containment_level = "MODERATE"
            containment_desc = "Moderate perimeter irregularity requiring standard flank containment lines."
        else:
            containment_level = "UNIFORM"
            containment_desc = "Compact circular/elliptical burn perimeter, favorable for direct line containment."

        # 3. Overall Incident Threat Level
        threat_score = (min(burned_km2, 100.0) / 100.0 * 40.0) + (high_sev_pct * 0.4) + (debris_score * 0.2)
        if threat_score >= 50.0 or burned_km2 >= 50.0:
            overall_threat = "CRITICAL / LEVEL 4"
        elif threat_score >= 25.0 or burned_km2 >= 15.0:
            overall_threat = "HIGH / LEVEL 3"
        elif threat_score >= 10.0 or burned_km2 >= 3.0:
            overall_threat = "ELEVATED / LEVEL 2"
        else:
            overall_threat = "LOW / LEVEL 1"

        # 4. Tactical BAER & Emergency Response Recommendations
        recommendations = [
            "Maintain continuous multi-sensor Sentinel-1 SAR and Sentinel-2 optical satellite monitoring on 5-day revisit cadence.",
            "Establish perimeter containment buffer and ground patrol monitoring along active flank borders.",
            "Conduct airborne infrared reconnaissance to verify residual subterranean smoldering and hot spots."
        ]
        if debris_level in ["HIGH / EXTREME", "MODERATE"]:
            recommendations.append("Immediate deployment of USGS / BAER soil burn severity emergency assessment team.")
            recommendations.append("Install hydrologic rain gauges and sediment retention barriers in downstream drainage channels.")
            recommendations.append("Establish early-warning flash-flood / debris-flow evacuation triggers for downstream communities.")

        if containment_level == "COMPLEX / FRAGMENTED":
            recommendations.append("Utilize satellite SAR all-weather updates to monitor active flank smoldering through smoke.")
            recommendations.append("Prioritize perimeter stabilization along isolated uncontained satellite spot fire clusters.")

        if burned_km2 >= 20.0 or high_sev_pct >= 20.0:
            recommendations.append("Implement aerial mulching / hydro-seeding on slopes > 25 degrees to stabilize hydrophobic ash beds.")
            recommendations.append("Conduct infrastructure inspection on nearby roadways, culverts, and electrical transmission corridors.")

        risk_data = {
            "overall_threat_level": overall_threat,
            "threat_score": round(threat_score, 1),
            "debris_flow_hazard": {
                "level": debris_level,
                "score": round(debris_score, 1),
                "assessment": debris_desc
            },
            "containment_complexity": {
                "level": containment_level,
                "shape_irregularity_ratio": round(shape_ratio, 2),
                "num_separate_clusters": num_clusters,
                "assessment": containment_desc
            },
            "soil_hydrophobicity_risk": "SEVERE" if high_sev_pct >= 20.0 else ("MODERATE" if high_sev_pct >= 8.0 else "MILD"),
            "tactical_recommendations": recommendations
        }

        self.log(f"Risk Assessment: Threat={overall_threat}, Debris Flow={debris_level}, Containment={containment_level}")
        return AgentResponse(
            agent_name=self.name,
            status="SUCCESS",
            data=risk_data,
            message=f"Threat Level: {overall_threat}. Debris flow hazard rated {debris_level} with {len(recommendations)} prescribed interventions."
        )
