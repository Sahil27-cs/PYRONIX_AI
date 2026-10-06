"""
Wildfire Multi-Agent Orchestrator
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Coordinates the end-to-end multi-agent workflow:
1. SensorArbitratorAgent: Ingestion assessment & model routing
2. DelineationAgent: Deep semantic segmentation
3. SeverityQuantifierAgent: Copernicus EMS damage stratification
4. RiskAssessmentAgent: Debris flow, hydrological hazard & tactical prescription
5. Synthesis Engine: Formats executive Situation Reports (SitRep) and publication figures.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from .base_agent import BaseAgent, AgentResponse
from .sensor_arbitrator import SensorArbitratorAgent
from .delineation_agent import DelineationAgent
from .severity_agent import SeverityQuantifierAgent
from .risk_agent import RiskAssessmentAgent

class WildfireOrchestrator:
    """
    Master coordinator orchestrating autonomous domain agents for satellite wildfire intelligence.
    """
    def __init__(self, models_dir: str, stats_json: str, s1_stats_json: Optional[str] = None, device: str = "cuda"):
        self.models_dir = models_dir
        self.stats_json = stats_json
        self.s1_stats_json = s1_stats_json
        self.device = device

        # Initialize specialist agents
        self.arbitrator = SensorArbitratorAgent(models_dir=models_dir)
        self.delineator = DelineationAgent(stats_json=stats_json, s1_stats_json=s1_stats_json, device=device)
        self.severity_quantifier = SeverityQuantifierAgent()
        self.risk_assessor = RiskAssessmentAgent()

    def run_analysis(self, context: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end multi-agent analysis on the provided satellite scene context.
        """
        start_time = time.time()
        incident_name = context.get("incident_name", "Wildfire Incident")
        print("\n" + "=" * 75)
        print(f"WILDFIRE MULTI-AGENT ORCHESTRATION PIPELINE: {incident_name}")
        print("=" * 75)

        # Step 1: Sensor Arbitration
        print("\n[PHASE 1/4] Sensor Arbitration & Routing Agent...")
        arbitration_resp = self.arbitrator.process(context)
        if arbitration_resp.status == "ERROR":
            raise RuntimeError(f"SensorArbitratorAgent failed: {arbitration_resp.message}")
        context["arbitration_decision"] = arbitration_resp.data

        # Step 2: Deep Learning Delineation
        print("\n[PHASE 2/4] Deep Semantic Delineation Agent...")
        delineation_resp = self.delineator.process(context)
        if delineation_resp.status == "ERROR":
            raise RuntimeError(f"DelineationAgent failed: {delineation_resp.message}")
        context["delineation_results"] = delineation_resp.data

        # Step 3: Burn Severity & Damage Quantification
        print("\n[PHASE 3/4] Burn Severity & Morphology Agent...")
        severity_resp = self.severity_quantifier.process(context)
        if severity_resp.status == "ERROR":
            raise RuntimeError(f"SeverityQuantifierAgent failed: {severity_resp.message}")
        context["severity_results"] = severity_resp.data

        # Step 4: Environmental & Cascading Risk Assessment
        print("\n[PHASE 4/4] Environmental Risk & Tactical Intervention Agent...")
        risk_resp = self.risk_assessor.process(context)
        if risk_resp.status == "ERROR":
            raise RuntimeError(f"RiskAssessmentAgent failed: {risk_resp.message}")
        context["risk_results"] = risk_resp.data

        elapsed = time.time() - start_time
        print("\n" + "=" * 75)
        print(f"ALL 4 SPECIALIST AGENTS COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
        print("=" * 75)

        # Synthesize Executive Situation Report (SitRep)
        sitrep = self._synthesize_sitrep(incident_name, context, elapsed)

        # Save artifacts if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{incident_name.lower().replace(' ', '_')}_sitrep.json")
            md_path = os.path.join(output_dir, f"{incident_name.lower().replace(' ', '_')}_sitrep.md")

            # Serializable JSON dictionary (strip large ndarrays)
            serializable_sitrep = self._make_serializable(sitrep)
            with open(json_path, "w") as f:
                json.dump(serializable_sitrep, f, indent=2)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(sitrep["markdown_report"])

            print(f"Saved Situation Report JSON to: {json_path}")
            print(f"Saved Situation Report Markdown to: {md_path}")

            # Generate multi-agent visualization figure
            fig_path = os.path.join(output_dir, f"{incident_name.lower().replace(' ', '_')}_multiagent_overview.png")
            self._generate_overview_plot(context, sitrep, fig_path)
            print(f"Saved Multi-Agent Overview Plot to: {fig_path}")

        return sitrep

    def _synthesize_sitrep(self, incident_name: str, context: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        """Synthesizes structured findings into an executive markdown Situation Report."""
        arb = context["arbitration_decision"]
        delin = context["delineation_results"]
        sev = context["severity_results"]
        risk = context["risk_results"]
        tiers = sev.get("severity_tiers", {})

        md = f"""# SATELLITE WILDFIRE SITUATION REPORT (SITREP)
**Incident Name**: {incident_name}  
**Generated By**: Autonomous Satellite Multi-Agent Wildfire Intelligence System  
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Processing Duration**: {elapsed:.2f} seconds  

---

## 1. EXECUTIVE SUMMARY & THREAT MATRIX
* **Overall Incident Threat Level**: **{risk.get('overall_threat_level', 'UNKNOWN')}** (Score: {risk.get('threat_score', 0.0)}/100)
* **Total Delineated Burned Area**: **{delin.get('burned_area_km2', 0.0):.2f} km²** ({delin.get('burned_area_acres', 0.0):.1f} acres)
* **Scene Burn Ratio**: **{delin.get('burn_percentage', 0.0):.2f}%** of monitored area ({delin.get('total_scene_km2', 0.0):.1f} km² total extent)
* **Fire Perimeter Boundary**: **{sev.get('fire_perimeter_km', 0.0):.2f} km**
* **Active Fire Clusters**: **{sev.get('num_fire_clusters', 0)}** distinct perimeter clusters
* **Containment Complexity**: **{risk.get('containment_complexity', {}).get('level', 'N/A')}** (Irregularity ratio: {risk.get('containment_complexity', {}).get('shape_irregularity_ratio', 0.0)})
* **Post-Fire Debris Flow Hazard**: **{risk.get('debris_flow_hazard', {}).get('level', 'N/A')}** ({risk.get('debris_flow_hazard', {}).get('assessment', 'N/A')})

---

## 2. MULTI-MODAL SENSOR ARBITRATION
* **Arbitration Mode**: `{arb.get('mode', 'N/A')}`
* **Selected Deep Learning Backbone**: `{arb.get('recommended_model', 'N/A')}`
* **Optical Quality Score**: {arb.get('optical_quality_score', 0.0):.2f} (Estimated Cloud Cover: {arb.get('cloud_cover_pct', 0.0):.1f}%)
* **SAR Quality Score**: {arb.get('sar_quality_score', 0.0):.2f} (Synthetic Aperture Radar Backscatter Integrity)
* **Sensor Weights**: Optical: {arb.get('sensor_weights', {}).get('optical', 0.0) * 100:.0f}%, SAR: {arb.get('sensor_weights', {}).get('sar', 0.0) * 100:.0f}%
* **Operational Rationale**: {arb.get('reasoning', 'N/A')}

---

## 3. COPERNICUS EMS DAMAGE STRATIFICATION
| Damage / Severity Tier | Burned Area (km²) | Percent of Fire | Field Damage Description |
| :--- | :---: | :---: | :--- |
| **Grade 1: Low Severity** | {tiers.get('low_severity', {}).get('km2', 0.0)} km² | {tiers.get('low_severity', {}).get('pct_of_fire', 0.0)}% | Surface litter scorched, canopy largely intact. |
| **Grade 2: Moderate Severity** | {tiers.get('moderate_severity', {}).get('km2', 0.0)} km² | {tiers.get('moderate_severity', {}).get('pct_of_fire', 0.0)}% | Significant understory consumption, partial crown scorch. |
| **Grade 3: High Severity** | {tiers.get('high_severity', {}).get('km2', 0.0)} km² | {tiers.get('high_severity', {}).get('pct_of_fire', 0.0)}% | Complete canopy mortality, deep ash bed, hydrophobic soil. |

---

## 4. TACTICAL & BAER INTERVENTION PRESCRIPTIONS
"""
        for i, rec in enumerate(risk["tactical_recommendations"], 1):
            md += f"{i}. **{rec}**\n"

        md += "\n---\n*Report generated autonomously by Satellite Wildfire AI Multi-Agent System.*\n"

        return {
            "incident_name": incident_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "processing_time_sec": round(elapsed, 2),
            "threat_level": risk["overall_threat_level"],
            "burned_area_km2": delin["burned_area_km2"],
            "burned_area_acres": delin["burned_area_acres"],
            "perimeter_km": sev["fire_perimeter_km"],
            "arbitration": arb,
            "delineation": delin,
            "severity": sev,
            "risk": risk,
            "markdown_report": md
        }

    def _generate_overview_plot(self, context: Dict[str, Any], sitrep: Dict[str, Any], save_path: str):
        """Generates a comprehensive 6-panel multi-agent overview plot."""
        optical = context.get("optical_data")
        sar = context.get("sar_data")
        dnbr = context.get("dnbr_data")
        delin = context["delineation_results"]
        sev = context["severity_results"]
        risk = context["risk_results"]

        prob_map = delin["probability_map"]
        bin_mask = delin["binary_mask"]
        sev_map = sev.get("severity_map")

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # Panel 1: S2 Optical RGB or SAR if no optical
        if optical is not None and optical.shape[0] >= 3:
            rgb = np.stack([optical[2], optical[1], optical[0]], axis=-1)
            rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title(f"1. Sensor Arbitration: Optical Input\n(Cloud: {sitrep['arbitration']['cloud_cover_pct']}%)", fontweight="bold", fontsize=11)
        else:
            axes[0, 0].imshow(sar[0], cmap="gray")
            axes[0, 0].set_title("1. Sensor Arbitration: SAR Input", fontweight="bold", fontsize=11)
        axes[0, 0].axis("off")

        # Panel 2: SAR VH or Backscatter
        if sar is not None and sar.shape[0] >= 2:
            sar_vh = np.clip(sar[1] / np.percentile(sar[1], 98), 0, 1)
            axes[0, 1].imshow(sar_vh, cmap="gray")
            axes[0, 1].set_title(f"2. Sensor Arbitration: SAR VH\n(Quality: {sitrep['arbitration']['sar_quality_score']})", fontweight="bold", fontsize=11)
        elif dnbr is not None:
            axes[0, 1].imshow(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=0.8)
            axes[0, 1].set_title("2. Reference dNBR Burn Ratio", fontweight="bold", fontsize=11)
        else:
            axes[0, 1].axis("off")
        axes[0, 1].axis("off")

        # Panel 3: Calibrated Delineation Probability
        im3 = axes[0, 2].imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title(f"3. Delineation Agent: Probability Map\n(Model: {sitrep['arbitration']['recommended_model']})", fontweight="bold", fontsize=11)
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[0, 2].axis("off")

        # Panel 4: Binary Delineated Burned Mask
        axes[1, 0].imshow(bin_mask, cmap="Reds")
        axes[1, 0].set_title(f"4. Delineated Burned Area\n({sitrep['burned_area_km2']} km² / {sitrep['burned_area_acres']} acres)", fontweight="bold", fontsize=11, color="darkred")
        axes[1, 0].axis("off")

        # Panel 5: Copernicus EMS Severity Classification
        if sev_map is not None:
            from matplotlib.colors import ListedColormap
            sev_cmap = ListedColormap(["#f7f7f7", "#fed976", "#fd8d3c", "#bd0026"])
            im5 = axes[1, 1].imshow(sev_map, cmap=sev_cmap, vmin=0, vmax=3)
            axes[1, 1].set_title("5. Severity Agent: Damage Tiers\n(Yellow: Low | Orange: Mod | Red: High)", fontweight="bold", fontsize=11)
        axes[1, 1].axis("off")

        # Panel 6: Tactical Risk & Threat Summary
        axes[1, 2].axis("off")
        threat_color = "#bd0026" if "CRITICAL" in sitrep["threat_level"] else ("#e6550d" if "HIGH" in sitrep["threat_level"] else "#2ca02c")
        axes[1, 2].text(0.05, 0.95, f"TACTICAL SITUATION MATRIX", fontsize=13, fontweight="bold")
        axes[1, 2].text(0.05, 0.83, f"Threat Level: {sitrep['threat_level']}", fontsize=12, fontweight="bold", color=threat_color)
        axes[1, 2].text(0.05, 0.72, f"Debris Flow Hazard: {risk['debris_flow_hazard']['level']}", fontsize=11, fontweight="bold")
        axes[1, 2].text(0.05, 0.62, f"Fire Perimeter: {sitrep['perimeter_km']:.1f} km", fontsize=10)
        axes[1, 2].text(0.05, 0.53, f"Fire Clusters: {sev['num_fire_clusters']}", fontsize=10)
        axes[1, 2].text(0.05, 0.44, f"Containment: {risk['containment_complexity']['level']}", fontsize=10)
        axes[1, 2].text(0.05, 0.32, "Top Emergency Prescriptions:", fontsize=10, fontweight="bold")
        for i, r in enumerate(risk["tactical_recommendations"][:3]):
            axes[1, 2].text(0.05, 0.22 - i*0.09, f"• {r[:45]}...", fontsize=8.5)

        plt.suptitle(f"Autonomous Multi-Agent Wildfire Intelligence Overview — {sitrep['incident_name']}", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively converts NumPy arrays and non-serializable objects for JSON dump."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        elif isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
