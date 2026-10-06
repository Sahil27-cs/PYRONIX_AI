"""
Stage 8 Multi-Agent System Verification Test Suite
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Verifies:
1. SensorArbitratorAgent routing across clear-sky, cloud-covered, optical-only, and SAR-only contexts.
2. DelineationAgent model loading, tile inference, probability mapping, and binary masking.
3. SeverityQuantifierAgent multi-tier damage classification, perimeter extraction, and morphology metrics.
4. RiskAssessmentAgent cascading hazard assessment and tactical BAER recommendations.
5. WildfireOrchestrator end-to-end execution and executive Situation Report synthesis.
"""

import os
import sys
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents import (
    SensorArbitratorAgent,
    DelineationAgent,
    SeverityQuantifierAgent,
    RiskAssessmentAgent,
    WildfireOrchestrator
)

def run_tests():
    print("=" * 70)
    print("STAGE 8 MULTI-AGENT SYSTEM VERIFICATION TEST SUITE")
    print("=" * 70)

    models_dir = os.path.join(project_root, "models")
    stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
    s1_stats_json = os.path.join(project_root, "data", "inspection", "s1_statistics.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    # -------------------------------------------------------------
    # Test 1: SensorArbitratorAgent - Clear Sky Routing
    # -------------------------------------------------------------
    arbitrator = SensorArbitratorAgent(models_dir=models_dir)
    dummy_opt = np.ones((6, 256, 256), dtype=np.float32) * 0.15
    dummy_sar = np.ones((3, 256, 256), dtype=np.float32) * 0.08
    ctx_clear = {
        "optical_data": dummy_opt,
        "sar_data": dummy_sar,
        "cloud_mask": np.zeros((256, 256), dtype=np.uint8)
    }
    resp1 = arbitrator.process(ctx_clear)
    results["1. Arbitrator Clear-Sky Routing"] = (
        resp1.status == "SUCCESS" and
        resp1.data["mode"] == "MULTIMODAL_FUSION" and
        resp1.data["cloud_cover_pct"] == 0.0
    )

    # -------------------------------------------------------------
    # Test 2: SensorArbitratorAgent - Heavy Cloud Routing to SAR
    # -------------------------------------------------------------
    ctx_cloud = {
        "optical_data": dummy_opt,
        "sar_data": dummy_sar,
        "cloud_mask": np.ones((256, 256), dtype=np.uint8) * 255  # 100% clouds
    }
    resp2 = arbitrator.process(ctx_cloud)
    results["2. Arbitrator Cloud-Occluded Routing to SAR"] = (
        resp2.status == "SUCCESS" and
        resp2.data["mode"] == "SAR_EXCLUSIVE" and
        "s1" in resp2.data["recommended_model"].lower()
    )

    # -------------------------------------------------------------
    # Test 3: SensorArbitratorAgent - Optical Only Routing
    # -------------------------------------------------------------
    ctx_opt_only = {
        "optical_data": dummy_opt[:3],  # 3 RGB bands
        "sar_data": None
    }
    resp3 = arbitrator.process(ctx_opt_only)
    results["3. Arbitrator Optical-Only Fallback"] = (
        resp3.status == "SUCCESS" and
        resp3.data["mode"] == "OPTICAL_ONLY" and
        "rgb" in resp3.data["recommended_model"].lower()
    )

    # -------------------------------------------------------------
    # Test 4: DelineationAgent - Deep Learning Inference
    # -------------------------------------------------------------
    delineator = DelineationAgent(stats_json=stats_json, s1_stats_json=s1_stats_json, device=device)
    ctx_delin = {
        "optical_data": dummy_opt,
        "sar_data": dummy_sar,
        "arbitration_decision": resp1.data,
        "threshold": 0.5
    }
    resp4 = delineator.process(ctx_delin)
    results["4. DelineationAgent Model Inference"] = (
        resp4.status == "SUCCESS" and
        "probability_map" in resp4.data and
        "binary_mask" in resp4.data and
        resp4.data["probability_map"].shape == (256, 256) and
        resp4.data["binary_mask"].shape == (256, 256)
    )

    # -------------------------------------------------------------
    # Test 5: SeverityQuantifierAgent - Multi-Tier Stratification
    # -------------------------------------------------------------
    severity_quantifier = SeverityQuantifierAgent()
    # Create synthetic burn patch in center: radius 40
    syn_mask = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    circle = (x - 128)**2 + (y - 128)**2 <= 40**2
    syn_mask[circle] = 1

    syn_prob = np.zeros((256, 256), dtype=np.float32)
    syn_prob[circle] = 0.85

    # Synthetic dNBR: 0.15 (low) on edges, 0.45 (mod) mid, 0.75 (high) core
    syn_dnbr = np.zeros((256, 256), dtype=np.float32)
    syn_dnbr[(x - 128)**2 + (y - 128)**2 <= 40**2] = 0.20
    syn_dnbr[(x - 128)**2 + (y - 128)**2 <= 25**2] = 0.45
    syn_dnbr[(x - 128)**2 + (y - 128)**2 <= 10**2] = 0.75

    ctx_sev = {
        "delineation_results": {
            "binary_mask": syn_mask,
            "probability_map": syn_prob
        },
        "dnbr_data": syn_dnbr
    }
    resp5 = severity_quantifier.process(ctx_sev)
    results["5. SeverityQuantifierAgent Damage Stratification"] = (
        resp5.status == "SUCCESS" and
        resp5.data["total_burned_area_km2"] > 0.0 and
        resp5.data["fire_perimeter_km"] > 0.0 and
        resp5.data["severity_tiers"]["high_severity"]["km2"] > 0.0 and
        resp5.data["num_fire_clusters"] == 1
    )

    # -------------------------------------------------------------
    # Test 6: RiskAssessmentAgent - Cascading Hazard Analysis
    # -------------------------------------------------------------
    risk_assessor = RiskAssessmentAgent()
    ctx_risk = {
        "severity_results": resp5.data,
        "delineation_results": resp4.data,
        "metadata": {"location": "Southern California Chaparral"}
    }
    resp6 = risk_assessor.process(ctx_risk)
    results["6. RiskAssessmentAgent Hazard Evaluation"] = (
        resp6.status == "SUCCESS" and
        "overall_threat_level" in resp6.data and
        "debris_flow_hazard" in resp6.data and
        len(resp6.data["tactical_recommendations"]) >= 3
    )

    # -------------------------------------------------------------
    # Test 7: WildfireOrchestrator - End-to-End Pipeline
    # -------------------------------------------------------------
    orchestrator = WildfireOrchestrator(
        models_dir=models_dir,
        stats_json=stats_json,
        s1_stats_json=s1_stats_json,
        device=device
    )
    full_ctx = {
        "incident_name": "Test Simulation Wildfire",
        "optical_data": dummy_opt,
        "sar_data": dummy_sar,
        "dnbr_data": syn_dnbr,
        "cloud_mask": np.zeros((256, 256), dtype=np.uint8)
    }
    sitrep = orchestrator.run_analysis(full_ctx, output_dir=None)
    results["7. Orchestrator End-to-End Execution"] = (
        sitrep is not None and
        "threat_level" in sitrep and
        "burned_area_km2" in sitrep and
        "markdown_report" in sitrep and
        len(sitrep["markdown_report"]) > 100
    )

    print("\nVERIFICATION RESULTS:")
    print("-" * 70)
    all_passed = True
    for test_name, passed in results.items():
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str}  {test_name}")
        if not passed:
            all_passed = False
    print("-" * 70)

    if all_passed:
        print("ALL 7 MULTI-AGENT VERIFICATION CHECKS PASSED PERFECTLY!\n")
    else:
        print("SOME VERIFICATION CHECKS FAILED! Check outputs.\n")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
