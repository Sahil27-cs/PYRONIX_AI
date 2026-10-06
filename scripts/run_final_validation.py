"""
Master System Validation Suite — Satellite Wildfire AI System
Stage 10: Final Integration, Validation & System Health Audit
Author: Lead AI/ML Engineer
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import ResNet34UNet
from src.agents import (
    SensorArbitratorAgent,
    DelineationAgent,
    SeverityQuantifierAgent,
    RiskAssessmentAgent,
    WildfireOrchestrator
)

def run_validation_suite():
    print("=" * 80)
    print("SATELLITE WILDFIRE AI SYSTEM — MASTER VALIDATION SUITE (STAGE 10)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "tests": {},
        "summary": {}
    }

    device = torch.device(results["platform"]["device"])
    print(f"Compute Device: {results['platform']['device_name']} ({device})")

    # -------------------------------------------------------------------------
    # TEST 1: Checkpoint Integrity & Parameter Counts
    # -------------------------------------------------------------------------
    print("\n[TEST 1/6] Auditing Model Checkpoints & Parameter Counts...")
    checkpoint_configs = [
        ("S2_Baseline", "best_s2_baseline_model.pt", 6),
        ("S1_Baseline", "best_s1_baseline_model.pt", 3),
        ("Multimodal_Fusion", "best_fusion_model.pt", 9),
        ("Ablation_S1_VV", "ablation_S1_VV_only.pt", 1),
        ("Ablation_S1_VH", "ablation_S1_VH_only.pt", 1),
        ("Ablation_S1_VV_VH", "ablation_S1_VV_VH.pt", 2),
        ("Ablation_S2_RGB", "ablation_S2_RGB_only.pt", 3),
        ("Ablation_S2_NIR_SWIR", "ablation_S2_NIR_SWIR.pt", 3),
    ]

    models_dir = os.path.join(project_root, "models")
    test1_passed = True
    model_details = {}

    for name, fname, in_ch in checkpoint_configs:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            print(f"  FAILED: Missing checkpoint {fname}")
            test1_passed = False
            model_details[name] = {"status": "MISSING", "path": path}
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model = ResNet34UNet(in_channels=in_ch, num_classes=1, pretrained=False)
            model.load_state_dict(state_dict)
            model.eval()
            total_params = sum(p.numel() for p in model.parameters())

            # Test forward pass with dummy tensor
            dummy = torch.randn(1, in_ch, 256, 256)
            with torch.no_grad():
                out = model(dummy)
                prob = torch.sigmoid(out)

            assert prob.shape == (1, 1, 256, 256), f"Unexpected output shape {prob.shape}"
            assert 0.0 <= prob.min().item() and prob.max().item() <= 1.0, "Probabilities out of bounds"

            model_details[name] = {
                "status": "VALID",
                "file": fname,
                "size_mb": round(size_mb, 2),
                "in_channels": in_ch,
                "parameters": total_params,
                "forward_pass": "PASSED"
            }
            print(f"  PASSED: {name:20s} | {in_ch} ch | {total_params:,} params | {size_mb:.1f} MB")
        except Exception as e:
            print(f"  FAILED: Error loading {fname}: {e}")
            test1_passed = False
            model_details[name] = {"status": "ERROR", "error": str(e)}

    results["tests"]["checkpoints_integrity"] = {
        "passed": test1_passed,
        "models": model_details
    }

    # -------------------------------------------------------------------------
    # TEST 2: Metrics & Benchmark Artifacts Audit
    # -------------------------------------------------------------------------
    print("\n[TEST 2/6] Auditing Stage 1-7 Metrics & Scientific Records...")
    metrics_files = [
        "s2_baseline_metrics.json",
        "s1_baseline_metrics.json",
        "s2_vs_s1_comparison.json",
        "fusion_metrics.json",
        "ablation_study.json",
        "palisades_generalization.json"
    ]
    metrics_dir = os.path.join(project_root, "results", "metrics")
    test2_passed = True
    metrics_summary = {}

    for mf in metrics_files:
        p = os.path.join(metrics_dir, mf)
        if not os.path.exists(p):
            print(f"  FAILED: Missing metric file {mf}")
            test2_passed = False
            continue

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics_summary[mf] = "VERIFIED"
        print(f"  PASSED: {mf} verified ({os.path.getsize(p):,} bytes)")

    # Extract key comparative table
    try:
        with open(os.path.join(metrics_dir, "ablation_study.json")) as f:
            abl = json.load(f)
        with open(os.path.join(metrics_dir, "palisades_generalization.json")) as f:
            pal = json.load(f)

        benchmark_table = {
            "Zenodo_Test_Set": {
                m: {
                    "iou": round(abl[m]["mean_iou"] * 100, 2),
                    "dice": round(abl[m]["mean_dice"] * 100, 2),
                    "precision": round(abl[m]["mean_precision"] * 100, 2),
                    "recall": round(abl[m]["mean_recall"] * 100, 2),
                } for m in abl if isinstance(abl[m], dict) and "mean_iou" in abl[m]
            },
            "Palisades_Generalization_Benchmark": {
                m: {
                    "iou": round(pal[m]["overall_metrics"]["iou"] * 100, 2),
                    "dice": round(pal[m]["overall_metrics"]["dice"] * 100, 2),
                    "precision": round(pal[m]["overall_metrics"]["precision"] * 100, 2),
                    "recall": round(pal[m]["overall_metrics"]["recall"] * 100, 2),
                } for m in pal if isinstance(pal[m], dict) and "overall_metrics" in pal[m]
            }
        }
        results["benchmark_table"] = benchmark_table
        print("  Scientific Benchmark Matrix compiled successfully.")
    except Exception as e:
        print(f"  WARNING: Could not parse benchmark table: {e}")

    results["tests"]["metrics_audit"] = {"passed": test2_passed, "files": metrics_summary}

    # -------------------------------------------------------------------------
    # TEST 3: Multi-Agent Subsystem Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 3/6] Auditing Autonomous Multi-Agent Hierarchy (Stage 8)...")
    test3_passed = True
    try:
        stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
        s1_stats_json = os.path.join(project_root, "data", "inspection", "s1_statistics.json")
        reports_dir = os.path.join(project_root, "results", "reports")

        orchestrator = WildfireOrchestrator(
            models_dir=models_dir,
            stats_json=stats_json,
            s1_stats_json=s1_stats_json,
            device=str(device)
        )

        # Create synthetic incident test
        np.random.seed(42)
        H, W = 256, 256
        synthetic_opt = np.random.uniform(0.05, 0.4, (4, H, W)).astype(np.float32)
        synthetic_sar = np.random.uniform(-25.0, -5.0, (2, H, W)).astype(np.float32)

        ctx = {
            "incident_name": "Stage 10 Master System Validation Sortie",
            "optical_data": synthetic_opt,
            "sar_data": synthetic_sar,
            "threshold": 0.5,
            "metadata": {"test": "Validation Suite"}
        }

        sitrep = orchestrator.run_analysis(ctx, output_dir=reports_dir)

        assert "arbitration" in sitrep, "Missing arbitration in SitRep"
        assert "delineation" in sitrep, "Missing delineation in SitRep"
        assert "severity" in sitrep, "Missing severity in SitRep"
        assert "risk" in sitrep, "Missing risk in SitRep"
        assert "markdown_report" in sitrep, "Missing markdown report"

        print(f"  PASSED: Multi-Agent pipeline executed end-to-end.")
        print(f"    Sensor Mode: {sitrep['arbitration']['mode']}")
        print(f"    Threat Rating: {sitrep['threat_level']}")
        print(f"    Debris Flow Hazard: {sitrep['risk']['debris_flow_hazard']['level']}")
        results["tests"]["multiagent_subsystem"] = {"passed": True, "sitrep_keys": list(sitrep.keys())}
    except Exception as e:
        print(f"  FAILED: Multi-Agent subsystem validation failed: {e}")
        test3_passed = False
        results["tests"]["multiagent_subsystem"] = {"passed": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # TEST 4: Live Web Application Endpoint Health (Stage 9)
    # -------------------------------------------------------------------------
    print("\n[TEST 4/6] Auditing Live Web Application Endpoints...")
    test4_passed = True
    base_url = "http://127.0.0.1:8000"
    api_health = {}

    try:
        # GET /api/status
        req = urllib.request.urlopen(f"{base_url}/api/status", timeout=5)
        st = json.loads(req.read().decode("utf-8"))
        assert st["status"] == "OPERATIONAL", "System status not operational"
        api_health["GET /api/status"] = f"200 OK (Device: {st['device_name']})"

        # GET /api/presets
        req = urllib.request.urlopen(f"{base_url}/api/presets", timeout=5)
        prs = json.loads(req.read().decode("utf-8"))
        assert len(prs) >= 3, "Presets missing"
        api_health["GET /api/presets"] = f"200 OK ({len(prs)} presets)"

        # POST /api/chat
        query_data = urllib.parse.urlencode({"query": "What is the debris flow hazard?"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/chat", data=query_data)
        chat_out = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert "reply" in chat_out and len(chat_out["reply"]) > 10, "Empty chat reply"
        api_health["POST /api/chat"] = "200 OK (Tactical Assistant Functional)"

        # GET /api/sitrep/markdown
        req = urllib.request.urlopen(f"{base_url}/api/sitrep/markdown", timeout=5)
        md_text = req.read().decode("utf-8")
        assert len(md_text) > 100, "SitRep markdown too short"
        api_health["GET /api/sitrep/markdown"] = f"200 OK ({len(md_text):,} bytes)"

        print("  PASSED: All FastAPI endpoints healthy and responding:")
        for k, v in api_health.items():
            print(f"    • {k}: {v}")
    except Exception as e:
        print(f"  FAILED: Web API endpoint health check failed: {e}")
        test4_passed = False
        api_health["error"] = str(e)

    results["tests"]["web_app_endpoints"] = {"passed": test4_passed, "endpoints": api_health}

    # -------------------------------------------------------------------------
    # TEST 5: Data Pipeline & Normalization Robustness
    # -------------------------------------------------------------------------
    print("\n[TEST 5/6] Auditing Radiometric Normalization & Preprocessing...")
    test5_passed = True
    try:
        from src.data.ablation_dataset import AblationPatchDataset
        from src.data.dataset import WildfirePatchDataset

        stats_json = os.path.join(project_root, "data", "inspection", "fusion_statistics.json")
        assert os.path.exists(stats_json), f"Missing {stats_json}"
        with open(stats_json) as f:
            stats = json.load(f)
        assert len(stats["mean"]) == 9, "Expected 9 channels in fusion stats"
        assert len(stats["std"]) == 9, "Expected 9 channels in fusion stats"

        # Check patch test directory
        test_dir = os.path.join(project_root, "data", "processed_fusion", "test")
        if os.path.exists(test_dir):
            ds = AblationPatchDataset(test_dir, channel_indices=[0, 1, 2], stats_json=stats_json)
            assert len(ds) > 0, "Empty ablation test dataset"
            sample_img, sample_mask = ds[0]
            assert sample_img.shape == (3, 256, 256), f"Unexpected shape {sample_img.shape}"
            assert sample_mask.shape == (1, 256, 256), f"Unexpected mask shape {sample_mask.shape}"
            assert not torch.isnan(sample_img).any(), "NaN found in normalized patch"
            print(f"  PASSED: AblationPatchDataset verified ({len(ds)} test patches, shape: {sample_img.shape})")

        print("  PASSED: Radiometric statistics & normalization loaders verified.")
        results["tests"]["data_pipeline"] = {"passed": True}
    except Exception as e:
        print(f"  FAILED: Data pipeline test failed: {e}")
        test5_passed = False
        results["tests"]["data_pipeline"] = {"passed": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # TEST 6: Report Generation & Visual Artifacts
    # -------------------------------------------------------------------------
    print("\n[TEST 6/6] Auditing Publication Artifacts & Visualizations...")
    viz_dirs = [
        "results/visualizations/ablation",
        "results/visualizations/palisades",
        "results/visualizations/multiagent",
        "results/reports"
    ]
    test6_passed = True
    art_count = 0
    for vd in viz_dirs:
        full_p = os.path.join(project_root, vd)
        if os.path.exists(full_p):
            cnt = len([f for f in os.listdir(full_p) if f.endswith(('.png', '.csv', '.json', '.md'))])
            art_count += cnt
            print(f"  PASSED: {vd:35s} contains {cnt} artifacts")
        else:
            print(f"  FAILED: Missing artifact dir {vd}")
            test6_passed = False

    results["tests"]["visual_artifacts"] = {"passed": test6_passed, "total_artifacts": art_count}

    # -------------------------------------------------------------------------
    # SUMMARY & FINAL VERDICT
    # -------------------------------------------------------------------------
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed and test5_passed and test6_passed
    results["summary"] = {
        "overall_status": "PASSED" if all_passed else "FAILED",
        "all_tests_passed": all_passed,
        "total_tests": 6,
        "tests_passed": sum([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed])
    }

    # Save validation report JSON
    out_json = os.path.join(project_root, "results", "metrics", "final_system_validation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"MASTER SYSTEM VALIDATION RESULT: {'PASSED [ALL 6/6 SUITES]' if all_passed else 'FAILED'}")
    print(f"Validation Report saved to: {out_json}")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
