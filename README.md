# Pyronix AI — Multi-Modal Satellite Wildfire Intelligence System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-RTX%204050-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![Status](https://img.shields.io/badge/Status-Complete%20(Stages%201--10)-brightgreen.svg)](STAGE_STATUS.md)

An end-to-end, deep learning and multi-agent system for satellite wildfire detection, all-weather synthetic aperture radar (SAR) penetration, damage severity stratification, and emergency tactical decision support.

---

## 1. System Overview

**Pyronix AI** leverages multi-spectral optical data from **ESA Sentinel-2** and synthetic aperture radar (SAR) data from **ESA Sentinel-1** to detect, delineate, and assess the severity of wildfire burn scars under all environmental conditions.

### Key Capabilities:
- **All-Weather Delineation:** Combines 6-band optical data ($B2, B3, B4, B8, B11, B12$) with dual-polarization C-band SAR ($VV, VH$) to penetrate dense pyrocumulus smoke plumes and cloud cover.
- **Deep ResNet-34 U-Net Backbone:** Symmetrical multi-scale segmentation network (~24.4M parameters) trained using composite BCE + Dice loss.
- **Cross-Continental Generalization:** Proven transferability to unseen California chaparral (May 2021 Pacific Palisades Wildfire, $235.9\text{ km}^2$).
- **Autonomous Multi-Agent Hierarchy:** 5 specialist agents collaborating via a shared context graph:
  1. `SensorArbitrator`: Real-time cloud/smoke masking and dynamic model routing.
  2. `DelineationAgent`: Deep semantic inference with Shannon entropy boundary uncertainty.
  3. `SeverityQuantifier`: Copernicus EMS damage stratification (Grades 1–3) and perimeter morphology.
  4. `RiskAssessment`: Post-fire debris flow hazards, soil hydrophobicity, and tactical BAER prescriptions.
  5. `WildfireOrchestrator`: Executive Situation Report (SitRep) compilation in JSON and Markdown.
- **Interactive Command Application:** Real-time FastAPI web interface with dynamic layer viewing, comparison sliders, tactical chatbot, and SitRep exports.

---

## 2. Project Stage Progression (All Stages Complete)

| Stage | Name | Status | Key Artifacts |
|:---:|:---|:---:|:---|
| **1** | Dataset Inspection & Foundation | **COMPLETE** | `data/inspection/dataset_report.md`, `label_distribution.csv` |
| **2** | Sentinel-2 Optical Baseline | **COMPLETE** | `models/best_s2_baseline_model.pt` (79.35% IoU, 88.49% Dice) |
| **3** | Sentinel-1 SAR Baseline | **COMPLETE** | `models/best_s1_baseline_model.pt` (58.62% IoU, 73.91% Dice) |
| **4** | S2 vs S1 Comparative Study | **COMPLETE** | `results/metrics/s2_vs_s1_comparison.json`, trade-off analysis |
| **5** | Multimodal Optical + SAR Fusion | **COMPLETE** | `models/best_fusion_model.pt` (75.60% IoU, 86.11% Dice) |
| **6** | Fusion Evaluation & Ablation Study | **COMPLETE** | 5 ablation checkpoints, atomic resume verification |
| **7** | External Generalization (Palisades) | **COMPLETE** | $235.9\text{ km}^2$ test benchmark, Optical (96.1% Recall) vs SAR (82.3% Precision) |
| **8** | Multi-Agent Tactical Decision System | **COMPLETE** | `src/agents/`, CEMS Damage Grading, Debris flow hazard modeling |
| **9** | User Application & Tactical Chatbot | **COMPLETE** | FastAPI backend (`app/server.py`), HUD, multi-layer viewer, chatbot |
| **10** | Final Integration & Research Report | **COMPLETE** | `scripts/run_final_validation.py`, `results/reports/final_research_report.md` |

---

## 3. Quantitative Performance Summary

### 3.1 In-Domain Copernicus EMS Test Benchmark (424 Patches)
| Model | Modality | Channels | IoU (%) | Dice (%) | Precision (%) | Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sentinel-2 Baseline** | Optical | 6 | **79.35%** | **88.49%** | 88.35% | 88.63% |
| **Multimodal Fusion** | Optical + SAR | 9 | **75.60%** | **86.11%** | 87.71% | 84.56% |
| **Ablation: S2 RGB Only** | Optical | 3 | 74.19% | 85.18% | 84.58% | 85.79% |
| **Ablation: S2 NIR + SWIR** | Optical | 3 | 73.97% | 85.04% | 86.84% | 83.31% |
| **Ablation: S1 VV + VH** | SAR Radar | 2 | 60.91% | 75.71% | 76.54% | 74.89% |
| **Sentinel-1 Baseline** | SAR Radar | 3 | 58.62% | 73.91% | 75.12% | 72.74% |
| **Ablation: S1 VH Only** | SAR Radar | 1 | 56.88% | 72.51% | 74.52% | 70.60% |
| **Ablation: S1 VV Only** | SAR Radar | 1 | 53.84% | 69.99% | 73.18% | 67.07% |

### 3.2 External Generalization Benchmark: Pacific Palisades Wildfire ($235.9\text{ km}^2$)
| Model | Modality | IoU (%) | Dice (%) | Precision (%) | Recall (%) | Unburned FPR (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S2 RGB Only** | Optical | **54.06%** | **70.18%** | 55.28% | **96.14%** | 12.39% |
| **S1 VV + VH** | SAR Radar | **50.77%** | **67.35%** | **82.32%** | 57.01% | **3.85%** |

---

## 4. Quick Start & Operational Usage

### 4.1 Launch the Web Application
Start the Pyronix AI interactive operations server on `http://127.0.0.1:8000`:

```bash
conda activate satai-train
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any modern web browser to access:
- 1-Click operational presets (Pacific Palisades, Copernicus Encinedo, Heavy Smoke Occlusion)
- GeoTIFF and NumPy drag-and-drop satellite uploads
- Multi-layer visualizer with comparison split-slider and opacity blend controls
- Real-time conversational incident chatbot ("Pyronix Tactical Officer")
- Executive Situation Report (SitRep) export and markdown download

### 4.2 Execute Multi-Agent Incident Analysis via CLI
To run autonomous multi-agent analysis on an operational scene:

```bash
python scripts/run_multiagent_analysis.py
```
Outputs are saved to `results/reports/` (JSON SitRep, Markdown SitRep, and 6-panel overview figures).

### 4.3 Run the Master Validation Suite
To execute the automated end-to-end audit across all 6 test suites:

```bash
python scripts/run_final_validation.py
```

---

## 5. Repository Structure

```
satellite_wildfire_project/
│
├── app/                              <- Web Application & API
│   ├── server.py                     <- FastAPI backend server
│   ├── templates/index.html          <- Modern single-page interface
│   └── static/
│       ├── css/style.css             <- Aerospace tactical CSS design system
│       └── js/app.js                 <- Reactive client-side controller
│
├── src/
│   ├── agents/                       <- Autonomous Multi-Agent Hierarchy
│   │   ├── base_agent.py             <- BaseAgent, message protocols
│   │   ├── sensor_arbitrator.py      <- Cloud/smoke quality arbitrator
│   │   ├── delineation_agent.py      <- Deep learning semantic segmenter
│   │   ├── severity_agent.py         <- Copernicus damage stratifier (EMS Tiers 1-3)
│   │   ├── risk_agent.py             <- Debris flow & BAER planner
│   │   └── orchestrator.py           <- WildfireOrchestrator pipeline
│   ├── models/
│   │   └── unet.py                   <- ResNet-34 U-Net implementation
│   ├── data/
│   │   ├── dataset.py                <- PyTorch dataset loaders
│   │   └── ablation_dataset.py       <- Sliced channel dataset loaders
│   └── training/
│       ├── trainer.py                <- Epoch training loops & loss functions
│       └── stage6_state.py           <- Atomic checkpointing & resume engine
│
├── models/                           <- Checkpoints (.pt)
│   ├── best_s2_baseline_model.pt     <- 6-band optical baseline
│   ├── best_s1_baseline_model.pt     <- 3-band SAR baseline
│   ├── best_fusion_model.pt          <- 9-band multimodal fusion
│   ├── ablation_S1_VV_only.pt        <- Single-pol VV ablation
│   ├── ablation_S1_VH_only.pt        <- Single-pol VH ablation
│   ├── ablation_S1_VV_VH.pt          <- Dual-pol SAR ablation
│   ├── ablation_S2_RGB_only.pt       <- 3-band RGB optical ablation
│   └── ablation_S2_NIR_SWIR.pt       <- NIR/SWIR optical ablation
│
├── results/
│   ├── metrics/                      <- Quantitative scientific metrics & CSVs
│   │   └── final_system_validation.json
│   ├── reports/                      <- Executive SitReps & Research Report
│   │   └── final_research_report.md  <- Complete academic research report
│   └── visualizations/               <- Publication figures & maps
│       ├── ablation/
│       ├── palisades/
│       └── multiagent/
│
├── scripts/
│   ├── run_final_validation.py       <- Master System Validation Suite
│   ├── run_multiagent_analysis.py    <- Standalone multi-agent pipeline
│   ├── evaluate_palisades_generalization.py
│   └── run_ablation_study.py
│
├── STAGE_STATUS.md                   <- Complete stage ledger
└── README.md                         <- Master repository documentation
```

---

## 6. Research Citation

If you use this system or code in your research, please cite:

```bibtex
@article{pyronix_ai_2026,
  title={Autonomous Multi-Modal Satellite Wildfire Intelligence: Deep Semantic Segmentation, Sensor Arbitration, and Multi-Agent Tactical Response},
  author={AI Research \& Engineering Team},
  journal={Advanced Earth Observation AI Reports},
  year={2026},
  url={https://github.com/.../satellite_wildfire_project}
}
```
