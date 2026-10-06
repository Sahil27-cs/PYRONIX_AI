"""
Stage 6 Pipeline State Management & Checkpoint Tracking
Author: Lead AI/ML Engineer
System: Satellite Wildfire AI System

Provides robust atomic persistence for:
1. Pipeline state (results/metrics/stage6_state.json)
2. Evaluation state and cache (results/metrics/stage6_evaluation_state.json, stage6_evaluation_cache.json)
3. Feature sensitivity state and cache (results/metrics/stage6_sensitivity_state.json, stage6_sensitivity_cache.json)
4. Checkpoint inspection and safety verification
"""

import os
import json
import torch

def atomic_save_json(data, filepath):
    """Atomically writes JSON to disk using a temporary file and replace."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)

def load_json(filepath, default=None):
    """Safely loads JSON from disk, returning default if file does not exist or is invalid."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}

def atomic_save_torch(data, filepath):
    """Atomically saves PyTorch checkpoint to disk using a temporary file and replace."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, filepath)

class Stage6StateManager:
    """
    Coordinates and persists the complete execution state of Stage 6,
    guaranteeing resumability across model training, evaluation, sensitivity testing,
    statistical analysis, and visualization.
    """
    MODELS = [
        "S1_VV_only",
        "S1_VH_only",
        "S1_VV_VH",
        "S2_RGB_only",
        "S2_NIR_SWIR"
    ]

    ALL_EVAL_MODELS = [
        "S1_VV_only",
        "S1_VH_only",
        "S1_VV_VH",
        "S1_Full",
        "S2_RGB_only",
        "S2_NIR_SWIR",
        "S2_Full",
        "Multimodal_Fusion"
    ]

    SENSITIVITY_EXPERIMENTS = [
        "Fusion_Baseline_All9",
        "Mask_S1_SAR_All",
        "Mask_S2_Optical_All",
        "Mask_VV_Only",
        "Mask_VH_Only",
        "Mask_Ratio_Only",
        "Mask_RGB_Only",
        "Mask_NIR_Only",
        "Mask_SWIR_Only"
    ]

    def __init__(self, metrics_dir, checkpoints_dir, models_dir):
        self.metrics_dir = metrics_dir
        self.checkpoints_dir = checkpoints_dir
        self.models_dir = models_dir

        self.state_file = os.path.join(metrics_dir, "stage6_state.json")
        self.eval_state_file = os.path.join(metrics_dir, "stage6_evaluation_state.json")
        self.eval_cache_file = os.path.join(metrics_dir, "stage6_evaluation_cache.json")
        self.sens_state_file = os.path.join(metrics_dir, "stage6_sensitivity_state.json")
        self.sens_cache_file = os.path.join(metrics_dir, "stage6_sensitivity_cache.json")

        self._init_states()

    def _init_states(self):
        """Initializes state files with default structure if not present, and syncs with disk."""
        # 1. Main Stage 6 Pipeline State
        self.pipeline_state = load_json(self.state_file, default={})
        for m in self.MODELS:
            if m not in self.pipeline_state or not isinstance(self.pipeline_state[m], dict):
                self.pipeline_state[m] = {
                    "status": "pending",
                    "last_completed_epoch": 0
                }
        for step in ["evaluation", "feature_sensitivity", "statistics", "visualizations"]:
            if step not in self.pipeline_state:
                self.pipeline_state[step] = "pending"
        if "overall_status" not in self.pipeline_state:
            self.pipeline_state["overall_status"] = "pending"

        self._sync_with_checkpoints()
        self.save_pipeline_state()

        # 2. Evaluation State
        self.eval_state = load_json(self.eval_state_file, default={})
        for m in self.ALL_EVAL_MODELS:
            if m not in self.eval_state:
                self.eval_state[m] = "pending"
        self.save_eval_state()

        # 3. Sensitivity State
        self.sens_state = load_json(self.sens_state_file, default={})
        for exp in self.SENSITIVITY_EXPERIMENTS:
            if exp not in self.sens_state:
                self.sens_state[exp] = "pending"
        self.save_sens_state()

    def _sync_with_checkpoints(self):
        """Syncs the pipeline state with physical checkpoints on disk."""
        for m in self.MODELS:
            latest_path = os.path.join(self.checkpoints_dir, f"{m}_latest.pt")
            best_path = os.path.join(self.models_dir, f"ablation_{m}.pt")
            if os.path.exists(latest_path):
                try:
                    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
                    current_epoch = ckpt.get("current_epoch", 0)
                    total_epochs = ckpt.get("total_epochs", 12)
                    is_complete = bool(ckpt.get("training_complete", False)) and (current_epoch >= total_epochs)
                    self.pipeline_state[m]["last_completed_epoch"] = current_epoch
                    self.pipeline_state[m]["status"] = "complete" if is_complete else "running"
                except Exception:
                    pass
            elif os.path.exists(best_path):
                try:
                    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                    is_complete = bool(ckpt.get("training_complete", False))
                    total_epochs = ckpt.get("total_epochs", 12)
                    epoch_saved = ckpt.get("epoch_saved", 0)
                    if is_complete and epoch_saved >= total_epochs:
                        self.pipeline_state[m]["status"] = "complete"
                        self.pipeline_state[m]["last_completed_epoch"] = epoch_saved
                except Exception:
                    pass

    def save_pipeline_state(self):
        atomic_save_json(self.pipeline_state, self.state_file)

    def save_eval_state(self):
        atomic_save_json(self.eval_state, self.eval_state_file)

    def save_sens_state(self):
        atomic_save_json(self.sens_state, self.sens_state_file)

    def update_model_state(self, model_name, status, last_completed_epoch):
        self.pipeline_state[model_name] = {
            "status": status,
            "last_completed_epoch": last_completed_epoch
        }
        self.save_pipeline_state()

    def update_step_status(self, step_name, status):
        self.pipeline_state[step_name] = status
        self.save_pipeline_state()

    def set_overall_status(self, status):
        self.pipeline_state["overall_status"] = status
        self.save_pipeline_state()

    # Evaluation caching
    def get_eval_status(self, model_name):
        return self.eval_state.get(model_name, "pending")

    def record_model_evaluation(self, model_name, eval_result, patch_results):
        cache = load_json(self.eval_cache_file, default={"model_eval_results": {}, "patch_level_dict": {}})
        if "model_eval_results" not in cache:
            cache["model_eval_results"] = {}
        if "patch_level_dict" not in cache:
            cache["patch_level_dict"] = {}
        cache["model_eval_results"][model_name] = eval_result
        cache["patch_level_dict"][model_name] = patch_results
        atomic_save_json(cache, self.eval_cache_file)
        self.eval_state[model_name] = "complete"
        self.save_eval_state()

    def load_cached_evaluations(self):
        cache = load_json(self.eval_cache_file, default={"model_eval_results": {}, "patch_level_dict": {}})
        return cache.get("model_eval_results", {}), cache.get("patch_level_dict", {})

    # Sensitivity caching
    def get_sens_status(self, exp_name):
        return self.sens_state.get(exp_name, "pending")

    def record_sensitivity_result(self, exp_name, result):
        cache = load_json(self.sens_cache_file, default={"sensitivity_results": {}})
        if "sensitivity_results" not in cache:
            cache["sensitivity_results"] = {}
        cache["sensitivity_results"][exp_name] = result
        atomic_save_json(cache, self.sens_cache_file)
        self.sens_state[exp_name] = "complete"
        self.save_sens_state()

    def load_cached_sensitivity(self):
        cache = load_json(self.sens_cache_file, default={"sensitivity_results": {}})
        return cache.get("sensitivity_results", {})

    def inspect_checkpoints(self):
        """
        Inspects all ablation checkpoints and reports their resume status.
        """
        results = {}
        for m in self.MODELS:
            latest_path = os.path.join(self.checkpoints_dir, f"{m}_latest.pt")
            best_path = os.path.join(self.models_dir, f"ablation_{m}.pt")
            has_latest = os.path.exists(latest_path)
            has_best = os.path.exists(best_path)

            if has_latest:
                try:
                    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
                    current_epoch = ckpt.get("current_epoch", 0)
                    total_epochs = ckpt.get("total_epochs", 12)
                    is_complete = bool(ckpt.get("training_complete", False)) and (current_epoch >= total_epochs)
                    results[m] = {
                        "last_epoch": current_epoch,
                        "total_epochs": total_epochs,
                        "complete": is_complete,
                        "can_resume": not is_complete,
                        "next_epoch": current_epoch + 1 if not is_complete else None,
                        "checkpoint_type": "latest_epoch_state",
                        "checkpoint_file": latest_path
                    }
                except Exception as e:
                    results[m] = {
                        "last_epoch": "CORRUPT",
                        "total_epochs": 12,
                        "complete": False,
                        "can_resume": False,
                        "next_epoch": None,
                        "checkpoint_type": "corrupt_latest",
                        "checkpoint_file": latest_path,
                        "error": str(e)
                    }
            elif has_best:
                try:
                    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                    is_complete = bool(ckpt.get("training_complete", False))
                    current_epoch = ckpt.get("epoch_saved", None)
                    total_epochs = ckpt.get("total_epochs", 12)
                    # Legacy files lack epoch, optimizer, scheduler, scaler
                    has_opt = "optimizer_state_dict" in ckpt
                    if is_complete and current_epoch is not None and current_epoch >= total_epochs:
                        results[m] = {
                            "last_epoch": current_epoch,
                            "total_epochs": total_epochs,
                            "complete": True,
                            "can_resume": False,
                            "next_epoch": None,
                            "checkpoint_type": "completed_best",
                            "checkpoint_file": best_path
                        }
                    else:
                        results[m] = {
                            "last_epoch": f"Unknown ({current_epoch if current_epoch is not None else 'legacy best-so-far'})",
                            "total_epochs": total_epochs,
                            "complete": False,
                            "can_resume": has_opt,
                            "next_epoch": (current_epoch + 1) if (has_opt and current_epoch) else 1,
                            "checkpoint_type": "legacy_best_weights_only",
                            "checkpoint_file": best_path,
                            "notice": "Missing optimizer/scheduler/scaler state. Starting from Epoch 1 while retaining best score."
                        }
                except Exception as e:
                    results[m] = {
                        "last_epoch": "CORRUPT",
                        "total_epochs": 12,
                        "complete": False,
                        "can_resume": False,
                        "next_epoch": None,
                        "checkpoint_type": "corrupt_best",
                        "checkpoint_file": best_path,
                        "error": str(e)
                    }
            else:
                results[m] = {
                    "last_epoch": 0,
                    "total_epochs": 12,
                    "complete": False,
                    "can_resume": True,
                    "next_epoch": 1,
                    "checkpoint_type": "none",
                    "checkpoint_file": None
                }

        return results
