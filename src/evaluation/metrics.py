"""
Global Segmentation Metrics Computation for Binary Burned-Area Evaluation
Computes IoU, Dice, Precision, Recall, F1, Accuracy, TP, TN, FP, FN globally.
"""

import torch
import numpy as np

class MetricTracker:
    """
    Accumulates true positives, false positives, true negatives, and false negatives
    across all validation/test batches to compute exact global corpus metrics.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total_pixels = 0

    def update(self, logits, targets):
        # logits: (B, 1, H, W)
        # targets: (B, 1, H, W) binary {0, 1}
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).long()
            targs = targets.long()

            tp = ((preds == 1) & (targs == 1)).sum().item()
            fp = ((preds == 1) & (targs == 0)).sum().item()
            tn = ((preds == 0) & (targs == 0)).sum().item()
            fn = ((preds == 0) & (targs == 1)).sum().item()

            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
            self.total_pixels += (tp + fp + tn + fn)

    def compute(self, eps=1e-7):
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        iou = tp / (tp + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = dice
        accuracy = (tp + tn) / (self.total_pixels + eps)

        gt_burned_pct = ((tp + fn) / (self.total_pixels + eps)) * 100.0
        pred_burned_pct = ((tp + fp) / (self.total_pixels + eps)) * 100.0

        return {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_pixels": int(self.total_pixels),
            "gt_burned_percentage": float(gt_burned_pct),
            "pred_burned_percentage": float(pred_burned_pct)
        }
