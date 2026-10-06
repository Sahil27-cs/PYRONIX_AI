"""
Loss Functions for Binary Semantic Segmentation
Implements 0.5 * Focal Loss + 0.5 * Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with sigmoid activation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class BinaryDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Dice = (2 * intersection + smooth) / (total + smooth)
    Loss = 1 - Dice
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        # Flatten spatial dimensions per batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        total = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return (1.0 - dice).mean()

class CombinedLoss(nn.Module):
    """
    Composite Loss: 0.5 * Focal Loss + 0.5 * Dice Loss
    """
    def __init__(self, alpha=0.75, gamma=2.0, smooth=1.0, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        focal_l = self.focal(logits, targets)
        dice_l = self.dice(logits, targets)
        return self.focal_weight * focal_l + self.dice_weight * dice_l, focal_l, dice_l
