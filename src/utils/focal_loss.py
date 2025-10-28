import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=1)
        log_prob = log_softmax.gather(1, targets.view(-1, 1)).squeeze()
        prob = torch.exp(log_prob)
        cross_entropy = -log_prob
        p_t = prob
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * cross_entropy

        if self.alpha is not None:
            # Assumes alpha is a 1D tensor with weights per class
            # Ensure alpha is on the same device as targets
            if isinstance(self.alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32).to(targets.device)
            elif isinstance(self.alpha, torch.Tensor):
                self.alpha = self.alpha.to(targets.device)
            else:
                 raise TypeError("alpha must be a list, numpy array, or torch.Tensor")
                 
            alpha_t = self.alpha.gather(0, targets.view(-1))
            loss = alpha_t * loss

        if self.reduction == 'mean':
            # Check if loss tensor is empty before calling mean()
            if loss.numel() == 0:
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True) # Return zero loss if batch/targets were empty
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # reduction == 'none'
            return loss

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Ensure FOCAL_LOSS_GAMMA exists in config, provide default if necessary
        gamma = config.get('TRAIN', {}).get('FOCAL_LOSS_GAMMA', 2.0)
        self.focal_loss = FocalLoss(gamma=gamma)
        # We might not actually use label_smoothing here if MixUp/CutMix are disabled
        # self.label_smoothing = config.get('AUGMENTATION', {}).get('LABEL_SMOOTHING', 0.0)
        # self.num_classes = config.get('NUM_CLASSES', None)

    def forward(self, inputs, targets):
        # Handle potential invalid targets (-1) from dataset error handling
        valid_indices = targets != -1
        if not valid_indices.all():
            # Filter out invalid targets and corresponding inputs
            inputs = inputs[valid_indices]
            targets = targets[valid_indices]
            # If the batch becomes empty after filtering, return zero loss
            if inputs.nelement() == 0:
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return self.focal_loss(inputs, targets)