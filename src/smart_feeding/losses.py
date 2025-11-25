import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _clip_ce(output_dict, target_dict):
    """
    Clip-level cross-entropy loss wrapper.

    Converts one-hot targets to class indices and applies nn.CrossEntropyLoss.
    Expects `output_dict['clipwise_output']` as logits of shape (B, C)
    and `target_dict['target']` as one-hot (B, C) or indices (B,).
    """
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    logits = output_dict['clipwise_output']
    target = target_dict['target']
    # if numpy -> tensor
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).to(logits.device)
    if target.dim() == 2:
        # one-hot -> indices
        target_idx = torch.argmax(target, dim=1).long()
    else:
        target_idx = target.long()
    return loss_func(logits, target_idx)


def get_loss_func(loss_type):
    """Return a loss function by name. Supported: 'clip_ce', 'clip_bce'."""
    if loss_type == 'clip_ce':
        return _clip_ce
    # Although your losses.py file has clip_bce, train_Fish.py calls 'clip_ce'.
    # We will implement 'clip_ce' here.
    elif loss_type == 'clip_bce':
        # This is Binary Cross Entropy, usually used for multi-label
        # Your task is multi-class, so 'clip_ce' is more appropriate.
        print("Warning: Using BCE for a task that seems to be multi-class.")
        return F.binary_cross_entropy
    else:
        raise ValueError(f"Loại loss '{loss_type}' không được hỗ trợ.")
    
