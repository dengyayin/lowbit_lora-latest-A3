import torch
from megatron.training import get_args

def grouped_lr_lora_plus(name, param):
    if not param.requires_grad:
        return False
    
    if 'lora_B' in name:
        return True
    
    else:
        return False

def grouped_lr_trainable_residual(name, param):
    if not param.requires_grad:
        return False
    
    if 'L_residual' in name or 'R_residual' in name:
        return True
    
    else:
        return False