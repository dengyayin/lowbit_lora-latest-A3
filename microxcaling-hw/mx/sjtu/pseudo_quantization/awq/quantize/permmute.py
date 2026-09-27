import torch

def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx

def permutation_zigzag(input:torch.tensor, group_size):
    
    keys = input.mean(dim=0, keepdim=True)
    
    

    sorted_idx = lexsort(keys)
    
    permute_x = input[:, sorted_idx]
    
    return permute_x, sorted_idx