import torch
import torch.nn as nn
import tqdm
import gc
import functools
import copy
from collections import defaultdict

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from ..models.llama_mx import LlamaForCausalLM_giant
from .qmodule_mxfp import MXFP_Linear

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if (isinstance(m, nn.Linear) or isinstance(m, MXFP_Linear))}


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, LlamaForCausalLM_giant):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers


