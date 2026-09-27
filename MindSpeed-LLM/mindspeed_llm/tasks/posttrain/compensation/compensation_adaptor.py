import torch
import torch_npu

from torch.nn.parameter import Parameter
from functools import partial

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu
# from megatron.core.utils import get_model_config
# from megatron.core.enums import ModelType
# from megatron.core.distributed import DistributedDataParallel as DDP
# from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear

# from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
# from mindspeed.ops.gmm import GMMFunction

# from mindspeed_llm.training.training import model_provider_func_wrapper
# from mindspeed_llm.tasks.posttrain.lora.lora_moe import LoraParallelLinearMoE

# from peft import PeftModel

from .utils import(
    get_compression_config,
    get_compressed_weight, 
    decompose_residual, 
    is_enable_compensation, 
    _get_mask_prune_magnitude,
    cast_input_dtype,
    # compute_delta_eora,
    partition_and_load,
)
from mx import finalize_mx_specs, quantize_mx_op
from typing import Union
import copy
import gc

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    
class CompressedLinear(torch.nn.Module):
    """
    Linear layer with pseudo compression (quantization and/or pruning).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_type = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.compute_type = compute_type if compute_type is not None else torch.bfloat16
        self.clinear = None
        self.quant = None
        self.prune = None

    @staticmethod
    def validate_params(quant, prune):
        if quant is not None:
            if not isinstance(quant, str):
                raise ValueError("Quant must be a string like 'nf4' or 'fp4' or 'mxfp4'.")
            quant_option = quant.lower()
            if quant_option not in ["nf4", "fp4", "mxfp4", "mxfp8", "hif4"]:
                raise ValueError("Only 'nf4' and 'fp4' and 'mxfp4' quantization supported currently.")
        if prune is not None:
            if not (isinstance(prune, float) or isinstance(prune, str)):
                raise ValueError("Prune must be a float ratio or a structured pruning string like '2:4'.")
            if isinstance(prune, float):
                if prune < 0.0 or prune >= 1.0:
                    raise ValueError("Prune ratio must be in [0.0, 1.0).")
            elif isinstance(prune, str):
                nm = prune.split(":")
                if len(nm) != 2:
                    raise ValueError("Structured pruning string must be like '2:4'.")
                try:
                    n = int(nm[0])
                    m = int(nm[1])
                except Exception:
                    raise ValueError("Structured pruning string must be like '2:4'.")
                if n <= 0 or m <= 0 or n > m:
                    raise ValueError("Structured pruning string must be like '2:4' with 0 < n <= m.")

    @classmethod
    def prepare_compression(cls, linear, quant, prune, compute_dtype, activation_quant=None):
        # cls.validate_params(quant, prune)
        # nn.Linear有in_features和out_features属性
        # RowParallelLinear和ColumnParallelLinear中为input_size和output_size
        # 为统一，直接从weight中获取
        in_features = linear.weight.size(1)
        out_features = linear.weight.size(0)
        # device = linear.weight.device
        obj = cls(
            in_features,
            out_features,
            linear.bias is not None,
            compute_dtype,
        )
        
        # linear.weight.copy_(W)
        obj.clinear = linear
        obj.quant = quant
        obj.prune = prune
        obj.activation_quant = activation_quant
        for p in obj.parameters():
            p.requires_grad_(False)
        return obj

    @torch.no_grad()
    def _do_compression(self):
        print_rank_0(f"Compressing the base linear layer with quant: {self.quant}, prune: {self.prune}")
        # previous_dtype = self.clinear.weight.dtype
        W = self.clinear.weight.detach().clone().contiguous()
        W = get_compressed_weight(W, quant=self.quant, prune=self.prune)
        
        self.clinear.weight.copy_(W)
        # print_rank_0(f"compressed base_layer weight: {self.clinear.weight}")
        
    def get_decompressed_weight(self):
        return self.clinear.weight.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = cast_input_dtype(x, self.compute_type)
        if self.activation_quant is not None:
            # print_rank_0(f"quantize activation to {self.activation_quant}")
            x = get_compressed_weight(x, quant=self.activation_quant, requires_grad=self.training)
        return self.clinear(x)

class CCLinear(torch.nn.Module):
    """
    Compressed-Compensated Linear:
        y = C(x) + (x @ residual.T) + bias
    
    校准时设置calibrate=True, residual_init="zero"
    校准后训练时设置calibrate=False, residual_init="zero"
    不校准则设置calibrate=False, residual_init="decompose"
    """

    def __init__(
        self,
        linear,
        quant: str = None,
        prune: Union[float, str] = None,
        residual_dtype: torch.dtype = torch.float16,
        compute_type = torch.bfloat16,
        compensate: bool = True,
        compensation_quant: str = None,
        rank: int = 32,
        calibrate: bool = False,    # 在校准阶段设为True，其余为False
        residual_init: str = "decompose",  # "decompose" or "zero"
        use_trainable_residual: bool = False,
        svdquant: bool = False,
        activation_quant: str = None
    ):
        super().__init__()
        
        self.quant = quant
        self.prune = prune
        self.residual_dtype = residual_dtype
        self.device = linear.weight.device
        self.compute_type = compute_type if compute_type is not None else torch.bfloat16

        # nn.Linear有in_features和out_features属性
        # RowParallelLinear和ColumnParallelLinear中为input_size和output_size
        # 为统一，直接从weight中获取
        self.in_features = linear.weight.size(1)
        self.out_features = linear.weight.size(0)
        # self.bias = None

        # 对mixtral的gate，out_features为8，rank=16>8，svd分解的结果会是秩会是更小的那个
        self.rank = min(rank, self.out_features)
        print(f"===========================compensate: {compensate}, compensation_quant: {compensation_quant}")
        self.compensate = compensate
        self.compensation_quant = compensation_quant
        self.svdquant = svdquant
        self.calibrate_mode = calibrate
        self.use_trainable_residual = use_trainable_residual
        self.activation_quant=activation_quant

        self.compressed_linear = CompressedLinear.prepare_compression(linear, quant, prune, compute_type, activation_quant)
        # print_rank_0(f"random init base_layer weight: {self.compressed_linear.get_decompressed_weight()}")
        self.L_residual = None
        self.R_residual = None
        self._build_residual_layers(linear)
        
        # self.L_residual.weight.zero_()
        # self.R_residual.weight.zero_()
        
        self.load_hook_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)
        

    def _build_residual_layers(self, linear):
        if not self.compensate and not self.svdquant:
            print_rank_0(f"L/R will be None")
            return
        args = get_args()
        
        if hasattr(linear, 'config'):
            megatron_config = copy.copy(linear.config)
        else:
            megatron_config = core_transformer_config_from_args(args)
            
        # print_rank_0(f"megatron_config: {megatron_config}")
        # print_rank_0(f"linear_config: {linear.config if hasattr(linear, 'config') else 'N/A'}")
        megatron_config.params_dtype = self.residual_dtype
        # megatron_config.perform_initialization = False

        input_is_parallel = True
        gather_output = False
        if isinstance(linear, RowParallelLinear):
            input_is_parallel = linear.input_is_parallel
        else:
            gather_output = linear.gather_output

        if isinstance(linear, ColumnParallelLinear):
            self.L_residual = torch.nn.Linear(self.in_features, self.rank, bias=False, dtype=self.residual_dtype, device=self.device)
            self.R_residual = ColumnParallelLinear(
                self.rank,
                self.out_features,
                bias=False,
                gather_output=gather_output,
                config=megatron_config,
                init_method=lambda w:w.zero_(),
                skip_bias_add=True,
            ).to(self.device)
        elif isinstance(linear, RowParallelLinear):
            self.L_residual = RowParallelLinear(
                self.in_features, 
                self.rank,
                bias=False,
                input_is_parallel=input_is_parallel,
                config=megatron_config,
                init_method=lambda w:w.zero_(),
                skip_bias_add=True,
            ).to(self.device)
            self.R_residual = torch.nn.Linear(self.rank, self.out_features, bias=False, dtype=self.residual_dtype, device=self.device)
        else:
            self.L_residual = torch.nn.Linear(self.in_features, self.rank, bias=False, dtype=self.residual_dtype, device=self.device)
            self.R_residual = torch.nn.Linear(self.rank, self.out_features, bias=False, dtype=self.residual_dtype, device=self.device)
        
        torch.zero_(self.L_residual.weight)
        torch.zero_(self.R_residual.weight)
        # when residual_weight not equal to zero, raise value error
        if self.L_residual.weight is not None and torch.any(self.L_residual.weight != 0):
            raise ValueError("L_residual weight is not zero after initialization.")
        if self.R_residual.weight is not None and torch.any(self.R_residual.weight != 0):
            raise ValueError("R_residual weight is not zero after initialization.")

        self.L_residual.weight.requires_grad_(self.use_trainable_residual)
        self.R_residual.weight.requires_grad_(self.use_trainable_residual)

    def _set_residual_weights(self, L: torch.Tensor, R: torch.Tensor):
        tensor_parallel_rank = mpu.get_tensor_model_parallel_rank()
        if isinstance(self.L_residual, RowParallelLinear):
            # L需要分块
            partition_and_load(
                self.L_residual.weight,
                L.to(self.residual_dtype),
                1,
                tensor_parallel_rank,
            )
        else:
            self.L_residual.weight.copy_(L.to(self.residual_dtype))
        
        if isinstance(self.R_residual, ColumnParallelLinear):
            # R需要分块
            partition_and_load(
                self.R_residual.weight,
                R.to(self.residual_dtype),
                0,
                tensor_parallel_rank,
            )
        else:
            self.R_residual.weight.copy_(R.to(self.residual_dtype))
            
    def calibrate(self, x: torch.Tensor, name: str = ""):
        pass
        if not self.calibrate_mode:
            return
        current_device = self.compressed_linear.clinear.weight.device
        x = x.to(current_device)
        # if self.original_weight is None:
            # raise ValueError("Original weight is None, cannot calibrate.")
        original_weight = self.compressed_linear.get_decompressed_weight().clone().to(current_device)   # 校准模式下，此时的clinear是未压缩的层
        # print_rank_0(f"before compress: {original_weight}")
        self.compressed_linear._do_compression()  # 对clinear进行压缩
        # print_rank_0(f"after compress: {self.compressed_linear.get_decompressed_weight()}")
        residual = original_weight - self.compressed_linear.get_decompressed_weight()  # [d_out, d_in]
        
        L, R = compute_delta_eora(
            x.to(torch.float32),
            residual.to(torch.float32),
            rank=self.rank,
            method="svd_lowrank",
        )

        # 检查L，R数值
        if torch.isnan(L).any() or torch.isinf(L).any() or torch.isnan(R).any() or torch.isinf(R).any():
            print_rank_0(f"NaN/Inf detected in L/R matrices for module-{name}. Replacing with zeros.")
            L.zero_()
            R.zero_()
        else:
            l_max = torch.abs(L).max()
            r_max = torch.abs(R).max()
            print(f"Max values in module-{name}: L={l_max:.5f}, R={r_max:.5f}")
            print(f"Norm values in module-{name}: L={torch.norm(L):.5f}, R={torch.norm(R):.5f}")

        tensor_parallel_rank = mpu.get_tensor_model_parallel_rank()
        if isinstance(self.L_residual, RowParallelLinear):
            # L需要分块
            partition_and_load(
                self.L_residual.weight,
                L.to(self.residual_dtype),
                1,
                tensor_parallel_rank,
            )
        else:
            self.L_residual.weight.copy_(L.to(self.residual_dtype))
        
        if isinstance(self.R_residual, ColumnParallelLinear):
            # R需要分块
            partition_and_load(
                self.R_residual.weight,
                R.to(self.residual_dtype),
                0,
                tensor_parallel_rank,
            )
        else:
            self.R_residual.weight.copy_(R.to(self.residual_dtype))
        
        self.L_residual.weight.requires_grad_(False)
        self.R_residual.weight.requires_grad_(False)

        del x, residual
        self.original_weight = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _do_compression_and_compensation(self):
        original_weight = self.compressed_linear.get_decompressed_weight().clone().to(self.compute_type).contiguous()
        # print_rank_0(f"pretrained base_layer weight: {original_weight}")
        
        if self.svdquant:
            L,R = decompose_residual(
                original_weight.to(torch.float32),
                method="svd_lowrank",
                rank=self.rank,
            )
            residual = original_weight - (R @ L)
            self.compressed_linear.clinear.weight.copy_(residual)
            # print_rank_0(f"after svdquant base_layer weight: {self.compressed_linear.get_decompressed_weight()}")
            self.compressed_linear._do_compression()
            # print_rank_0(f"compressing base_layer weight: {self.compressed_linear.get_decompressed_weight()}")
            self._set_residual_weights(L, R)
        else:
            self.compressed_linear._do_compression()
            print_rank_0(f"Doing compression for base_layer, quant: {self.quant}, prune: {self.prune}, compensate: {self.compensate}")
            if self.compensate:
                print_rank_0(f"Doing compensation for base layer with rank: {self.rank}")
                compressed_weight = self.compressed_linear.get_decompressed_weight()
                residual = original_weight - compressed_weight
                L, R = decompose_residual(
                    residual.to(torch.float32),
                    method="svd_lowrank",
                    rank=self.rank,
                )
                recon = (R @ L).to(self.residual_dtype)
                print_rank_0(f"no compensation: {torch.norm(residual)/torch.norm(original_weight)}")
                print_rank_0(f"with compensation: {torch.norm(original_weight - (compressed_weight + recon))/torch.norm(original_weight)}")
                self._set_residual_weights(L, R)
    
    @staticmethod
    def _load_state_dict_post_hook(module, incompatible_keys):
        missing_keys, unexpected_keys = incompatible_keys
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            # print_rank_0(f"missing keys: {missing_keys}")
            # print_rank_0(f"unexpected keys: {unexpected_keys}")
            module._do_compression_and_compensation()
        
        if module.compensation_quant is not None:
            module._do_residual_compression()
            
        module.load_hook_handle.remove()
    
    @torch.no_grad()
    def _do_residual_compression(self):
        if self.compensation_quant is None:
            return
        if self.L_residual is None or self.R_residual is None:
            return
        
        print_rank_0(f"Compressing Residual_L/R weights, quant: {self.compensation_quant}")
        W_L = self.L_residual.weight.detach().clone().contiguous()
        W_R = self.R_residual.weight.detach().clone().contiguous()
        
        W_L = get_compressed_weight(W_L, quant=self.compensation_quant, prune=None)
        W_R = get_compressed_weight(W_R, quant=self.compensation_quant, prune=None)
        
        self.L_residual.weight.copy_(W_L)
        self.R_residual.weight.copy_(W_R)
        # print_rank_0(f"after compression L_residual weight: {self.L_residual.weight}")
        # print_rank_0(f"after compression R_residual weight: {self.R_residual.weight}")
        
    def forward(self, x: torch.Tensor):
        # x = cast_input_dtype(x, self.compute_type)
        compressed_outs = self.compressed_linear(x)
        if isinstance(compressed_outs, tuple):
            h, bias_to_return = compressed_outs
        else:
            h = compressed_outs
            bias_to_return = None
        
        if self.L_residual is not None and self.R_residual is not None:
            residual_L_outs = self.L_residual(x)
            if isinstance(residual_L_outs, tuple):
                residual_L_outs = residual_L_outs[0]
            residual_R_outs = self.R_residual(residual_L_outs)
            if isinstance(residual_R_outs, tuple):
                residual_R_outs = residual_R_outs[0]
        else:
            residual_R_outs = torch.zeros_like(h, device=h.device, dtype=h.dtype)

        out = h + residual_R_outs
        return out, bias_to_return

@torch.no_grad()
def apply_compensation_v2(model):
    args = get_args()
    
    # if not is_enable_compensation():
    #     return model
    
    print_rank_0(f"Applying Compression and Compensation to the model ...")
    for layer_id, layer in enumerate(model.decoder.layers):
        for name, module in layer.named_modules():
            last_name = name.rsplit('.', 1)[-1]
            is_target_module = any(last_name == target_module_name for target_module_name in args.compression_target_modules)
            if is_target_module:
                do_compensate = any(last_name == target_module_name for target_module_name in args.compensation_target_modules)
                global_layer_id = int(layer_id + args.num_layers / mpu.get_pipeline_model_parallel_world_size() * mpu.get_pipeline_model_parallel_rank())
                
                q_cfg, p_cfg = get_compression_config(last_name)
                
                if args.high_precision_front_back_attn and global_layer_id in args.high_precision_front_back_attn and last_name in ["linear_qkv", "linear_proj"]:
                    print_rank_0(f"high_precision_front_back_attn enabled for layer-{global_layer_id}: {last_name}")
                    q_cfg = "mxfp8"
                    p_cfg = None
                    
                print_rank_0(f"Layer{global_layer_id}.{name}, will be compressed to quant: {q_cfg}, prune: {p_cfg}, compensation: {do_compensate}")
                
                parent_name, target_name = name.rsplit('.', 1) if '.' in name else ("", name)
                parent_module = layer.get_submodule(parent_name) if parent_name else layer
                # 这里假设传入的model是已经添加了LoRA的模型
                # TODO:若传入的不是peftmodel，该如何进行replace
                target_module = layer.get_submodule(f"{name}.base_layer") if hasattr(module, 'base_layer') else module
                
                new_layer = CCLinear(
                    linear=target_module,
                    quant=q_cfg,
                    prune=p_cfg,
                    residual_dtype=torch.float16 if args.compensation_dtype == "fp16" else torch.bfloat16,
                    compute_type=torch.bfloat16,
                    compensate=do_compensate,
                    compensation_quant=args.compensation_quant,
                    rank=args.compensation_r,
                    calibrate=False,
                    residual_init="zero",
                    use_trainable_residual=args.use_trainable_residual,
                    svdquant=args.use_svdquant,
                    activation_quant=args.activation_quant,
                )
                    
                if hasattr(module, 'base_layer'):
                    setattr(module, "base_layer", new_layer)
                else:
                    setattr(parent_module, target_name, new_layer)

                # print_rank_0(f"LoRA_A shape: {module.lora_A['default'].weight.shape}")
                # print_rank_0(f"LoRA_B shape: {module.lora_B['default'].weight.shape}")
    print_rank_0(f"Applying Compensation to the model done")
