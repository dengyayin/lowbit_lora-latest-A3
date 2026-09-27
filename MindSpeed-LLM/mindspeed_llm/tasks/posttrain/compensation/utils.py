import torch
import torch_npu
from collections import defaultdict

from megatron.training import get_args, print_rank_0
from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear
from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
from mx import finalize_mx_specs, quantize_mx_op
from mx.fp4_ops import quantize_hif4_kernel, quantize_hif4_kernel_ste

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

def is_enable_compensation():
    args = get_args()
    enable_compensation = (hasattr(args, 'use_compensation') and args.use_compensation) or (hasattr(args, 'use_svdquant') and args.use_svdquant)
    if enable_compensation and args.compression_target_modules:
        return True
    # if hasattr(args, 'use_compensation'):
    #     return True
    return False

@torch.no_grad()
def cast_input_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype:
        return x
    else:
        return x.to(dtype)

@torch.no_grad()
def find_all_linear_names(model):
    linear_cls = (torch.nn.Linear, RowParallelLinear, ColumnParallelLinear)
    
    compensate_module_names = list()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            if "lm_head" in name or "lora_A" in name or "lora_B" in name:
                continue
            compensate_module_names.append(name)

    return compensate_module_names

@torch.no_grad()
def _get_mask_prune_magnitude(
    W,
    sparsity_ratio: float,
    prune_n: int,
    prune_m: int,
    largest: bool,
    offload: bool,
) -> torch.tensor:
    W_metric = torch.abs(W)
    if prune_n != 0:
        W_mask = torch.zeros_like(W, dtype=torch.bool)
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii : (ii + prune_m)].float()
                idx_to_keep = torch.topk(tmp, prune_n, dim=1, largest=largest)[1]
                W_mask.scatter_(1, ii + idx_to_keep, True)
    else:
        if largest:
            thresh = torch.sort(W_metric.flatten(), descending=True)[0][int(W.numel() * sparsity_ratio)].cpu()
            W_mask = W_metric >= thresh
        else:
            thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * sparsity_ratio)]
            W_mask = W_metric < thresh
    if offload:
        return W_mask.cpu()
    else:
        return W_mask

@torch.no_grad()
def decompose_residual(residual, method="svd", rank=32):
    # residual: [out_features, in_features]
    if method == "svd":
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, S.size(0))
    elif method == "svd_lowrank":
        r = rank + 8
        # print_rank_0(f"Perform svd_lowrank start, with r={r}")
        U, S, V = torch.svd_lowrank(residual, q=r, niter=10)
        # print_rank_0(f"Perform svd_lowrank done, U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = V[:, :rank].T
        # print_rank_0(f"Perform svd_lowrank done, U shape: {U_r.shape}, S shape: {S_r.shape}, V shape: {Vh_r.shape}")

        sqrt_S_r = torch.sqrt(S_r)
        Sigma_r_half = torch.diag(sqrt_S_r)

        # print_rank_0(f"Perform R compute")
        R = U_r @ Sigma_r_half      # [out_features, rank]
        # print_rank_0(f"R compute done, shape: {R.shape}")
        
        # print_rank_0(f"Perform L compute")
        L = Sigma_r_half @ Vh_r     # [rank, in_features]
        # print_rank_0(f"compute done, shape, L: {L.shape}, R: {R.shape}")
        
        # TODO: cast L/R dtype according to args
    return L, R


def partition_and_load(weight, tensor_to_load, partition_dim, parallel_rank):
    partition_size = weight.shape[partition_dim]
    start_index = parallel_rank * partition_size
    end_index = start_index + partition_size
    
    # 根据分区维度选择正确的切片方式
    if partition_dim == 0:
        partition = tensor_to_load[start_index:end_index, :]
    elif partition_dim == 1:
        partition = tensor_to_load[:, start_index:end_index]
    else:
        raise ValueError(f"Invalid partition_dim: {partition_dim}")

    with torch.no_grad():
        # self.weight = Parameter(
        #     torch.empty(
        #         self.output_size_per_partition,
        #         self.input_size,
        #         device=torch.cuda.current_device(),
        #         dtype=config.params_dtype,
        #     )
        # )
        # print_rank_0(f"weight shape: {weight.shape}, partition shape: {partition.shape}")
        weight.copy_(partition)

def validate_params(args):
    if args.quant_method is not None:
        if not isinstance(args.quant_method, str):
            raise ValueError("Quant must be a string like 'nf4' or 'fp4' or 'mxfp4' or 'mxfp8'.")
        quant_option = args.quant_method.lower()
        if quant_option not in ["nf4", "fp4", "mxfp4", "mxfp8"]:
            raise ValueError("Only 'nf4' and 'fp4' and 'mxfp4' and 'mxfp8' quantization supported currently.")
    if args.prune_method is not None:
        if not (isinstance(args.prune_method, float) or isinstance(args.prune_method, str)):
            raise ValueError("Prune must be a float ratio or a structured pruning string like '2:4'.")
        if isinstance(args.prune_method, float):
            if args.prune_method < 0.0 or args.prune_method >= 1.0:
                raise ValueError("Prune ratio must be in [0.0, 1.0).")
        elif isinstance(args.prune_method, str):
            nm = args.prune_method.split(":")
            if len(nm) != 2:
                raise ValueError("Structured pruning string must be like '2:4'.")
            try:
                n = int(nm[0])
                m = int(nm[1])
            except Exception:
                raise ValueError("Structured pruning string must be like '2:4'.")
            if n <= 0 or m <= 0 or n > m:
                raise ValueError("Structured pruning string must be like '2:4' with 0 < n <= m.")
    if is_enable_qlora():
        raise ValueError("--use-compensation is not compatible with --qlora")


def get_compressed_weight(weight, quant=None, prune=None, requires_grad=False):
    if quant is None and prune is None:
        return weight

    # validate_params(args)
    previous_dtype = weight.dtype
    if requires_grad:
        # 激活量化路径：保留计算图以支持 STE 直通梯度
        W = weight.contiguous().to(torch.float32)
    else:
        # 权重量化路径：无需梯度，detach 避免计算图开销
        W = weight.detach().clone().contiguous().to(torch.float32)
    # W = weight.detach().clone().contiguous()

    if prune:
        if isinstance(prune, float):
            prune_mask = _get_mask_prune_magnitude(W, prune, 0, 0, False, True)
        else:
            nm = prune.split(":")
            prune_mask = _get_mask_prune_magnitude(W, 0, int(nm[0]), int(nm[1]), False, True)
        prune_mask = prune_mask.to(W.device)
        W[prune_mask] = 0
        # print_rank_0(f"W sparsity ratio: {torch.sum(W == 0).item() / W.numel()}")

    if quant:
        quant_option = quant.lower()
        if quant_option == "nf4":
            W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="nf4")
            W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="nf4")
            # del W_quant, W_quant_state
        elif quant_option == "fp4":
            W_quant, W_quant_state = bnb.functional.quantize_4bit(W, quant_type="fp4")
            W = bnb.functional.dequantize_4bit(W_quant, W_quant_state, quant_type="fp4")
            # del W_quant, W_quant_state
        elif quant_option == "mxfp4":
            mx_specs = {
                "scale_bits": 8,
                "w_elem_format": "fp4_e2m1",
                "bfloat": 16,
                "block_size": 32,
                "round": "even",
                "custom_cuda": False,
                # for QAT, use FP32 for backprop
                "quantize_backprop": True,
            }
            mx_specs = finalize_mx_specs(mx_specs)

            # print_rank_0(f"Apply quant on {W.device}, W shape: {W.shape}")
            W = quantize_mx_op(
                W,
                mx_specs,
                elem_format=mx_specs['w_elem_format'],
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
        elif quant_option == "mxfp8":
            mx_specs = {
                "scale_bits": 8,
                "w_elem_format": "fp8_e4m3",
                "bfloat": 16,
                "block_size": 32,
                "round": "even",
                "custom_cuda": False,
                # for QAT, use FP32 for backprop
                "quantize_backprop": True,
            }
            mx_specs = finalize_mx_specs(mx_specs)

            W = quantize_mx_op(
                W,
                mx_specs,
                elem_format=mx_specs['w_elem_format'],
                axes=[-1],
                round=mx_specs["round_mx_output"],
            )
        elif quant_option == "hif4":
            if requires_grad:
                W = quantize_hif4_kernel_ste(W, N=4)
            else:
                W = quantize_hif4_kernel(W, N=4)
        else:
            raise ValueError("Only 'nf4' and 'fp4' and 'mxfp4' and 'mxfp8' quantization supported currently.")

    return W.to(previous_dtype)
    # return W


def get_compression_config(last_name):
    args = get_args()
    is_high_bit = any(last_name == high_bit_module_name for high_bit_module_name in args.high_bit_target_modules)
    is_low_bit = any(last_name == low_bit_module_name for low_bit_module_name in args.low_bit_target_modules)
    is_high_sparsity = any(last_name == high_sparsity_module_name for high_sparsity_module_name in args.high_sparsity_target_modules)
    is_low_sparsity = any(last_name == low_sparsity_module_name for low_sparsity_module_name in args.low_sparsity_target_modules)
    
    q_cfg = None
    p_cfg = None
    if is_high_bit:
        q_cfg = args.high_bit_quant_method
    elif is_low_bit:
        q_cfg = args.low_bit_quant_method
    else:
        q_cfg = args.quant_method
    
    if is_high_sparsity:
        p_cfg = args.high_sparsity_prune_method
    elif is_low_sparsity:
        p_cfg = args.low_sparsity_prune_method
    else:
        p_cfg = args.prune_method
    return q_cfg, p_cfg


@torch.no_grad()
def _calc_compressed_bytes(shape, quant_method, elem_size=2):
    """Calculate the theoretical compressed storage size for a given quantization method.

    Args:
        shape: tensor shape (tuple), e.g. (M, C) for a weight
        quant_method: str or None, e.g. 'hif4', 'mxfp4', 'mxfp8', 'nf4', 'fp4', None
        elem_size: bytes per element in original dtype (2 for bf16)

    Returns:
        compressed_bytes (int), ratio description
    """
    M, C = shape[-2], shape[-1]

    if quant_method is None:
        # No quantization: stored as original dtype
        return M * C * elem_size, "no compression"

    q = quant_method.lower()

    if q == "hif4":
        # HiF4: data=uint8(M, C/2) + scale=fp32(M, ceil(C/64))
        data_bytes = M * (C // 2) * 1
        n_blocks = (C + 64 - 1) // 64
        scale_bytes = M * n_blocks * 4
        total = data_bytes + scale_bytes
        ratio_str = f"hif4 data(M,{C}//2)+scale(M,ceil(C/64))"

    elif q == "mxfp4":
        # mxfp4: elem=4bit, block=32, scale=8bit
        #   data: M*C*0.5 bytes (4-bit per elem)
        #   scale: M*ceil(C/32)*1 bytes (8-bit per block)
        data_bytes = int(M * C * 0.5)
        n_blocks = (C + 32 - 1) // 32
        scale_bytes = M * n_blocks * 1  # 8-bit scale
        total = data_bytes + scale_bytes
        ratio_str = "mxfp4(MX block=32, elem=4bit, scale=8bit)"

    elif q == "mxfp8":
        # mxfp8: elem=8bit, block=32, scale=8bit
        data_bytes = M * C * 1
        n_blocks = (C + 32 - 1) // 32
        scale_bytes = M * n_blocks * 1
        total = data_bytes + scale_bytes
        ratio_str = "mxfp8(MX block=32, elem=8bit, scale=8bit)"

    elif q in ("nf4", "fp4"):
        # bitsandbytes 4-bit: stored as uint8 with 2 elems/byte (like hif4)
        # quant_state (absmax etc.) is negligible compared to weight size
        data_bytes = M * (C // 2) * 1
        # bnb uses 64-element blocks internally for absmax at fp16
        n_blocks = (C + 64 - 1) // 64
        scale_bytes = M * n_blocks * 2  # fp16 absmax per block
        total = data_bytes + scale_bytes
        ratio_str = f"{q}(4-bit, block=64, fp16 scale)"

    else:
        # Unknown quant method: assume no compression (conservative)
        total = M * C * elem_size
        ratio_str = f"unknown({q}), treated as no compression"

    return total, ratio_str


@torch.no_grad()
def measure_total_params(model, print_detail=True):
    """Calculate the total parameter storage bytes for the model, all PP ranks combined.

    Each rank computes its own parameters and categories, then all_reduce across
    pipeline-parallel ranks to get the complete model total.

    Category explanation:
      - **CCLinear base weights**: counted as THEORETICAL compressed storage
        (hif4 → data+scale, mxfp4 → 4-bit+scale, etc.), not raw bf16.
      - **SALR residuals** (L_residual / R_residual): counted as bf16 (extra overhead).
      - **LoRA A/B**: counted as bf16 (or compressed if --lora-quant is set).
      - **All other parameters** (norm, embedding, lm_head, etc.): counted as
        their actual dtype size.

    Args:
        model: A single model chunk (e.g. model[0] from setup_model_and_optimizer).
               Each PP rank calls this with its own chunk; results are aggregated
               via all_reduce internally.
        print_detail: whether to print the aggregated report (rank 0 only).

    Returns:
        dict with:
          - total_bytes: total parameter storage bytes across ALL PP ranks
          - total_params_count: total number of elements across ALL PP ranks
    """
    from mindspeed_llm.tasks.posttrain.compensation.compensation_adaptor import CCLinear

    args = get_args()

    # ---- Local computation: this rank's own parameters ----
    per_param = []
    total_bytes = 0
    total_params_count = 0

    # Collect CCLinear module names for this rank's model chunk
    cc_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, CCLinear):
            cc_modules[name] = module

    # Classify each parameter
    for param_name, param in model.named_parameters():
        numel = param.numel()
        elem_size = param.element_size()
        total_params_count += numel

        matched_cc_name = None
        for cc_name in cc_modules:
            if param_name.startswith(cc_name):
                matched_cc_name = cc_name
                break

        if matched_cc_name is None:
            if "lora_A" in param_name or "lora_B" in param_name:
                if args.lora_quant is not None:
                    param_bytes, _ = _calc_compressed_bytes(param.shape, args.lora_quant, elem_size)
                    category = f"LoRA A/B ({args.lora_quant} quant)"
                else:
                    param_bytes = numel * elem_size
                    category = "LoRA A/B"
            else:
                param_bytes = numel * elem_size
                category = "other (norm, emb, etc.)"

            total_bytes += param_bytes
            per_param.append({
                "name": param_name,
                "category": category,
                "shape": str(list(param.shape)),
                "bytes": param_bytes,
            })
            continue

        # Part of a CCLinear module
        cc_module = cc_modules[matched_cc_name]
        suffix = param_name[len(matched_cc_name) + 1:]

        if suffix == "compressed_linear.clinear.weight":
            quant = cc_module.quant
            if quant is not None:
                param_bytes, _ = _calc_compressed_bytes(param.shape, quant, elem_size)
                category = f"CCLinear_base ({quant} quant)"
            else:
                param_bytes = numel * elem_size
                category = "CCLinear_base (no quant)"
        elif "L_residual" in suffix or "R_residual" in suffix:
            compensation_quant = cc_module.compensation_quant
            if compensation_quant is not None:
                param_bytes, _ = _calc_compressed_bytes(param.shape, compensation_quant, elem_size)
                category = f"SALR residual ({compensation_quant} quant)"
            else:
                param_bytes = numel * elem_size
                category = "SALR residual (bf16)"
        else:
            param_bytes = numel * elem_size
            category = "CCLinear_other"

        total_bytes += param_bytes
        per_param.append({
            "name": param_name,
            "category": category,
            "shape": str(list(param.shape)),
            "bytes": param_bytes,
        })

    # ---- Aggregate across all PP ranks via all_reduce ----
    from megatron.core import parallel_state as mpu

    # Build per-category bytes for this rank
    local_cat_bytes = defaultdict(int)
    for p in per_param:
        local_cat_bytes[p["category"]] += p["bytes"]

    # Fixed category list for all_reduce (consistent across ranks)
    FIXED_CATEGORIES = [
        "CCLinear_base (hif4 quant)",
        "CCLinear_base (mxfp4 quant)",
        "CCLinear_base (mxfp8 quant)",
        "CCLinear_base (nf4 quant)",
        "CCLinear_base (fp4 quant)",
        "CCLinear_base (no quant)",
        "SALR residual (bf16)",
        "SALR residual (hif4 quant)",
        "SALR residual (mxfp4 quant)",
        "SALR residual (mxfp8 quant)",
        "LoRA A/B",
        "LoRA A/B (hif4 quant)",
        "LoRA A/B (mxfp4 quant)",
        "LoRA A/B (mxfp8 quant)",
        "LoRA A/B (nf4 quant)",
        "CCLinear_other",
        "other (norm, emb, etc.)",
    ]

    # Convert to tensors (fixed categories + 1 total)
    device = 'npu' if torch.npu.is_available() else 'cpu'
    n_cats = len(FIXED_CATEGORIES)
    cat_bytes_tensor = torch.zeros(n_cats + 1, dtype=torch.long, device=device)

    for i, cat in enumerate(FIXED_CATEGORIES):
        cat_bytes_tensor[i] = local_cat_bytes.get(cat, 0)
    cat_bytes_tensor[n_cats] = total_bytes

    # all_reduce across PP ranks to combine all stages
    torch.distributed.all_reduce(cat_bytes_tensor)

    global_total_bytes = cat_bytes_tensor[n_cats].item()

    # Also all_reduce params count
    param_count_tensor = torch.tensor([total_params_count], dtype=torch.long, device=device)
    torch.distributed.all_reduce(param_count_tensor)
    global_total_params_count = param_count_tensor.item()

    # ---- Print aggregated result on rank 0 ----
    if print_detail and torch.distributed.get_rank() == 0:
        print_rank_0("\n" + "=" * 105)
        print_rank_0("  SALR Total Parameter Storage Report (All PP Ranks Combined)")
        print_rank_0("=" * 105)

        print_rank_0("  Summary by Category:")
        print_rank_0(f"  {'Category':<50s} {'Bytes':>14s}")
        print_rank_0("-" * 105)
        for i, cat in enumerate(FIXED_CATEGORIES):
            gb = cat_bytes_tensor[i].item()
            if gb > 0:
                print_rank_0(f"  {cat:<50s} {gb:>14,}")
        print_rank_0("-" * 105)
        print_rank_0(f"  {'TOTAL (All PP Ranks)':<50s} {global_total_bytes:>14,}")
        print_rank_0(f"  {'Total Params Count':<50s} {global_total_params_count:>14,}")
        print_rank_0("=" * 105)

    return {
        "total_bytes": global_total_bytes,
        "total_params_count": global_total_params_count,
    }
