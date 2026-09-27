#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import sys
import types
import logging as logger
from .models import get_megatron_model
from .optim import get_optim_processor
from .optim_converter import get_optim_converter

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                            'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of megatron repository')
    parser.add_argument('--add-qkv-bias', action='store_true',
                        help='Add bias for attention qkv', default=False)
    parser.add_argument('--add-dense-bias', action='store_true',
                        help='Add bias for attention dense', default=False)
    parser.add_argument('--embed-layernorm', action='store_true',
                        help='Add embed layernorm for word embedding', default=False)
    parser.add_argument('--params-dtype', type=str,
                        help='Set weight dtype', default='fp16')
    group.add_argument('--post-norm', action='store_true',
                       help='post norm after attention or mlp.', default=False)
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='Usr moe grouped gemm.')
    group.add_argument('--load-from-legacy', action='store_true',
                       help='Is loader legacy')
    group.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                            'that returns a spec to customize transformer layer, depending on the use case.')
    group.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int, default=1,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-data-parallel-size', type=int, default=1,
                       help='Target data parallel size')
    group.add_argument("--w-pack", type=bool,
                       help='True is w_pack weight for llm',
                       default=False)
    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, seperated by comma; e.g., 4,4,4,4')
    group.add_argument('--use-mcore-models', action='store_true',
                       help='Use the implementation from megatron core')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='Usr moe grouped gemm.')
    group.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                            'that returns a spec to customize transformer layer, depending on the use case.')


def build_metadata(args, margs):
    # Metadata.

    # Layernorm has bias; RMSNorm does not.
    if hasattr(margs, 'normalization'):
        norm_has_bias = margs.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.spec = args.spec
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None
    md.checkpoint_args = margs
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.embed_layernorm = margs.embed_layernorm
    md.moe_grouped_gemm = margs.moe_grouped_gemm
    md.spec = margs.spec
    md.num_experts = getattr(margs, "num_experts", None)
    md.n_shared_experts = getattr(margs, "n_shared_experts", None)
    md.qk_layernorm = getattr(margs, "qk_layernorm", False)
    md.moe_intermediate_size = getattr(margs, "moe_intermediate_size", None)
    md.first_k_dense_replace = getattr(margs, "first_k_dense_replace", None)
    md.moe_layer_freq = getattr(margs, "moe_layer_freq", None)
    md.multi_head_latent_attention = getattr(margs, "multi_head_latent_attention", False)
    if md.multi_head_latent_attention:
        md.qk_rope_head_dim = getattr(margs, "qk_rope_head_dim", None)
        md.qk_nope_head_dim = getattr(margs, "qk_nope_head_dim", None)
        md.q_lora_rank = getattr(margs, "q_lora_rank", None)
        md.kv_lora_rank = getattr(margs, "kv_lora_rank", None)
        md.v_head_dim = getattr(margs, "v_head_dim", None)

    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    return md


def _load_checkpoint(model_provider, args):
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    if args.use_mcore_models and args.load_from_legacy:
        args.use_mcore_models = False

    model_src = get_megatron_model(model_provider, args_cmd=args)
    model_src.initialize_megatron_args(loader_megatron=True)

    src_margs = model_src.get_args()
    src_margs.moe_grouped_gemm = args.moe_grouped_gemm
    src_margs.spec = args.spec
    md = build_metadata(args, src_margs)

    model_dst = get_megatron_model(model_provider=model_provider, args_cmd=args, md=md)
    model_dst.initialize_megatron_args(saver_megatron=True)
    dst_margs = model_dst.get_args()

    src_optim_processor = get_optim_processor(src_margs, 'source')
    src_optim_processor.create_param_index_maps_for_checkpoints()
    src_optim_processor.split_optimizer_ckpt()
    dst_margs.num_layer_list = args.num_layer_list

    dst_optim_processor = get_optim_processor(dst_margs, 'target')

    optim_converter = get_optim_converter(src_optim_processor, dst_optim_processor)

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=4)
    for key in ['param', 'exp_avg', 'exp_avg_sq']:
        optim_converter.run(key, executor)
    executor.shutdown()

    dst_optim_processor.merge_optimizer_ckpt()
    optim_converter.remove_optimizer_tmp()

    logger.info("Done!")


def load_checkpoint(model_provider, args):
    try:
        _load_checkpoint(model_provider, args)
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the checkpoint: {e}")