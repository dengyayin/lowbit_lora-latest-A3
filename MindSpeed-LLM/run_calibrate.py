import torch
import torch_npu
import megatron

from transformers import AutoTokenizer

from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt import GPTModel

from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM
from mindspeed_llm.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from mindspeed_llm.tasks.posttrain.compensation.compensation_adaptor import apply_compensation_v2


def model_provider(pre_process, post_process):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model
    
def load_mg_model_and_tokenizer(mg_model_name_or_path, tokenizer_name_or_path, trust_remote_code, model_provider_func):
    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider_func,
        pretrained_model_name_or_path=mg_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code, local_files_only=True)
    return model, tokenizer

if __name__ == "__main__":
    initialize_megatron(
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True
        }
    )
    args = get_args()
    # print_rank_0("Arguments:")
    # for key, value in vars(args).items():
    #     print_rank_0(f"  {key}: {value}")
    
    print_rank_0("Loading model ...")
    model, tokenizer = load_mg_model_and_tokenizer(args.load, args.tokenizer_name_or_path, args.trust_remote_code, model_provider)
    print_rank_0("Model loaded.")
    print_rank_0(model)
    # for name, module in model.named_modules():
    #     print_rank_0(f"{name}")

    # print_rank_0(f"model on {model.device}")
    # model = model.cpu()
    # apply_compensation_v2(model)
    # print_rank_0("After applying compensation:")
    # print_rank_0(model)
    
    # import peft
    # from packaging import version
    # from peft import LoraConfig, get_peft_model, PeftModel, LoraModel
    # if version.parse(peft.__version__) <= version.parse('0.11.1'):
    #     setattr(peft.tuners.lora.LoraLayer, 'merge', peft.tuners.lora.Linear.merge)
    #     setattr(peft.tuners.lora.LoraLayer, 'unmerge', peft.tuners.lora.Linear.unmerge)
    #     setattr(peft.tuners.lora.LoraLayer, 'get_delta_weight', peft.tuners.lora.Linear.get_delta_weight)
    # from peft.tuners.lora import tp_layer
    # from mindspeed_llm.tasks.posttrain.lora.lora_moe import LoraParallelLinearMoE
    # tp_layer.LoraParallelLinear = LoraParallelLinearMoE

    # if hasattr(args, 'lora_fusion') and args.lora_fusion:
    #     from peft.tuners.lora.tp_layer import LoraParallelLinear
    #     from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import CCLoraParallelLinearForward
    #     LoraParallelLinear.forward = CCLoraParallelLinearForward
    #     if args.use_fused_mlp:
    #         from mindspeed_llm.tasks.posttrain.lora.cc_lora_mlp_forward import ParallelSwigluMLPLoRAForward
    #         from megatron.legacy.model.transformer import ParallelMLP
    #         from megatron.core.transformer.mlp import MLP
    #         ParallelMLP.forward = ParallelSwigluMLPLoRAForward
    #         MLP.forward = ParallelSwigluMLPLoRAForward

    # config = core_transformer_config_from_args(args)
    # lora_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=args.lora_target_modules,
    #     lora_dropout=0.0,
    #     bias="none",
    #     megatron_config=config,
    #     megatron_core="megatron.core",
    # )

    # model = get_peft_model(model, lora_config)
