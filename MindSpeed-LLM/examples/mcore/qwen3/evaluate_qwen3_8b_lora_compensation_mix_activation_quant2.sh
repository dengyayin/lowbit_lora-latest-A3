#!/bin/bash
source /home/z00663592/set_env.sh

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


lr=5e-5
# lr_mult=$2
output="qwen3_8b_pp4_lora_rank16_compensate_rank16_lr${lr}_attn_hif4_0.1_mlp_hif4_0.15"

# please fill these path configurations
TOKENIZER_PATH="/data/z00663592/models/Qwen3-8B/"
CHECKPOINT="/data/z00663592/ckpt/qwen3_8b/${output}"
DATA_PATH="/data/z00663592/datasets/cais-mmlu/data/test"
TASK="mmlu"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6035
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=4
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-layers 36 \
        --hidden-size 4096 \
        --ffn-hidden-size 12288 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens 2 \
        --max-position-embeddings 4096 \
        --disable-bias-linear \
        --swiglu \
        --norm-epsilon 1e-6 \
        --padded-vocab-size 151936 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --load ${CHECKPOINT} \
        --no-chat-template \
        --kv-channels 128 \
        --qk-layernorm \
        --untie-embeddings-and-output-weights \
        --rotary-base 1000000 \
        --use-rotary-position-embeddings \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --normalization RMSNorm \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-gradient-accumulation-fusion \
        --attention-softmax-in-fp32 \
        --tokenizer-not-use-fast \
        --exit-on-missing-checkpoint \
        --no-masked-softmax-fusion \
        --micro-batch-size 1 \
        --no-load-rng \
        --no-load-optim \
        --seed 42 \
        --bf16 \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --use-compensation \
        --compensation-r 16 \
        --compensation-dtype bf16 \
        --quant-method hif4 \
        --high-sparsity-prune-method 0.15 \
        --low-sparsity-prune-method 0.1 \
        --high-sparsity-target-modules linear_fc1 linear_fc2 \
        --low-sparsity-target-modules linear_qkv linear_proj \
        --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --high-precision-front-back-attn \
        --lora-quant mxfp8 \
        --compensation-quant mxfp8 \
        --activation-quant mxfp8 \
        2>&1 | tee logs/evaluate/qwen3/evaluate_${output}_lora_residual_activation_mxfp8.log