#!/bin/bash
source /home/z00663592/set_env.sh
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

a_quant=mxfp8
# please fill these path configurations
TOKENIZER_PATH="/data/z00663592/models/Qwen3-32B/"
CHECKPOINT="/data/z00663592/ckpt/qwen3_32b/qwen3_32b_pp4_lr_5e-5_lora_rank_16_compensation_rank16_attn_hif4_0.05_fc1_hif4_0.2_fc2_hif4_0.05_lr_mult0.006"
DATA_PATH="/data/z00663592/datasets/cais-mmlu/data/test2"
TASK="mmlu"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=16
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
        --num-layers 64 \
        --hidden-size 5120 \
        --ffn-hidden-size 25600 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens 2 \
        --max-position-embeddings 32768 \
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
        --norm-topk-prob \
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
        --high-sparsity-prune-method 0.2 \
        --low-sparsity-prune-method 0.05 \
        --high-sparsity-target-modules linear_fc1 \
        --low-sparsity-target-modules linear_qkv linear_proj linear_fc2 \
        --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --activation-quant $a_quant \
        --lora-quant mxfp8 \
        --compensation-quant mxfp8 \
        2>&1 | tee logs/evaluate/qwen3-32b/evaluate_qwen3_32b_pp4_lr_5e-5_lora_rank_16_compensation_rank16_attn_hif4_0.05_fc1_hif4_0.2_fc2_hif4_0.05_lr_mult0.006_${TASK}_lora_residual_mxfp8_activation_${a_quant}.log
