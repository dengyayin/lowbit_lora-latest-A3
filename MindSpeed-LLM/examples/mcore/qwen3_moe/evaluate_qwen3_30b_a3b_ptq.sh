#!/bin/bash
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

source /home/z00663592/set_env.sh

# please fill these path configurations
TOKENIZER_PATH="/data/z00663592/models/Qwen3-30B-A3B-Base"
CHECKPOINT="/data/z00663592/mcore_model/qwen3-30b-a3b-pp4"
DATA_PATH="/data/z00663592/datasets/cais-mmlu/data/test"
TASK="mmlu"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=4
EP=1
SEQ_LENGTH=4096
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-intermediate-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type allgather \
    --moe-aux-loss-coeff 0.001
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
         $MOE_ARGS \
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --num-layers 48 \
        --hidden-size 2048 \
        --ffn-hidden-size 6144 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 4 \
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
        --use-compensation \
        --quant-method hif4 \
        --compensation-r 4 \
        --compensation-dtype bf16 \
        --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
        --max-eval-samples 20 \
        | tee logs/evaluate/qwen3_moe/evaluate_qwen3_30b_a3b_hif4_compensation_r4_no_finetune.log
