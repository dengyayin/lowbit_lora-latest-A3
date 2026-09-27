#!/bin/bash
source /home/z00663592/set_env.sh

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

MASTER_ADDR=localhost
MASTER_PORT=6009
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

TP=1
PP=4
SEQ_LEN=4096

# lr=$1
# quant_method=$2
output_name="Mixtral-8x7B-mcore-pp4-no-groupedgemm-hif4"
CHECKPOINT="/data/z00663592/mcore_model/${output_name}"

TOKENIZER_PATH="/data/z00663592/models/Mixtral-8x7B-Instruct-v0.1"
DATA_PATH="/data/z00663592/datasets/cais-mmlu/data/test"
TASK="mmlu"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 8 \
    --moe-router-topk 2 \
    --expert-model-parallel-size 1 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --task $TASK \
    --task-data-path $DATA_PATH \
    --max-new-tokens 2 \
    --alternative-prompt \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --rotary-base 1000000 \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --exit-on-missing-checkpoint \
    --attention-softmax-in-fp32 \
    --load ${CHECKPOINT}  \
    --no-load-optim \
    --no-load-rng \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --bf16 \
    --seed 42 \
    --max-eval-samples 30 \
"



torchrun $DISTRIBUTED_ARGS evaluation.py \
    $GPT_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl 2>&1 | tee logs/evaluate/mixtral/evaluation_${output_name}_${TASK}_hif4_no_finetune.log
