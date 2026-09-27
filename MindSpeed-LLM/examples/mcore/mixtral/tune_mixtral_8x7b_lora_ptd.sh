#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6005
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

lr=$1
output_name="mixtral-8x7b-pp4_lr-${lr}_lora-rank16_base"

CKPT_SAVE_DIR="/home/wusen/ckpt/mixtral/${output_name}"
DATA_PATH="/home/wusen/finetune_dataset/mixtral/mmlu"
TOKENIZER_MODEL="/home/wusen/hf_model/Mixtral-8x7B-Instruct-v0.1"
CKPT_LOAD_DIR="/home/wusen/mcore_model/Mixtral-8x7B-mcore-pp4/"
LORA_CHECKPOINT="your lora ckpt path"

TP=1
PP=4
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=32

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-pre-softmax
"

GPT_ARGS="
    --num-workers 16 \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --vocab-size 32000 \
    --rotary-base 1000000 \
    --no-masked-softmax-fusion \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --make-vocab-size-divisible-by 1 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --save ${CKPT_SAVE_DIR} \
    --load ${CKPT_LOAD_DIR} \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --lr-warmup-fraction 0.01 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr ${lr} \
    --train-iters 3121 \
    --lr-decay-iters 3121 \
    --lr-decay-style constant \
    --min-lr 1.0e-6 \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
    --prompt-type mixtral \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --moe-grouped-gemm \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 95,5,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 3121 \
    --eval-interval 1000 \
    --eval-iters 100 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $MOE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    --log-throughput \
    --distributed-backend nccl \
    2>&1 | tee logs/tune/mixtral/tune_${output_name}_mmlu.log 