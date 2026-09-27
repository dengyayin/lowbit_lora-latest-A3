#!/bin/bash
source /home/z00663592/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6009
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
quant_method=$2
output_name="mixtral-8x7b-pp4_lr-${lr}_lora-rank16_compensation-rank32_${quant_method}"

CKPT_SAVE_DIR="/data/z00663592/ckpt/mixtral/${output_name}"
DATA_PATH="./finetune_dataset/mixtral/mmlu"
TOKENIZER_MODEL="/data/z00663592/models/Mixtral-8x7B-Instruct-v0.1"
CKPT_LOAD_DIR="/data/z00663592/mcore_model/Mixtral-8x7B-mcore-pp4-no-groupedgemm/"

TP=1
PP=4
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=32
SEQ_LEN=32768

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
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
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
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr ${lr} \
    --train-iters 3121 \
    --lr-decay-iters 3121 \
    --lr-decay-style cosine \
    --min-lr 0 \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
    --prompt-type mixtral \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 32 \
    --compensation-dtype bf16 \
    --quant-method ${quant_method} \
    --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1100 \
    --eval-interval 3121 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $MOE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FINETUNE_ARGS \
    $COMPENSATION_ARGS \
    --log-throughput \
    --distributed-backend nccl \
    2>&1 | tee logs/tune/mixtral/tune_${output_name}_mmlu.log 