#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6080
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

lr=$1
quant_method=$2
output_name="llama2_7b_lora_compensation_lr-${lr}_lora-rank64_compensation-rank64_${quant_method}_sparse0.1"

CKPT_SAVE_DIR="/home/wusen/ckpt/llama2/${output_name}"
DATA_PATH="./finetune_dataset/llama2_new/mmlu"
TOKENIZER_MODEL="/home/wusen/hf_model/Llama-2-7b-chat-hf"
CKPT_LOAD_DIR="/home/wusen/mcore_model/Llama-2-7b-chat-pp4"

TP=1
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

DIST_ALGO="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
"

MODEL_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
"

TRAINING_ARGS="
    --use-mcore-models \
    --prompt-type llama2 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --make-vocab-size-divisible-by 1 \
    --train-iters 3121 \
    --lr ${lr} \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.03 \
    --weight-decay 0.0 \
    --clip-grad 0.3 \
    --swiglu \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --init-method-std 0.01 \
    --initial-loss-scale 65536 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --variable-seq-lengths \
    --rotary-base 10000 \
    --norm-epsilon 1e-05 \
    --vocab-size 32000 \
    --log-throughput \
    --bf16 \
"
# 补偿暂不支持lora-fusion
FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"
COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 64 \
    --compensation-dtype bf16 \
    --quant-method ${quant_method} \
    --prune-method 0.1 \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 3121 \
    --eval-interval 500 \
    --eval-iters 50 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $DIST_ALGO \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $FINETUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $COMPENSATION_ARGS \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    --log-throughput \
    --distributed-backend nccl \
    2>&1 |tee logs/tune/llama2/tune_${output_name}_mmlu.log
