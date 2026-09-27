#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6088
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

quant_method=$1
output_name="llama2_7b_compensation-rank64_${quant_method}_calibrated"

CKPT_SAVE_DIR="/home/wusen/ckpt/${output_name}"
DATA_PATH="./finetune_dataset/llama2/mmlu"
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
    --micro-batch-size 8 \
    --global-batch-size 32 \
    --make-vocab-size-divisible-by 1 \
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
    --variable-seq-lengths \
    --rotary-base 10000 \
    --norm-epsilon 1e-05 \
    --vocab-size 32000 \
    --log-throughput \
    --no-gradient-accumulation-fusion \
    --bf16 \
"

LORA_ARGS="
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 64 \
    --compensation-dtype bf16 \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --quant-method ${quant_method} \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 10000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS run_calibrate.py \
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
    2>&1 | tee logs/calibrate/llama2/calibrate_test.log
