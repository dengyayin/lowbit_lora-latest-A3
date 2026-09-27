#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=0

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6081
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

lr=$1
output_name="Llama-2-7b-chat-no-mp-mxfp4_lr${lr}_lora-rank64_base"

CKPT_SAVE_DIR="/home/wusen/ckpt/llama2/${output_name}"
DATA_PATH="./finetune_dataset/llama2/mmlu"
TOKENIZER_MODEL="/home/wusen/hf_model/Llama-2-7b-chat-hf"
CKPT_LOAD_DIR="/home/wusen/mcore_model/Llama-2-7b-chat-no-mp-mxfp4"

TP=1
PP=1

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
    --train-iters 3100 \
    --lr $lr \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --min-lr 1.25e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
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
    --adam-beta2 0.98 \
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

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 3100 \
    --eval-interval 1000 \
    --eval-iters 100 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $DIST_ALGO \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $FINETUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    --log-throughput \
    --distributed-backend nccl \
    2>&1 | tee logs/tune/llama2/tune_${output_name}.log
