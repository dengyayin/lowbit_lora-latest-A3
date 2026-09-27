#!/bin/bash
source /home/z00663592/set_env.sh

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NPU_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

lr=$1
quant_method=$2
output_name="llama-3.1-8b-pp4_lr${lr}_lora-rank32_compensation-rank32_${quant_method}"

CKPT_SAVE_DIR="/data/z00663592/ckpt/llama31/${output_name}"
DATA_PATH="./finetune_dataset/llama31/mmlu"
CKPT_LOAD_DIR="/data/z00663592/mcore_model/llama3.1-8b-pp4"
TOKENIZER_PATH="/data/z00663592/models/Llama-3.1-8B-Instruct/"

TP=1
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --micro-batch-size 8 \
    --global-batch-size 32 \
    --sequence-parallel \
    --use-flash-attn \
    --prompt-type llama3 \
    --variable-seq-lengths \
    --use-rotary-position-embeddings \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --rotary-percent 1.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr $lr \
    --train-iters 3121 \
    --lr-decay-style cosine \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42 \
    --vocab-size 128256 \
"

FINETUNE_ARGS="
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 32 \
    --compensation-dtype bf16 \
    --quant-method ${quant_method} \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 3121 \
    --eval-interval 300 \
    --eval-iters 50 \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $FINETUNE_ARGS \
    $OUTPUT_ARGS \
    $COMPENSATION_ARGS \
    --distributed-backend nccl \
    | tee logs/tune/llama31/tune_${output_name}.log
