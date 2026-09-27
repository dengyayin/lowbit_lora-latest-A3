#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="your data path"
SFT_LOAD_DIR="your actor/ref model ckpt path"
REWARD_LOAD_DIR="your reward/critic model ckpt path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_SAVE_DIR="your model save ckpt path"
TP=2
PP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-fused-swiglu \
    --use-mcore-models \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --bf16 \
    --use-rotary-position-embeddings \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 32.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 131072 \
    --max-position-embeddings 131072 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --lr 1e-7 \
    --train-iters 4 \
    --lr-decay-style constant \
    --min-lr 1e-7 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.00 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --seq-length 8192 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
"

RL_ARGS="
    --stage trl_ppo \
    --max-new-tokens 256 \
    --max-length 512
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $RL_ARGS \
    $PROFILE_ARGS \
    --tokenizer-not-use-fast \
    --distributed-backend nccl \
    --ref-model ${SFT_LOAD_DIR} \
    --reward-model ${REWARD_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/trl_ppo_llama32_1b.log
