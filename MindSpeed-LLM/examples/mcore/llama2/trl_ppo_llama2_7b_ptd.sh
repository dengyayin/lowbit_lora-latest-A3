#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6004
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
ACTOR_CKPT_LOAD_DIR="your model ckpt path"
REWARD_CKPT_LOAD_DIR="your model ckpt path"

TP=4
PP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --load $ACTOR_CKPT_LOAD_DIR \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --prompt-type llama2 \
    --seq-length 512 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --make-vocab-size-divisible-by 1 \
    --lr 1e-7 \
    --train-iters 1000 \
    --lr-decay-style constant \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --split 100,0,0
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

RL_ARGS="
    --stage trl_ppo \
    --max-new-tokens 256 \
    --max-length 512 \
    --rollout-batch-size 2 \
    --ref-model ${ACTOR_CKPT_LOAD_DIR} \
    --reward-model ${REWARD_CKPT_LOAD_DIR} \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $RL_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/trl_ppo_llama2_7b.log