#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NPU_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

lr=$1
output_name="Llama-3.1-8b-instruct-tuned-mxfp4-pp4"

# modify script model path and tokenizer path
TOKENIZER_PATH="/home/wusen/hf_model/Llama-3.1-8b-Instruct"
CHECKPOINT="/home/wusen/mcore_model/${output_name}"  #model path
# LORA_CHECKPOINT="/home/wusen/ckpt/llama31/${output_name}"

# configure task and data path
DATA_PATH="/home/wusen/datasets/mmlu/data/test"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# configure generation parameters
torchrun $DISTRIBUTED_ARGS evaluation.py   \
    --task-data-path ${DATA_PATH} \
    --task ${TASK}\
    --load ${CHECKPOINT}  \
    --exit-on-missing-checkpoint \
    --no-load-rng \
    --no-load-optim \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --max-new-tokens 2 \
    --evaluation-batch-size 1 \
    --micro-batch-size 1  \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 4  \
    --use-mcore-models \
    --seq-length 131072 \
    --max-position-embeddings 131072 \
    --num-layers 32  \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32  \
    --group-query-attention \
    --num-query-groups 8 \
    --swiglu \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --bf16  \
    --prompt-type llama3 \
    --seed 42 | tee logs/evaluate/llama31/evaluation_${output_name}_${TASK}.log



