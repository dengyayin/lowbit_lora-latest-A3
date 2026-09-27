#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

export NPU_VISIBLE_DEVICES=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

lr=$1
output_name="Llama-2-7b-chat-pp4-mxfp4-hf2mg_lr${lr}_lora-rank64_base_hsprompt"

# modify script model path and tokenizer path
TOKENIZER_PATH="/home/wusen/hf_model/Llama-2-7b-chat-hf"  #tokenizer path
CHECKPOINT="/home/wusen/ckpt/llama2/${output_name}"  #model path
# LORA_CHECKPOINT="/home/wusen/ckpt/${output_name}"  # finetune results path
# configure task and data path
DATA_PATH="/home/wusen/datasets/mmlu/data/test"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# configure generation parameters
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 4096 \
       --max-new-tokens 2 \
       --evaluation-batch-size 1 \
       --max-position-embeddings 4096 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 4  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32  \
       --swiglu \
       --disable-bias-linear \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --bf16  \
       --micro-batch-size 1  \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       --use-mcore-models \
       --lora-r 64 \
       --lora-alpha 128 \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --prompt-type llama2 \
       --seed 42 2>&1 | tee logs/evaluate/llama2/evaluation_${output_name}_${TASK}.log



