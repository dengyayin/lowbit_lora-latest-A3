export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# # 或指定碎片阈值，防止保留内存过度膨胀：
# export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed:$PYTHONPATH

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

lr=5e-5
output_name="qwen3_32b_hif4_lr${lr}_rank16"

# please fill these path configurations
CKPT_LOAD_DIR="/data/models/Qwen3-32B_mcore"
CKPT_SAVE_DIR="/data/d30082694/ckpt/qwen3-32b/${output_name}"
DATA_PATH="/data/l00814573/datasets/finetune_dataset/mmlu_instruction_qwen3_32b"
TOKENIZER_PATH="/data/models/Qwen3-32B"

TP=1
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=100

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
    # --use-distributed-optimizer
OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr $lr \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

# --sequence-parallel 开启TP则需加入该命令
MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 64 \
    --hidden-size 5120 \
    --ffn-hidden-size 25600 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --load ${CKPT_LOAD_DIR} \
	--save ${CKPT_SAVE_DIR} \
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --variable-seq-lengths \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    
"

COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 16 \
    --compensation-dtype bf16 \
    --quant-method hif4 \
    --high-sparsity-prune-method 0.2 \
    --low-sparsity-prune-method 0.05 \
    --high-sparsity-target-modules linear_fc1 \
    --low-sparsity-target-modules linear_qkv linear_proj linear_fc2 \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --use-trainable-residual \
    --lr-mult 0.006 \
"

torchrun $DISTRIBUTED_ARGS /data/l00814573/lowbit_lora-latest/MindSpeed-LLM/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $TUNE_ARGS \
    $MODEL_PARALLEL_ARGS \
    $COMPENSATION_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee /data/d30082694/lora_hif4/MindSpeed-LLM/logs/tune_${output_name}_$(date +%Y%m%d_%H%M%S).log