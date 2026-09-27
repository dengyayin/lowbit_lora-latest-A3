source /home/z00663592/set_env.sh

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_VISIBLE_DEVICES=12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15

# export NPU_VISIBLE_DEVICES=4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7


NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

lr=5e-5
output="qwen3_8b_pp4_lora_rank16_compensate_rank16_lr${lr}_attn_hif4_0.1_mlp_hif4_0.15"

# output="qwen3_8b_pp4_hif4_debug"

# please fill these path configurations
CKPT_LOAD_DIR="/data/z00663592/mcore_model/qwen3-8b-pp4"
CKPT_SAVE_DIR="/data/z00663592/ckpt/qwen3_8b/${output}"
DATA_PATH="./finetune_dataset/qwen3_8b/mmlu"
TOKENIZER_PATH="/data/z00663592/models/Qwen3-8B/"

TP=1
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=3121

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

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
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr $lr \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.03 \
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

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP}
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval 300 \
    --eval-iters 50 \
    --no-load-optim \
    --no-load-rng
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
    --variable-seq-lengths \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

COMPENSATION_ARGS="
    --use-compensation \
    --compensation-r 16 \
    --compensation-dtype bf16 \
    --quant-method hif4 \
    --high-sparsity-prune-method 0.15 \
    --low-sparsity-prune-method 0.1 \
    --high-sparsity-target-modules linear_fc1 linear_fc2 \
    --low-sparsity-target-modules linear_qkv linear_proj \
    --compensation-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --compression-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $TUNE_ARGS \
    $COMPENSATION_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/tune/qwen3/tune_${output}.log
