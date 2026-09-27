# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /home/z00663592/set_env.sh

# export NPU_VISIBLE_DEVICES=15
# export ASCEND_RT_VISIBLE_DEVICES=15

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /data/z00663592/models/Qwen3-30B-A3B-Base \
    --save-dir /data/z00663592/mcore_model/qwen3-30b-a3b-pp4/ \
    --tokenizer-model /data/z00663592/models/Qwen3-30B-A3B-Base/tokenizer.json \
    --params-dtype bf16 \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe