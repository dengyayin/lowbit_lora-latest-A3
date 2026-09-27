set -e

# source /home/z30083333/9.1.0/cann-9.1.0/set_env.sh
# source /home/cann/8.5.0/cann-8.5.0/set_env.sh
source /home/pxy/CANN9B030/cann-9.0.1/set_env.sh

export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed-LLM:$PYTHONPATH
export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed:$PYTHONPATH

# ---- NPU 可见性(按你的卡号) ----
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 若你的 vllm-ascend 需要退回 V1 图模式再打开下面这行;新版默认 V1 可注释掉
# export VLLM_USE_V1=0

# ---- 路径 ----
BASE_DIR="/data/models/Qwen3-32B"                                    # 原 TOKENIZER_PATH
MERGED="/data/d30082694/ckpt/qwen3-32b/Qwen3_32b_lr5e-5_rank16_hif4_w4a4_1000_hf_merged"
DATA="/data/l00814573/datasets/mmlu_ori/test"                        # 原 DATA_PATH
DEV="/data/l00814573/datasets/mmlu_ori/dev"                          # few-shot(可选, 原脚本未用)
LOG="/data/l00814573/lowbit_lora-latest/MindSpeed-LLM/logs/evaluate/mmlu_vllm_$(date +%Y%m%d_%H%M%S).log"

python dyy_vllm_mmlu.py \
    --model "$MERGED" \
    --data  "$DATA" \
    --dev   "$DEV" \
    --n-shots 5 \
    --max-tokens 2 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --seed 42 \
    --output /data/l00814573/lowbit_lora-latest/MindSpeed-LLM/logs/evaluate/mmlu_vllm_result.json \
    2>&1 | tee "$LOG"

echo "finish, log: $LOG"