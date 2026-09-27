export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的并行配置
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir /home/wusen/mcore_model/Mixtral-8x7B-mcore-pp4 \
    --lora-load /home/wusen/ckpt/mixtral/mixtral-8x7b-pp4_lr-5e-5_lora-rank16_base \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --save-lora-to-hf \
    --save-dir /home/wusen/hf_model/Mixtral-8x7B-Instruct-v0.1 \
    --use-mcore-models \
    --model-type-hf mixtral 2>&1 | tee logs/ckpt_convert/mixtral-8x7b_mcore2hf_lora.log
