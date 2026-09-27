export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/Mixtral-mcore \
    --lora-load ./ckpt/mixtral-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/mixtral-lora2mcore \
    --use-mcore-models \
    --model-type-hf mixtral
