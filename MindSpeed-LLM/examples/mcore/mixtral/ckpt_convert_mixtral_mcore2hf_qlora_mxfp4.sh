export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /home/z00663592/set_env.sh

# 设置需要的并行配置
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --load-dir /data/z00663592/models/Mixtral-8x7B-Instruct-v0.1 \
    --save-dir /data/z00663592/mcore_model/Mixtral-8x7B-mcore-pp4-no-groupedgemm-hif4/ \
    --tokenizer-model /data/z00663592/models/Mixtral-8x7B-Instruct-v0.1/tokenizer.model \
    --use-mcore-models \
    --qlora-hif4 \
    --model-type-hf mixtral 2>&1 | tee logs/ckpt_convert/mixtral-8x7b_hf2mcore_pp4_no_groupedgemm_hif4.log
