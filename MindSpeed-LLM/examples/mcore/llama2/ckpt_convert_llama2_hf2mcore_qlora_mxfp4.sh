export CUDA_DEVICE_MAX_CONNECTIONS=1

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 开启 --qlora-nf4 选项使用 QLoRA
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --load-dir /home/wusen/hf_model/Llama-2-7b-chat-hf \
    --save-dir /home/wusen/mcore_model/Llama-2-7b-chat-pp4-mxfp4 \
    --tokenizer-model /home/wusen/hf_model/Llama-2-7b-chat-hf/tokenizer.model \
    --use-mcore-models \
    --qlora-mxfp4 \
    --model-type-hf llama2 \
    --params-dtype bf16 \
    2>&1 | tee logs/ckpt_convert/llama2_7b_chat_hf2mcore_pp4_qlora_mxfp4.log
