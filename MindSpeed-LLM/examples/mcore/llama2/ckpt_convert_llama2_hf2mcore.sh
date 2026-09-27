# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置，--num-layers-per-virtual-pipeline-stage 5，--params-dtype bf16 结合需要使用
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --load-dir /home/wusen/hf_model/Llama-2-7b-chat-hf-mxfp4 \
    --save-dir /home/wusen/mcore_model/Llama-2-7b-chat-pp4-mxfp4-hf2mg/ \
    --tokenizer-model /home/wusen/hf_model/Llama-2-7b-chat-hf/tokenizer.model \
    --use-mcore-models \
    --model-type-hf llama2 \
    --params-dtype bf16 \
    2>&1 | tee logs/ckpt_convert/Llama-2-7b-chat-pp4-mxfp4-hf2mg.log
