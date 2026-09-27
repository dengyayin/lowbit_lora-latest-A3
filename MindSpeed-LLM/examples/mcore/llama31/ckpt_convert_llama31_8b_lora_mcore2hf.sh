# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir /home/wusen/mcore_model/Llama-3.1-8b-instruct-pp4/ \
    --lora-load /home/wusen/ckpt/llama31/Llama-3.1-8b-instruct-pp4_lr5e-5_lora-rank32_base \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --save-lora-to-hf \
    --save-dir /home/wusen/hf_model/Llama-3.1-8b-Instruct/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama31_hf/mg2hg/