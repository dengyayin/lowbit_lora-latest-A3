source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir /home/wusen/mcore_model/Llama-2-7b-chat-pp4-mxfp4 \
    --lora-load /home/wusen/ckpt/llama2/Llama-2-7b-chat-pp4-mxfp4_lr9e-5_lora-rank64_base_no_fusion_align_to_hf_loraplus_test \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-lora-to-hf \
    --save-dir /home/wusen/hf_model/Llama-2-7b-chat-hf    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/