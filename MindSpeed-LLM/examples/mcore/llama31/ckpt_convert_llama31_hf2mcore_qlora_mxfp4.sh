# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf llama2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 4 \
   --load-dir /home/wusen/hf_model/Llama-3.1-8b-Instruct/ \
   --save-dir /home/wusen/mcore_model/Llama-3.1-8b-instruct-pp4-nf4 \
   --tokenizer-model /home/wusen/hf_model/Llama-3.1-8b-Instruct/tokenizer.json \
   --qlora-nf4 \
   2>&1 | tee logs/ckpt_convert/llama31_8b_instruct_hf2mcore_pp4_qlora_nf4.log