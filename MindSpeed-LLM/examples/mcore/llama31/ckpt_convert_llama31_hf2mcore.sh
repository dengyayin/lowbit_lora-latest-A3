# 修改 ascend-toolkit 路径
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

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
   --load-dir /data/z00663592/models/Llama-3.1-8B-Instruct/ \
   --save-dir /data/z00663592/mcore_model/llama3.1-8b-pp4-hif4 \
   --qlora-hif4 \
   --tokenizer-model /data/z00663592/models/Llama-3.1-8B-Instruct/tokenizer.json  # --num-layer-list 17,20,22,21 等参数根据模型需求添加