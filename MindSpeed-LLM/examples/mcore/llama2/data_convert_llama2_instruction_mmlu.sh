# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset/llama2_new

python ./preprocess_data.py \
    --input /home/wusen/datasets/mmlu/auxiliary_train/train-00000-of-00001.parquet \
    --tokenizer-name-or-path /home/wusen/hf_model/Llama-2-7b-chat-hf/ \
    --output-prefix ./finetune_dataset/llama2_new/mmlu \
    --workers 32 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --overwrite-cache \
    --handler-name MMLUStyleInstructionHandler \
    --prompt-type llama2 \
    2>&1 | tee ./logs/dataset_convert/llama2_mmlu_data_convert_hsprompt.log
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传