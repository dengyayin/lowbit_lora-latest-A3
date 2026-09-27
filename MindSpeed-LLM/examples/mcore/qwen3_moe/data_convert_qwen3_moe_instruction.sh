# 请按照您的真实环境修改 set_env.sh 路径
source /home/z00663592/set_env.sh
mkdir ./finetune_dataset/qwen3_moe

python ./preprocess_data.py \
    --input /data/z00663592/datasets/cais-mmlu/auxiliary_train/train-00000-of-00001.parquet \
    --tokenizer-name-or-path /data/z00663592/models/Qwen3-30B-A3B-Base \
    --output-prefix ./finetune_dataset/qwen3_moe/mmlu \
    --handler-name MMLUStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 16 \
    --log-interval 1000 \
    --prompt-type qwen
