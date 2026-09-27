# 请按照您的真实环境修改 set_env.sh 路径
source /home/z00663592/set_env.sh
mkdir ./finetune_dataset/mixtral

python ./preprocess_data.py \
    --input /data/z00663592/datasets/cais-mmlu/auxiliary_train/train-00000-of-00001.parquet \
    --tokenizer-name-or-path /data/z00663592/models/Mixtral-8x7B-Instruct-v0.1 \
    --output-prefix ./finetune_dataset/mixtral/mmlu \
    --workers 32 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --overwrite-cache \
    --handler-name MMLUStyleInstructionHandler \
    --prompt-type mixtral
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
