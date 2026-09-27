# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./pair_dataset

python ./preprocess_data.py \
        --input ./dataset/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B-v0.1/ \
        --output-prefix ./pair_dataset/orca_rlhf_mixtral \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type mixtral \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'
