# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./prm_dataset

python ./preprocess_data.py \
        --input ./data/math_shepherd/train-00000-of-00002.parquet \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/Mixtral-hf/ \
        --output-prefix ./prm_dataset/math_shepherd_prm \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStyleProcessRewardHandler \
        --seq-length 4096 \
        --placeholder-token ки \
        --reward-tokens + - \