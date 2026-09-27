# Demo提供的是Alpaca单轮对话数据集，多轮对话场景下需要更换数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /home/z00663592/set_env.sh
mkdir -p ./finetune_dataset/llama31

python ./preprocess_data.py \
	--input /data/z00663592/datasets/cais-mmlu/auxiliary_train/train-00000-of-00001.parquet\
	--tokenizer-name-or-path /data/z00663592/models/Llama-3.1-8B-Instruct/ \
	--output-prefix ./finetune_dataset/llama31/mmlu \
	--handler-name MMLUStyleInstructionHandler \
	--tokenizer-type PretrainedFromHF \
	--workers 16 \
	--log-interval 1000 \
	--prompt-type llama3 \
# 若使用Alpaca多轮对话数据集需要增加以下参数
# -map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
