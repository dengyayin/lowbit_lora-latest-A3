# Demo提供的是Alpaca单轮对话数据集，多轮对话场景下需要更换数据集
# 请按照您的真实环境修改 set_env.sh 路径


source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset/llama31-test

python ./preprocess_data.py \
	--input /home/wusen/datasets/mmlu/all/test-00000-of-00001.parquet \
	--tokenizer-name-or-path /home/wusen/hf_model/Llama-3.1-8b-Instruct/ \
	--output-prefix ./finetune_dataset/llama31-test/mmlu \
	--workers 32 \
	--log-interval 1000 \
	--tokenizer-type PretrainedFromHF \
	--handler-name MMLUStyleInstructionHandler \
	--prompt-type llama3 \
    2>&1 | tee ./logs/dataset_convert/llama31_mmlu_data_convert.log
# 若使用Alpaca多轮对话数据集需要增加以下参数
# -map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
