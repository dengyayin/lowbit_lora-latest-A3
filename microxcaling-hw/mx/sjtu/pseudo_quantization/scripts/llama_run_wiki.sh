MODEL_SIZE=${1:-"7"}
TASKS=${2:-"arc_challenge"}
SHOTS=${3:-"0"}
QUANT_MODE=${4:-"ant"}
MXFP_MODE=${5:-"fp4_e2m1"}
GROUP_SIZE=${6:-"-1"}
WEIGHT_BIT=${7:-"4"}
METRIC=${8:-"ppl"}
W_ROUND=${9:-"w_search"}
X_ROUND=${10:-"up"}
# MSE_TYPE=${8:-"weight"}
# OUTLIER_TYPE=${7:-"none"}
# OUTLIER_RATIO=${8:-"-1"}
DESC=${11:-""}

# MODEL=/localdata_ssd/model/llama-${MODEL_SIZE}b-hf-transformers-4.29
# MODEL=/home/hyzhang/models/huawei/llama-${MODEL_SIZE}b-hf-transformers-4.29
# MODEL=/state/partition/wmhu/model/llama-2-${MODEL_SIZE}b-hf
# MODEL=/home/hyzhang/models/llama-2-${MODEL_SIZE}b-hf
# MODEL=/home/hyzhang/models/gpt2-xl
# MODEL=/state/partition/model/llama-2-7b-hf
MODEL=/localssd/hyzhang/llama-2-7b-hf
# MODEL=/localssd/hyzhang/model/llama-${MODEL_SIZE}b-hf-transformers-4.29
# MODEL=/localssd/hyzhang/model/llama-2-13b-hf
# MODEL=/localssd/hyzhang/model/opt-6.7b

# MODEL=/localssd/hyzhang/quantized_model
# OUTPUT_NAME=llama-${MODEL_SIZE}b
# OUTPUT_NAME=llama-2-${MODEL_SIZE}b

OUTPUT_NAME=llama-2-${MODEL_SIZE}b
OUTPUT_DIR=output/output_test_0625

mkdir -p $OUTPUT_DIR
# --dump_quant $OUTPUT_NAME \
# --load_awq /localdata_ssd/wmhu/llm/llm-awq/awq-model-zoo/llama-7b-w4-g128.pt

python -m awq.entry_wikitext --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --w_bit $WEIGHT_BIT  \
    --q_backend fake \
    --no_zero_point \
    --quant_mode $QUANT_MODE \
    --mxfp_mode $MXFP_MODE \
    --q_group_size $GROUP_SIZE \
    --metric $METRIC \
    --w_rounding_method $W_ROUND \
    --x_rounding_method $X_ROUND \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${WEIGHT_BIT}bit_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}_wr_${W_ROUND}_xr_${X_ROUND}_${DESC}.log 2>&1
