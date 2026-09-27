
# ######！！！注意！！！######
# ######！！！注意！！！######
# ######！！！注意！！！######
# ######！！！注意！！！######
# ######！！！注意！！！######
# 在启动该脚本之前先修改下面文件的内容
# /data/l00814573/lowbit_lora-latest/MindSpeed-LLM/mindspeed_llm/tasks/checkpoint/models.py

# def get_modules_from_pretrained(self, device_map="cpu", trust_remote_code=False):
#     Load Huggingface model.
#     if self.args_cmd.save_model_type == "hf":
#         load_dir = self.args_cmd.save_dir
#     else:
#         load_dir = self.args_cmd.load_dir

# 上面代码修改为↓

# def get_modules_from_pretrained(self, device_map="cpu", trust_remote_code=False):
#     load_dir = "/data/models/Qwen3-32B"

# 若不修改，则会从HF_OUT中加载模型，无法成功转换


source /home/z30083333/9.1.0/cann-9.1.0/set_env.sh

# export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed-LLM:/data/l00814573/lowbit_lora-latest/MindSpeed:$PYTHONPATH
export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed-LLM:$PYTHONPATH
export PYTHONPATH=/data/l00814573/lowbit_lora-latest/MindSpeed:$PYTHONPATH


lr=5e-5
TRAIN_ITERS=1000
OUT_NAME="Qwen3_32b_lr${lr}_rank16_hif4_w4a4_${TRAIN_ITERS}"
MG_CKPT="/data/d30082694/ckpt/qwen3-32b/${OUT_NAME}"
HF_OUT="/data/d30082694/ckpt/qwen3-32b/${OUT_NAME}_hf"

# 创建目标目录（包括 mg2hf 子目录）
mkdir -p ${HF_OUT}/mg2hf/

# 复制 tokenizer 文件
cp /data/models/Qwen3-32B/tokenizer.json ${HF_OUT}/mg2hf/ 2>/dev/null
cp /data/models/Qwen3-32B/tokenizer.model ${HF_OUT}/mg2hf/ 2>/dev/null || true
cp /data/models/Qwen3-32B/tokenizer_config.json ${HF_OUT}/mg2hf/ 2>/dev/null || true
cp /data/models/Qwen3-32B/added_tokens.json ${HF_OUT}/mg2hf/ 2>/dev/null || true
cp /data/models/Qwen3-32B/special_tokens_map.json ${HF_OUT}/mg2hf/ 2>/dev/null || true
cp /data/models/Qwen3-32B/config.json ${HF_OUT}/ 2>/dev/null || true

    # --load-hf-from-config \
python /data/l00814573/lowbit_lora-latest/MindSpeed-LLM/convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf qwen3 \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ${MG_CKPT} \
    --save-dir ${HF_OUT} \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-checkpoint-loosely \
    --qlora-hif4 \
    --save-lora-to-hf \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --lora-r 16 --lora-alpha 32