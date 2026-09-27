import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 基础模型（完整权重）+ 适配器
base_dir = "/data/models/Qwen3-32B"
adapter_dir = "/data/d30082694/ckpt/qwen3-32b/Qwen3_32b_lr5e-5_rank16_hif4_w4a4_1000_hf/mg2hf"

# 1) 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
)

# 2) 加载 LoRA 适配器并合并进主权重
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()          # 把 LoRA 折进 base 权重
model = model.to(torch.bfloat16)

# 3) 保存合并后的完整模型（加 config + tokenizer）
merged_dir = "/data/d30082694/ckpt/qwen3-32b/Qwen3_32b_lr5e-5_rank16_hif4_w4a4_1000_hf_merged"
model.save_pretrained(merged_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True).save_pretrained(merged_dir)
print("saved to", merged_dir)