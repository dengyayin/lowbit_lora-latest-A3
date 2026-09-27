import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir  = "/data/models/Qwen3-32B"
merged_dir = "/data/d30082694/ckpt/qwen3-32b/Qwen3_32b_lr5e-5_rank16_hif4_w4a4_1000_hf_merged"

# 1) 加载 base 和 merged（两个独立完整模型）
tok = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)  # merged 里已有 tokenizer
base   = AutoModelForCausalLM.from_pretrained(base_dir,  torch_dtype=torch.bfloat16, trust_remote_code=True)
merged = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)

# 2) 确认 merged 参数规模（与 base 相同是预期内，LoRA 已合入权重）
print("base params  :", sum(p.numel() for p in base.parameters())/1e9, "B")
print("merged params:", sum(p.numel() for p in merged.parameters())/1e9, "B")

# 3) logits 差异：merged 应明显偏离 base（说明 LoRA 效果保留），且非完全随机
ids = tok("What can you do?", return_tensors="pt")
with torch.no_grad():
    lb = base(**ids).logits
    lm = merged(**ids).logits
print("max |base vs merged logits diff| =", (lb - lm).abs().max().item())
# >>0 → 合并后的模型确实带着 LoRA 效果；==0 → 说明 LoRA 没合进去（异常）

# 4) 生成对比（同一个 prompt）
for name, mdl in [("base", base), ("merged(base+LoRA)", merged)]:
    out = mdl.generate(**ids, max_new_tokens=30)
    print(f"[{name}]", tok.decode(out[0], skip_special_tokens=True))

# 5) 权重一致性抽查：确认合并后的全连接权重和 base 不一样（LoRA 已写入主干）
import torch
def layer_state(mdl, idx=0):
    return mdl.model.layers[idx].self_attn.q_proj.weight
print("q_proj weight identical to base?",
    torch.equal(layer_state(base), layer_state(merged)))
# False → 主干权重已带 LoRA 增量（正确）；True → 没合进去（异常）