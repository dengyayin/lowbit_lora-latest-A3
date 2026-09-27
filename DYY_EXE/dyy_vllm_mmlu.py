import argparse, os, glob, csv, json, re
from vllm import LLM, SamplingParams

# 昇腾 NPU 后端注册(需 pip install vllm-ascend)
try:
    import vllm_ascend  # noqa: F401
except ImportError:
    print("[warn] vllm_ascend 未安装")

try:
    from transformers import PreTrainedTokenizerBase, AddedToken
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        def _ase(self):
            out = []
            for t in self.all_special_tokens:
                out.append(t if isinstance(t, AddedToken) else AddedToken(t))
            return out
        PreTrainedTokenizerBase.all_special_tokens_extended = property(_ase)
        print("[patch] all_special_tokens_extended fallback installed")
except Exception as e:
    print("[patch warn]", repr(e))

CHOICES = ["A", "B", "C", "D"]
MMLU_TMPL = "The following are multiple choice questions (with answers) about {subject}.\n\n{question}\nAnswer: "


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)          # 已合并 LoRA 的完整 HF 模型
    p.add_argument("--data", required=True)           # .../mmlu_ori/test
    p.add_argument("--dev", default=None)             # .../mmlu_ori/dev (few-shot, 可选)
    p.add_argument("--adapter", default=None)         # 未合并时挂 LoRA(可选)
    p.add_argument("--max-tokens", type=int, default=2)          # = --max-new-tokens 2
    p.add_argument("--max-model-len", type=int, default=4096)    # = --seq-length 4096
    p.add_argument("--n-shots", type=int, default=0)
    p.add_argument("--tensor-parallel-size", type=int, default=4)  # = 4 卡(原 TP1*PP4)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)                # = --seed 42
    p.add_argument("--output", default=None)
    return p


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for q, a, b, c, d, ans in csv.reader(f):
            rows.append({"question": q, "A": a, "B": b, "C": c, "D": d, "answer": ans.strip()})
    return rows


def fmt_question(r):
    return (f"{r['question']}\n"
            f"A. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}")


def build_prompt(r, subject, dev_rows, n_shots):
    shot_block = ""
    for s in (dev_rows[:n_shots] if dev_rows else []):
        shot_block += fmt_question(s) + f"\nAnswer: {s['answer']}\n\n"
    q = shot_block + fmt_question(r)
    return MMLU_TMPL.format(subject=subject, question=q)


def strip_thinking(text):
    return re.sub(r"<\|?begin_of_thought\|?>.*?<\|?end_of_thought\|?>", "", text, flags=re.S)


def extract(answer):
    if not answer:
        return None
    cleaned = strip_thinking(answer)
    lines = cleaned.strip().splitlines()
    m_last = re.search(r"\b([ABCD])\b", lines[-1] if lines else cleaned)
    if m_last:
        return m_last.group(1)
    m_any = re.search(r"\b([ABCD])\b", cleaned)
    return m_any.group(1) if m_any else None


def main():
    args = build_parser().parse_args()

    enable_lora = bool(args.adapter)
    llm = LLM(model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,  # 4 卡
            trust_remote_code=True,
            seed=args.seed,                                  # 42
            dtype="bfloat16",                                # --bf16
            max_model_len=args.max_model_len,                # 4096
            enable_lora=enable_lora,
            max_lora_rank=16 if enable_lora else None,
            max_num_seqs=args.batch_size)
    sp = SamplingParams(temperature=0,                         # 贪心
                        max_tokens=args.max_tokens,            # 2
                        stop=["\n"] if args.max_tokens < 8 else None)

    dev_by_subj = {}
    if args.dev:
        for f in glob.glob(os.path.join(args.dev, "*_dev.csv")):
            subj = os.path.basename(f)[:-len("_dev.csv")]
            dev_by_subj[subj] = read_csv(f)

    prompts, refs, subjs = [], [], []
    for f in sorted(glob.glob(os.path.join(args.data, "*_test.csv"))):
        subj = os.path.basename(f)[:-len("_test.csv")]
        dev = dev_by_subj.get(subj, [])
        for r in read_csv(f):
            prompts.append(build_prompt(r, subj, dev, args.n_shots))
            refs.append(r["answer"])
            subjs.append(subj)

    preds = []
    for i in range(0, len(prompts), args.batch_size):
        outs = llm.generate(prompts[i:i + args.batch_size], sp, lora_request=None)
        preds += [extract(o.outputs[0].text) for o in outs]

    total = len(refs)
    acc_all = sum(1 for p, r in zip(preds, refs) if p and p == r)
    answered = sum(1 for p in preds if p)

    per_subj = {}
    for s, p, r in zip(subjs, preds, refs):
        per_subj.setdefault(s, [0, 0])
        per_subj[s][0] += (1 if p and p == r else 0)
        per_subj[s][1] += 1

    print("\n=== MMLU 总分 ===")
    print(f"Acc = {acc_all}/{total} = {acc_all/total:.4f} | 有答案 {answered}/{total} ({answered/total:.4f})")
    print("\n=== 各科目 ===")
    sacc = 0.0
    for s, (c, n) in sorted(per_subj.items()):
        a = c / n if n else 0.0
        sacc += a
        print(f"  {s:30s} {c:4d}/{n:<4d} = {a:.4f}")
    print(f"\nAverage(subject-level) = {sacc/max(len(per_subj),1):.4f}")

    if args.output:
        json.dump({"total_acc": acc_all / total, "answered": answered / total,
                    "n": total, "per_subject": {s: v for s, v in per_subj.items()},
                    "preds": preds},
                open(args.output, "w"), ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()