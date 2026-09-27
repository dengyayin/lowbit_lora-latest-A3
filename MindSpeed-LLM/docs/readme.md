## MindSpeed-LLM Docs 目录

<table>
  <thead>
    <tr>
      <th>内容</th>
      <th>链接</th>
      <th>备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">环境安装指导</td>
      <td><a href="./pytorch/install_guide.md">install_guide.md</a></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1">快速入门</td>
      <td><a href="./quick_start.md">quick_start.md</a></td>
      <td>基于pytorch/mindspore后端的入门指导，从环境安装到预训练拉起</td>
    </tr>
    <tr>
      <td rowspan="3">仓库支持模型清单</td>
      <td><a href="pytorch/models/dense_model.md">dense_models</a></td>
      <td>稠密模型清单</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/moe_model.md">MOE_models</a></td>
      <td>MOE模型清单</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/ssm_model.md">SSM_models</a></td>
      <td>SSM模型清单</td>
    </tr>
    <tr>
      <td rowspan="1">特性清单</td>
      <td rowspan="1"><a href="pytorch/features/">features</a></td>
      <td rowspan="1">收集了部分仓库支持的性能优化和显存优化的特性</td>
    </tr>
    <tr>
      <td rowspan="4">模型前处理操作</td>
      <td rowspan="1"><a href="pytorch/solutions/checkpoint_convert.md">checkpoint_convert</a></td>
      <td rowspan="1">支持mcore、hf、lora等各种不同格式权重间的部分转换路径</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/pretrain/pretrain_dataset.md">pretrain_dataset</a></td>
      <td rowspan="3">支持的数据集格式包括<a href="pytorch/solutions/finetune/datasets/alpaca_dataset.md">alpaca</a>,<a href="pytorch/solutions/finetune/datasets/pairwise_dataset.md">pairwise</a>,<a href="pytorch/solutions/finetune/datasets/sharegpt_dataset.md">sharegpt</a></td>
    </tr>
    <tr>
      <td rowspan="1">finetune_dataset(待补充)</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/preference-alignment/process_reward_dataset.md">reward_dataset</a></td>
    </tr>
    <tr>
      <td rowspan="2">预训练方案</td>
      <td><a href="pytorch/solutions/pretrain/pretrain.md">pretrain</a></td>
      <td>多样本预训练方法</td>
    </tr>
    <tr>
      <td><a href="pytorch/solutions/pretrain/pretrain_eod.md">pretrain_eod</a></td>
      <td>多样本pack预训练方法</td>
    </tr>
    <tr>
      <td rowspan="5">模型微调方法</td>
      <td rowspan="1"><a href="pytorch/solutions/finetune/instruction_finetune.md">instruction_finetune</a></td>
      <td rowspan="1">模型全参微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/multi_sample_pack_finetune.md">multi_sample_pack_finetune</a></td>
      <td rowspan="1">多样本Pack微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/multi-turn_conversation.md">multi-turn_conversation</a></td>
      <td rowspan="1">多轮对话微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/lora_finetune.md">lora_finetune</a></td>
      <td rowspan="1">模型lora微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/qlora_finetune.md">qlora_finetune</a></td>
      <td rowspan="1">模型qlora微调方案</td>
    </tr>
    <tr>
      <td rowspan="3">模型推理方法</td>
      <td rowspan="1"><a href="pytorch/solutions/inference/inference.md">inference</a></td>
      <td rowspan="1">模型推理</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/inference/chat.md">chat</a></td>
      <td rowspan="1">对话</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/features/yarn.md">yarn</a></td>
      <td rowspan="1">使用yarn方案来扩展上下文长度，支持长序列推理</td>
    </tr>
    <tr>
      <td rowspan="3">模型评估</td>
      <td><a href="pytorch/solutions/evaluation/evaluation_guide.md">evaluation</a></td>
      <td rowspan="1">模型评估方案</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/models