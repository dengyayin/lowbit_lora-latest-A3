# 压缩补偿参数说明文档

该部分文档详细说明了模型补偿（Compensation）、量化（Quantization）与剪枝（Pruning）相关的配置参数。

## 基础补偿配置 (General Compensation)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--use-compensation`** | `flag` | `False` | **启用补偿机制**。<br>开启后将允许使用额外的补偿适配器（Adapter）来弥补量化或剪枝带来的精度损失。 |
| **`--compensation-r`** | `int` | `64` | **补偿适配器的秩 (Rank)**。<br>类似于 LoRA 的 rank，决定补偿矩阵的大小。必须为正整数。 |
| **`--compensation-dtype`** | `str` | `'bf16'` | **补偿适配器的数据类型**。<br>例如 `bf16`, `fp16`。 |
| **`--use-svdquant`** | `flag` | `False` | **启用 SVD 量化模式**。<br>开启后将使用基于奇异值分解 (SVD) 的量化策略。 |
| **`--use-trainable-residual`** | `flag` | `False` | **启用可训练残差**。<br>根据可训练的残差项对参数进行分组。 |

## 量化设置 (Quantization Settings)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--quant-method`** | `str` | `None` | **模型权重的全局量化方法**。<br>支持的值：`nf4`, `mxfp4`, `mxfp8`, `hif4`。 |
| **`--activation-quant`** | `str` | `None` | **模型激活值的量化方法**。<br>指定用于激活值的量化策略。 |
| **`--compensation-quant`** | `str` | `None` | **补偿适配器的量化方法**。<br>如果补偿模块本身也需要量化，可在此指定。 |
| **`--lora-quant`** | `str` | `None` | **LoRA 模块的量化方法**。<br>指定用于 LoRA 参数的量化策略。 |

### 混合精度量化 (Mixed Precision Quantization)
用于对模型不同部分应用不同位宽的量化策略。

- **`--high-bit-quant-method`**: 高位宽模块的量化方法。
- **`--low-bit-quant-method`**: 低位宽模块的量化方法。
- **`--high-bit-target-modules`**: 指定应用高位宽量化的目标模块列表。
- **`--low-bit-target-modules`**: 指定应用低位宽量化的目标模块列表。
- **`--high-precision-front-back-attn`**: 指定前后若干层使用更高精度。<br>输入为层索引列表（整数），通常用于保护模型首尾层的 Attention 部分以维持性能。

## 剪枝设置 (Pruning Settings)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--prune-method`** | `str/float` | `None` | **全局剪枝方法**。<br>支持两种格式：<br>1. **浮点数** (0 < r < 1)：表示非结构化剪枝的比例。<br>2. **字符串** (如 `"2:4"`)：表示 N:M 结构化剪枝模式。 |

### 混合稀疏度剪枝 (Mixed Sparsity Pruning)
用于对模型不同部分应用不同的剪枝稀疏度。

- **`--high-sparsity-prune-method`**: 高稀疏度模块的剪枝比例或模式。
- **`--low-sparsity-prune-method`**: 低稀疏度模块的剪枝比例或模式。
- **`--high-sparsity-target-modules`**: 指定应用高稀疏度的目标模块列表。
- **`--low-sparsity-target-modules`**: 指定应用低稀疏度的目标模块列表。

## 模块目标与分组 (Target Modules & Grouping)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--compensation-target-modules`** | `list` | `[]` | **补偿目标模块**。<br>指定哪些模块需要挂载补偿适配器。 |
| **`--compression-target-modules`** | `list` | `[]` | **压缩目标模块**。<br>指定哪些模块是主要的压缩对象（量化或剪枝）。 |
| **`--lr-mult`** | `float` | `1.0` | **分组参数的学习率倍率**。<br>用于调整特定参数组的学习率。 |

## 路由与 MoE 相关 (Router & MoE)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`--train-router`** | `flag` | `False` | **训练 Router**。<br>在训练过程中是否更新 Router（路由网络）的参数。 |
| **`--enable-router-replay`** | `flag` | `False` | **启用 Router 回放机制**。<br>可能是指复用预先记录的路由决策。 |
| **`--router-replay-data-dir`** | `str` | `None` | **Router 回放数据目录**。<br>指定存储 Router 回放数据的文件夹路径。 |