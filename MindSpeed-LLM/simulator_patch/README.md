#Simulator_patch

simulator_patch针对MindSpeed目前有0.8.0和0.12.1两个版本，针对不同版本仿真时使用对应的simullator_patch

##使用方法：

1. 以 GPT 模型为例：将本模块置于 Megatron-LM or Mindspeed-LLM，确保与 'megatron' 位于同级目录：
```
Megatron-LM/
|-- megatron
|-- ...
|-- megatron_npu_adaptor
|-- ...
`-- simulator_patch
```
2. 两行代码轻松使能 MindSpeed 仿真适配，修改 ```pretrain_gpt.py``` 文件，在 ```import mindspeed.megatron_adaptor``` 上
新增一行 ```from simulator_patch import pre_patch```, 在 ```import mindspeed.megatron_adaptor```后
新增一行```from simulator_patch import post_patch```，即如下修改：
```
import os
import torch
+from simulator_patch import pre_patch
import mindspeed.megatron_adaptor
+from simulator_patch import post_patch
from functools import partial
```
3. 进行仿真训练