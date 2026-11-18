# S5 使用指南：Transformer 与小型 ViT

源码版本 `v0.5.0-alpha.2`。使用已有运行环境双击 [CPU 启动入口](../启动S5-CPU.cmd) 或 [CUDA 启动入口](../启动S5-CUDA.cmd)。两个入口分别保存到 `%LOCALAPPDATA%\ModelCircuitStudio-S5-CPU` 和 `%LOCALAPPDATA%\ModelCircuitStudio-S5-CUDA`，不会覆盖 S4 便携版状态。S4 EXE 不会加载新代码。

1. 在实验面板选择「ImageFolder 图片分类」，设置数据目录、图像大小和通道，检查数据。可先创建分类起始模型，再将旧分类器替换为 ViT。保留对应的模型输入和 `logits` 输出。
2. 在左侧「Transformer」分类或搜索框查找「小型 ViT 分类器」。配置图像大小、patch、类别数、头数和维度。初次插入新格式芯片时确认升级；可一步撤销插入与升级。
3. 将图片输入连接到 `x`，将 ViT 的 `logits` 接到模型同名输出。默认模型接受 32×32 RGB 图片，patch 为 4，包含 65 个 token、4 个头、2 层 Encoder；默认使用随机权重。调整类别数时匹配数据检查结果。实验 batch 必须符合图的资源契约。
4. 双击 ViT 逐层展开 Encoder、Block 和 Attention。选中名为 `attention` 的 Softmax 调用，在张量探针中设置样本序号、头号及查询/键区间，点击「下次采集」。区间采用左闭右开；采集内容来自选定 checkpoint 的验证前向。
5. 在实验面板选择 `cpu` 或现有 `cuda:0`，运行实验。结束后查看测试结果、权重变化和热图。保存 JSON 可恢复画布、页签与采样请求；重新打开后需重新检查外部数据。改变探针请求后重新运行才有对应结果。
6. 通过 Python 双向编辑导出独立 PyTorch 模型，可在受支持的语法范围内修改并导回。Python 导出保存模型结构与初始化方式，已训练权重在 checkpoint 内。不能导入任意 Python 程序。

Attention 可单独配置 causal 或外部布尔掩码；`true` 表示保留，不能静默合并两种掩码。首批不支持预训练模型下载、动态序列长度、文本生成、多 GPU、AMP 或任意任务后端。
