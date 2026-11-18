# S6 使用指南 · CSV 回归与组件库

在 ModelCircuit Studio `v0.6.0-alpha.1` 中，可以从一份 CSV 建立单目标回归实验。画布可编辑模型，实验面板负责选择数据、训练设备和检查点。

## 启动 Windows 桌面版

| 运行方式 | 本地入口 |
| --- | --- |
| CPU | `studio/启动模路S6-CPU.cmd` |
| CUDA | `studio/启动模路S6-CUDA.cmd` |

也可直接打开 `studio/dist/s6-cpu/ModelCircuitStudio/ModelCircuitStudio.exe` 或 `studio/dist/s6-cuda/ModelCircuitStudio/ModelCircuitStudio.exe`。复制到其他目录时，请保留完整的 `ModelCircuitStudio` 文件夹和 `_internal`。CUDA 版的设备列表也可选 `cpu`。两个 S6 包分别使用 `%LOCALAPPDATA%\ModelCircuitStudio-S6-CPU` 和 `%LOCALAPPDATA%\ModelCircuitStudio-S6-CUDA` 保存工作区。已有工作区通过「打开 JSON」迁移。

## 完成一次实验

1. 在右侧「实验」选择「CSV 数值回归」。进入「数据设置」，选取 UTF-8 CSV，点击「读取 CSV 列」。可用仓库里的[回归示例](../examples/s6-regression.csv)练习。
2. 示例的连续目标列是 `response`，特征列是 `length` 和 `mass`，分组列是 `group`，划分列是 `split`。实际数据可以使用自定义列名。请显式选择目标列。同一受试者有多行时，将受试者标识配置为分组列。
3. 点击「检查并冻结数据摘要」。显式划分使用 `train`、`val`、`test`。未指定划分列时，按随机种子划分，整组留在同一集合中。标准化统计量从训练集拟合。同一分组或完全相同的模型输入若出现在多个集合，检查会阻止训练。遇到重复测量时，先指定分组列或显式划分列，让重复记录留在同一集合中。
4. 点击「应用并关闭」，再点击「从数据创建起始模型」并确认替换。默认生成一个可编辑的线性回归图。也可自己连线，其根输入需要匹配数据的特征数，输出命名为 `prediction`，类型为 float32，形状为 `[N, 1]`。
5. 选择 `cpu` 或可用的 `cuda:N`。示例可用 10 个 epoch、batch size 4、学习率 0.05 和 AdamW。运行过程中可以停止。训练完成后可以查看损失曲线、MAE、RMSE、R²、预测张量和参数变化。
6. 如需续训，在「数据设置」指定上一次运行的 `last.pt`，设置更高的总 epoch 数，再运行。模型、数据、随机种子、批大小和优化器设置需要一致。

每次运行都会保存数据摘要、训练配置、最佳与最后检查点、指标和逐样本预测。最佳检查点由验证集的 MSE 选择。MAE 和 RMSE 使用目标的原始单位。常量目标或仅一个样本的集合显示「R² 未定义」。`predictions.csv` 中的 `residual` 为预测值减目标值。模型 Python 导出可以按平台支持的语法重新导入，实验数据和配置通过工作区 JSON 保存。

当前范围是单目标表格回归、MSE 和单设备执行。示例数据用于验证软件流程。训练产物与导出的逐样本文件保存在本机实验目录。

## 从分库中添加模型

左侧「全部库」依次显示基础组件、深度学习模型、传统机器学习模型和网络结构与融合。可按库筛选，也可搜索 `Linear`、`ResNet`、`ViT`、`SVM` 或中文名称。直接把条目拖入画布，点击「配置」可在放置前编辑参数。工程芯片与本地芯片另有各自入口。

支持向量分类（SVC）对应「CSV 数值分类」，支持向量回归（SVR）对应「CSV 数值回归」。数据检查后，点击「创建 SVC 起始图」或「创建 SVR 起始图」可生成已连接的模型。手工搭建时，将表格输入接到估计器的 `x`，将估计器的 `out` 接到名为 `prediction` 的根输出。右侧属性可编辑正则化常数 `C`，核函数支持 `rbf` 与 `linear`。

SVC 和 SVR 使用单个 CPU 估计器。运行后可在「最终评估与文件」查看分类准确率或回归指标、逐样本预测和可复现运行记录。Python 导出提供拟合与预测函数，接收已预处理的特征。导出代码内 `C` 和 `kernel` 的受支持修改可重新导入。旧版 PyTorch 检查点续训用于深度学习实验。

## 从源码运行或重建

在已有 CPU 环境时，可以从仓库根目录执行：

```powershell
& .\studio\.venv\Scripts\python.exe .\studio\desktop_entry.py --data-dir "$env:LOCALAPPDATA\ModelCircuitStudio-S6-Source"
```

已有对应训练环境和打包依赖时，可构建 CPU 或 CUDA 便携包。打包脚本使用对应的虚拟环境。

```powershell
& .\studio\build-s4-desktop.ps1 -Stage S6 -Runtime CPU
& .\studio\build-s4-desktop.ps1 -Stage S6 -Runtime CUDA
```
