# ML Forge Studio：原生桌面层级芯片工作台

2026-11-17最新源码：14个可展开模型入口（ResNet18/34/50、VGG11/13/16/19特征、SqueezeNet1.0/1.1）、本地芯片与原生工厂扩展、hardware-v5容量评估，以及v3页签/分屏保存恢复。新策略无固定参数量上限，旧工程可通过“硬件容量策略…”明确升级。使用`启动S4源码预览.cmd`或`启动S4-CUDA源码预览.cmd`，步骤和真实边界见[多模型与工作区指南](native/S4_MODELS_WORKSPACE_GUIDE.md)。此前[组件/连线验收](native/S4_COMPONENT_WORKBENCH_VERIFICATION.md)保留为历史。固定便携EXE仍为下文S3，新EXE打包尚未进行。

更新：2026-11-15，S3 自定义芯片与图片/表格配对分类。

**当前主入口是 Windows DearPyGui 桌面版。** 原版 ML Forge 就是 DearPyGui 程序；本阶段在 `studio/native` 中接入独立图执行核心与 S3 自定义芯片、配对数据真实分类，保留原生窗口、节点编辑器、层级芯片和左右面板。原 `ml_forge/` 业务源码保留。

S3 增加具名多输出芯片、接口/公开参数/绑定编辑、版本化芯片库、语义撤销/重做，以及图片+表格配对分类，保留 S2 CSV/ImageFolder 与 S1 演示。S3 源码、CPU/CUDA EXE 已通过本机实际训练与原生界面验收；每环境 backend 运行175项（174通过、1项 Windows symlink 权限跳过），native 67项全部通过。详情见 [S3 验证记录](native/S3_VERIFICATION.md)、[可携带验收记录](../docs/评审附件/2026-11-15-S3验收记录.json) 和[原生界面对照](../docs/images/native-s3-comparison.png)。[S2 验证记录](native/S2_VERIFICATION.md) 继续作为历史证据保留。

此前 `studio/frontend` 是 React/Vite 网页实验界面，作为早期 S1 实现与参考保留。桌面直接使用本地协调器和独立 worker，无需 Vite、浏览器或 HTTP 服务；网页控件与原生版的覆盖不同。

## 打开桌面版

双击 `studio/启动桌面版.cmd`，或进入以下目录打开程序：

```text
studio/dist/desktop/MLForgeStudio/
├── MLForgeStudio.exe
└── _internal/
```

CUDA 版入口保持为 `studio/启动CUDA桌面版.cmd` 或 `studio/dist/desktop-cuda/MLForgeStudio/MLForgeStudio.exe`。CPU/CUDA 固定输出路径现均为已验收的 S3 EXE，身份见 S3 验证记录。

这是目录式便携包，复制或移动时必须保留 exe 与 `_internal`。完整包无需用户安装 Python 或 Node.js。请按验证记录核对所用包版本；包存在不代表已完成干净 Windows 机器、签名安装器或跨机器兼容性验收。

详细步骤见 [原生桌面版使用说明](native/README.md)。默认状态目录为 `%LOCALAPPDATA%\MLForgeStudio`：`workspace.json` 保存模型与实验设置，启动优先恢复它并兼容旧 `draft.json`；`runs/<run_id>.json` 留存运行快照，真实训练文件位于 `runs/<run_id>/`；日志为 `logs/desktop.log`。“保存 JSON”另存完整工作区，也可打开旧纯图 JSON。

中文界面只读取 `C:/Windows/Fonts/msyh.ttc`，不修改系统字体或将其打包；缺失时中文可能无法完整显示。

## 自定义芯片与可回退编辑

进入目标芯片后打开“接口与芯片”，在“当前作用域接口”增删/改名输入输出、编辑 shape 和输出来源；上方定义选择器用于公开参数与库目标。已消费接口须先断开才能删除，新增第二输出会把原唯一输出的旧引用明确迁移到具名端口。多个输出分别显示插座，连线与探针选择具体端口；`call:id:port` 支持新结构，单输出旧 `call:id` 保持兼容。

“公开参数与绑定”提供真实参数行，编辑类型、默认值与局部 `instance.parameter` 路径列表。顶部撤销/重做覆盖语义命令，最多32状态、undo+redo 共32 MiB；移动、选择、导航和实验设置不进入模型历史，历史不会代替显式保存。

封装支持根图和芯片内部，可暴露多个输出；保留原片段并新增独立芯片，不自动替换原图或重接下游。选中片段内共享保留，跨封装边界不继承权重身份。`*.mlfchip.json` 保存版本化架构与依赖，包含接口/默认参数/绑定，不含训练权重；同 ID/版本不能被不同内容覆盖，导入冲突会一致改名。详细流程见 [原生指南](native/README.md)。

## 真实分类的操作路径

1. 在右侧“实验”选择“CSV 数值分类”、“ImageFolder 图片分类”或“配对图片与表格分类”，打开“数据设置”。CSV 指定路径、标签与数值特征列；ImageFolder 指定目录、输入尺寸和通道。
2. 配置 val/test 比例、显式 split 或患者分组，再点击“检查并冻结数据摘要”，核对三个分区的实际数量、类别、输入形状和内容摘要。CSV 标准化只拟合 train；坏文件或跨 split 内容/分组泄漏会明确失败。
3. 使用匹配的单输入或配对双输入模型，或点击“从数据创建起始模型…”并**确认替换当前模型**。CSV 生成 MLP，图片生成小型 CNN，配对模式生成 CNN+clinical MLP→Concat→logits；不会把同一数据张量隐式接到原默认双输入模型。
4. 设置 epochs、SGD/AdamW、学习率、batch、seed 与设备后运行。每轮完整训练 train，再评估 val；仅按 val loss 选最佳 checkpoint，最后加载最佳模型评估一次 held-out test。
5. 分别查看 train/val 曲线、最终 test 指标、实际 val 探针及参数梯度/更新来源。模型、实验设置或数据摘要变化会使旧结果过期，节点移动不影响测量身份。


配对 CSV 一行定义同一样本的图片与数值。必填角色默认列名为 `sample_id`（唯一非空）、`patient_id`（必选组）、`image_path`，另有标签与数值特征列；在列选择器中可映射相应列。`image_root` 留空指 CSV 父目录，图片不能越出该根目录。显式 split 或固定种子的患者组拆分同时作用于两路，单个 DataLoader 联合 shuffle，缺模态、坏图、重复 ID 和跨 split 患者/图片泄漏会失败；不同患者相同数值特征允许。默认 `image → image`、`tabular → clinical`，映射必须覆盖两个不同图输入，且纳入 checkpoint 身份。

同一患者有多行 CSV 时应指定 `group_column`，包括使用显式 train/val/test 列的情况；同一组不能跨分区。ImageFolder 支持 `根目录/类别/图片` 的分层拆分，或 `根目录/{train,val,test}/类别/图片` 的固定目录拆分。ImageFolder 模式没有患者分组字段，多张图属于同一患者时需在外部先按患者分好目录；配对模式则使用 CSV 中必选的患者组列。内容查重不能替代患者级隔离。

CPU 包使用 `torch 2.6.0+cpu`。独立 CUDA 环境 `studio/.venv-cuda` 和 CUDA EXE 使用 `torch 2.6.0+cu124`。S3 配对/芯片功能的源码及 CPU/CUDA EXE 均已实际完成训练、恢复、独立推理和原生界面验收；GPU 使用本机 RTX 4060 Laptop，CUDA EXE 内显式选 CPU 也已验证。CUDA 入口为 `studio/启动CUDA桌面版.cmd`，EXE 位于 `studio/dist/desktop-cuda/MLForgeStudio/MLForgeStudio.exe`，选择真实任务后可明确选择 `cpu` 或一个可用 `cuda:N`。原 CPU 包和启动入口保留。不可用 CUDA 会失败，不静默回退；不支持多 GPU、DDP 或远程训练。当前范围也不含 DICOM/NIfTI、3D、分割、LLM/RAG 或完整医学科研流程。

## checkpoint 与模型代码

真实分类每个完整 epoch 原子保存 `last.pt`，按 val loss 保存 `best.pt`，另留 manifest、run spec、指标与成功运行的 `predictions.csv`。在数据设置中选择 `last.pt` 可从完整 epoch 边界继续；图、数据、预处理、优化器、lr、batch、seed 和配对模态/图端口映射必须匹配。epochs 是总目标轮数，例如已完成2轮、设为10则继续第3至10轮。取消保留已提交的最后完整 epoch，当前未完成轮次不作为恢复点；第一轮前取消可能没有 checkpoint。重启不会自动恢复进程或加载历史实验列表。

工作区 JSON 保存模型和实验。Python 导入只静态解析受支持算子和显式 forward，**只替换模型，保留数据与训练设置**；导入后重新校验数据兼容性。CNN 加入 Conv2d、MaxPool2d、AdaptiveAvgPool2d、Flatten，使用版本化受限 profile；具名芯片为 `ports-pytorch-v3`，静态字典返回与字面量端口选择保留共享及调用次数。未知 Python、动态选择端口和动态控制流仍会拒绝。导出的 `.py` 是模型源码，不包含完整数据或训练 checkpoint。

S1 演示继续按原固定合成批次运行，它的 loss_before/loss_after 与准确率均来自同一批次，不是独立测试集，也没有训练恢复。具体语义及图编辑限制见原生指南。

## 桌面开发与构建

以下命令从仓库根目录执行。开发环境为 Python 3.12、CPU PyTorch 2.6.0、DearPyGui 2.3.1、PyInstaller 6.22.3，以及 Pillow、NumPy 等资源依赖；原生开发和构建不需要 Node.js。

```powershell
# 仅首次准备；已有 studio/.venv 时使用原环境
py -3.12 -m venv .\studio\.venv
.\studio\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\studio\.venv\Scripts\python.exe -m pip install Pillow numpy -r .\studio\requirements-desktop.txt

# 原生源码入口
.\studio\.venv\Scripts\python.exe .\studio\desktop_entry.py

# 执行 native/tests，失败即停止；随后生成目录式便携包
.\studio\build-desktop.ps1
```

CUDA 开发环境使用独立目录；以下是重建环境的开发命令，便携包使用者无需执行：

```powershell
.\studio\.venv\Scripts\python.exe -m venv .\studio\.venv-cuda
.\studio\.venv-cuda\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\studio\.venv-cuda\Scripts\python.exe -m pip install -r .\studio\requirements-desktop.txt
.\studio\build-desktop.ps1 -Runtime CUDA
```

CPU 输出为 `studio/dist/desktop/MLForgeStudio/MLForgeStudio.exe` 与同目录 `_internal`；CUDA 使用 `-Runtime CUDA`，输出在 `studio/dist/desktop-cuda/`。`-InstallDependencies` 可安装 `requirements-desktop.txt`，CPU torch 与额外资源依赖仍按上述方式准备。本轮两版实际构建及冻结运行均已完成，日志和包身份见 S3 验证记录。

原生版已提供接口、公开参数/绑定、本地芯片库、共享/冻结、嵌套封装与受限 Python 往返。多 loss/训练协议图、交互式单步、完整训练工程导出、3D/分割、LLM/RAG、多 GPU 与远程训练仍属后续范围。旧 `.mlf` 与 Studio `schema_version=1` 图格式不能直接混用。根包仍只发现 `ml_forge*`、排除 `studio*`，旧 `ml-forge` 入口和 Studio 入口相互独立；S1 源码基线与历史证据见 [VERIFICATION.md](VERIFICATION.md)。

S1 与 S2 历史归档保留于 `studio/dist/desktop/`；S2 文件为 `MLForgeStudio-Windows-x64-S2-CPU.zip`、`MLForgeStudio-Windows-x64-S2-CUDA.zip`、`MLForgeStudio-source-S2.zip`，身份见 [S2 记录](native/S2_VERIFICATION.md)。同目录 S3 的 `MLForgeStudio-Windows-x64-S3-CPU.zip`、`MLForgeStudio-Windows-x64-S3-CUDA.zip`、`MLForgeStudio-source-S3.zip` 已生成并通过 CRC；源码包逐项匹配当前文件。详细身份和冻结证据见 [native/S3_VERIFICATION.md](native/S3_VERIFICATION.md)，最终源码包哈希在本地 `output/native-s3/archive-source.json`。

## 私有上传状态

GitHub 上传按用户要求暂停。计划目标仍为用户个人 private 私有仓库，仅本人访问，不添加协作者；当前没有远端、提交或协作者，也未建立 Release 或分享链接。接续说明见 [HANDOFF.md](HANDOFF.md)。

## 早期网页实验入口（保留参考）

以下所有运行边界、测试数量和测量说明仅描述此前 React/Vite S1 网页实现，保留作历史参考；当前桌面 S3 能力以前文和 S3 验证记录为准，S2 记录作为历史保留。`启动工作台.cmd` 是网页启动器；桌面请使用前述 `启动桌面版.cmd`。网页数据路径和控件操作与原生版不同。

本机已准备本版本的 CPU PyTorch 与前端依赖；本次可追溯核查见 [VERIFICATION.md](VERIFICATION.md)。双击本目录的 **启动工作台.cmd**，或从 PowerShell 执行。下面所有相对路径命令均从仓库根目录执行；若已经进入 `studio`，启动命令改为 `.\start.ps1 -OpenBrowser`。

```powershell
.\studio\start.ps1 -OpenBrowser
```

工作台地址：`http://127.0.0.1:4173/`。Python API 监听 `127.0.0.1:8765`。服务仅绑定本机，不部署到外部网络。

首次在其他环境准备依赖，需要 Node.js/npm、Python 3.11 或 3.12，可运行：

```powershell
.\studio\setup.ps1 -PythonPath 'C:\path\to\python.exe'
.\studio\start.ps1 -OpenBrowser
```

安装的是 CPU PyTorch 2.6.0 和前端依赖，不下载训练数据或模型权重。日志位于 `studio/logs`。停止本启动器记录的服务：

```powershell
.\studio\stop.ps1
```

停止服务会结束尚未完成的本地实验，应先在界面停止任务并保存项目。

## 网页实验界面的使用路径

1. 打开默认双路分类模型，确认两个编码调用引用同一个 `encoder` 实例。
2. 双击编码芯片，查看内部 `Linear → ReLU`，点击面包屑“模型”返回根图。
3. 选择组件，在右侧修改公开参数、冻结状态或输入输出连接。语义改变后重新校验。
4. 在根图按 Shift 多选调用，使用“封装”生成独立芯片定义。也可从组件库“新建”模板，或在内部选择“另存芯片”。
5. 点击“运行实验”。默认任务在固定合成批次上执行20次SGD更新，不读取真实影像或临床数据。
6. 查看训练曲线、节点数据探针、参数梯度与更新范数；进入芯片时，探针按具体调用路径定位。
7. 打开“Python”，导出独立模型文件。对受支持代码进行修改后，使用“重新解析到画布”。
8. 用保存按钮导出项目 JSON。浏览器也保存最近草稿；需要归档时使用显式 JSON 文件。

模型输入默认是 `a、b: float32[B,4]`，共享芯片映射到8维，经相加和分类头输出2类logits与融合特征。模型总计58个参数，其中编码器被调用两次但只拥有一份权重。

## S1 核心支持与边界

支持具名根图输入/多输出、真正按边执行、分支、共享实例、嵌套单输出芯片、公开参数与立即子组件绑定、冻结、七种基础算子（Linear/ReLU/GELU/Add/Concat/LayerNorm/Dropout）。参数、shape、循环、非法引用及可推导的资源预算在构建前检查；实际张量大小和结果 JSON 字节预算在运行或传输时另行检查。普通环被拒绝。

代码 profile 为 `s1-pytorch-v1`：普通PyTorch组件类、ModuleDict、显式forward调用和受支持的源码修改。导入只解析当前源码，不执行上传代码。未知任意Python、动态控制流、自定义helper语义、opaque内部保留等未在当前版本实现，会明确拒绝；不声称已覆盖计划书全部R2/R3能力。导出模型文件不依赖工作台，但需要安装PyTorch并由调用者提供tensor输入；完整数据/训练实验包导出是后续能力。

该历史 S1 网页版本的运行按钮仅启用本地CPU、固定合成数据的二分类学习探针。数据源、seed、batch和训练步数显示在配置及运行快照中。没有接入真实医学队列、3D训练、大模型、RAG、传统ML后端或远程GPU，不能把这一闭环当成全面科研平台完成。

结构校验通过不等于符合该演示训练任务：所有根输入必须是二维 `[batch, features]`，batch 为 `null` 或配置的 `batch_size`；必须保留名为 `logits`、形状为 `[batch, 2]` 的输出，且其计算依赖至少一个可训练参数。将 `logits` 重命名为其他名称、冻结全部有效参数或改变输出类别数后，需要先恢复这些任务条件才能运行；其他合法图仍可编辑、校验或导出模型代码。

## 测量含义

- loss_before/loss_after和准确率均在同一个固定合成批次、`eval()` 模式下评估，不是独立测试集结果，也不能说明泛化性能。
- history是 `train()` 模式下各次更新前的训练loss；含 Dropout 时，不应将它与 `eval()` 下的前后损失视为完全同一计算条件。参数梯度是末次更新前的梯度，更新范数比较初始化与最后状态。
- 探针统计使用完整张量，`values`为有界预览；显示原始数量和预览采样数，未采集与零值不同。
- root `outputs`也是有界预览，`output_metadata`声明原shape、元素数、预览数与是否截断；模型forward仍返回完整tensor。
- 非有限输出在S1中使实验明确失败。进程结果和持久化有独立JSON字节预算，超限会报告错误。
- 设计revision和运行revision分开。运行中继续编辑会令旧结果标为过期；worker不回读画布。

## 网页实验版的文件与运行

`backend/studio_core`是独立图校验、执行和代码转换；`backend/server.py`是本地协调器；`backend/worker.py`在独立进程运行PyTorch；`frontend/src`是层级图、属性、代码与仪器界面。

每次运行的快照、配置、事件和测量结果写入 `studio/runs/<run_id>.json`。最多两个本地实验并行。正常取消由worker响应，超时或异常由监管器根据真实进程退出确认。该版本保存运行证据，不实现中断后的训练恢复，也不把纯权重当作恢复checkpoint。

运行 JSON 是证据文件；当前协调器重启不会自动把磁盘记录恢复到运行列表，界面也没有历史实验浏览入口。不要把磁盘留档理解为恢复执行或完整实验管理。

原 `ml_forge/` 目录与本地审查基线的哈希对照、根 `pyproject.toml` 的有意打包调整，以及本次未执行的验证项目，见 [VERIFICATION.md](VERIFICATION.md)。根目录的旧 `ml-forge` 安装入口与本节 Studio 启动器是两个入口；`pip install -e .` 不会安装 Studio 前端或 `studio_core`。

## 网页实验版的开发验证记录

```powershell
$env:PYTHONPATH = (Resolve-Path '.\studio\backend')
.\studio\.venv\Scripts\python.exe -m unittest discover -s studio/backend/tests -v
Set-Location .\studio\frontend
node --test tests/graph-model.test.mjs tests/editor-session.test.mjs tests/flow-state.test.mjs
npm run build
npm run test:sites
```

2026-11-14的集成验收：后端61项、前端图/编辑会话26项、打包运行时4项测试通过，生产构建通过。实际Chrome操作、截图比较、已修复问题与边界见 [design-qa.md](frontend/design-qa.md)；详细测试及源码基线记录见 [VERIFICATION.md](VERIFICATION.md)。这些结论限于本机S1，不代表所有目标任务或跨机器安装已验收。

