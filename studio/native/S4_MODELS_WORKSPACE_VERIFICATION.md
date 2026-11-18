# S4 多模型、硬件容量与工作区验证

日期：2026-11-17。限定源码里程碑已通过完整回归、各任务审查和整轮独立交叉审查，无剩余P1/P2；[可携带证据记录](../../docs/评审附件/2026-11-17-S4-多模型与硬件工作区验收记录.json)汇总当前源与实际验证边界。使用步骤见[S4_MODELS_WORKSPACE_GUIDE.md](S4_MODELS_WORKSPACE_GUIDE.md)。S3固定EXE/ZIP保持原样，本轮未打包新EXE。

## 当前实现

14模型入口、36静态组件：ResNet18/34/50主干与标准分类器，VGG11/13/16/19标准非BN卷积特征＋GAP读出，SqueezeNet1_0/1_1主干与标准分类器。全部为可展开JSON芯片，平台默认随机初始化；没有预训练权重。纯图工厂/ModelRegistry与本地JSON包扩展均进入实际目录；受信任注册表可注入桌面构造参数。

hardware-v5不设固定参数/张量/累计执行元素天花板，静态编辑/校验/导出不查询硬件。运行时估算器分别报告状态、梯度、优化器、输入/激活/采集/临时以及CPU暂存/数据/checkpoint/恢复。真实必需下界超过实际所选CPU/GPU容量时提前拒绝；保守投影加余量只预警。旧profile保持兼容；左侧可显式、可撤销地升级工程，加入合法旧芯片不会自动降策略。

工作区v3保存/显式打开/自动与退出/启动恢复覆盖页签、调用路径、逻辑选择、探针端口、布局和焦点；未知/损坏源备份及自动写保护保留，损坏视图可以保留有效模型/实验。模型摘要不包含视图或硬件瞬时观察。已知错误和实际探针有作用域定位。

## 完整回归与设备证据

| 验证 | 实测结果 | 原始记录 |
|---|---|---|
| CPU backend完整套件 | 319运行、318通过、1既有Windows symlink权限跳过；86.415s，exit0 | `output/native-s4-models-workspace/final/backend-cpu-final-full.*` |
| CPU native完整套件 | 253通过；82.327s，exit0 | 同目录 `native-cpu-final-full.*` |
| CPU设备微型对照 | 6组通过，输出/输入梯度/参数梯度/完整状态最大差0 | `integration/hardware-device-cpu.json` |
| 实际cuda:0微型对照 | 6组通过，相同四类最大差0 | `integration/hardware-device-cuda.json` |
| 真实图片短训练与续训 | 12张生成PNG、ResNet18两类、32px/B2、四次SGD；连续2轮与1轮后恢复到2轮的122项状态bit-exact；保留128探针及硬件报告 | `integration/hardware-resnet-training-resume.json` |
| 实际协调器/worker全路径 | CPU、cuda:0各一轮ResNet50两类（23,512,130参数，超过旧20M）、两次SGD、真实参数更新；运行结果、硬件报告、所有产物在清理前核对，子进程关闭 | `integration/hardware-worker-cpu.json` / `hardware-worker-cuda.json` |

上述integration路径均位于`output/native-s4-models-workspace/`。CPU/CUDA六组为ResNet18（7类）、VGG11、SqueezeNet1_1（7类）、ResNet34/50（标准1000类）、VGG19，输入32px/B2，比较独立手写Torch参考和独立Python Model，包含一步SGD及BN buffers。ResNet34/50标准分类器分别21,797,672/25,557,032参数，VGG19卷积特征20,024,384；均实际越过旧20M门槛。

CUDA验收固定初始化seed、TF32关闭、cuDNN deterministic；保持rtol2e-5/atol2e-6。初次普通cuDNN下有输入梯度微差，独立诊断在同一平台模型重复运行中复现，单变量确定性设置消除差异，生产训练未改动。诊断原始报告/trace保留于`integration/cuda-diagnosis/`，没有将容差调宽。GPU默认运行/多输出多损失/续训的位级复现及完整CUDA套件不在此声明。

M2独立审查另外核查全部14默认项纯静态Python有/无metadata往返、9个旧v4闭包黄金哈希、超20M模型32px训练对照及R50/VGG19的224px/B2 eval；全部标准状态名严格映射，无zip猜测。源码编辑结果优先于旧元数据。生成Python v5跟随实际PyTorch分配器，没有复制平台完整估算器。

## 原生证据与审查过程

N1最终9张真实framebuffer、19次注册callback、两次按实际目标node几何读取drag_payload投放（误差0），并保存/重开hardware-v5/v3模型，视图和模型身份相同。W1-fix1另6个单/左右/上下默认/最小布局帧、10callback，以及真实CPU eval测量的探针定位。UI图像采用应用自身framebuffer，未用电脑/OS/浏览器输入控制。

任务审查发现并修复：W1深层损坏视图与指数溢出绕过保护；E1未读超大文件占预算导致好芯片消失；H1把配置批次当必需下界及完成resume变体；N1共用footer与只改策略操作矛盾。原红灯、反例、每轮冻结与范围复审报告完整保留；所有修复对应明确行为断言，不以绿色旧套件掩盖发现。

137份源字节基线、最终155份Python中的38份新增/修改、5256行全diff供整轮独立交叉审查。源及审查清单在`.superpowers/sdd/2026-11-17-s4-model-registry-workspace/`；可携带记录会包含当前源/测试/审查/官方拓扑来源/图像及旧产物哈希。

## 限定范围

内存估算不保证实际峰值或完全阻止OOM；批次仍为明确图契约（1..64），运行为float32单CPU/单GPU。有限文件/节点/解析/递归边界继续保留。无模型/数据/依赖下载、长训练、Git提交/上传或新EXE构建。临时生成数据和checkpoint验证后清理，不把小样本测试指标当科研性能。

现有`ml_forge/`源、S3 EXE与三个ZIP须在最终记录中再次核对；Git HEAD/remote继续不存在，index不得改变。精确pan/zoom、人手手势、minimap精确边界、任意密集布线、多GPU/远程、3D/分割/检测/文本/LLM/RAG和通用第三方Python无损导入没有被此里程碑认证。
