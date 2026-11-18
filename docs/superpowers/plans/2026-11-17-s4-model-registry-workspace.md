# S4 Model Registry and Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 交付可展开的多模型族与本地芯片扩展，并保存恢复真实桌面工作区。

**Architecture:** 纯 JSON 图工厂通过统一注册层接入现有组件目录；现有模型执行和 Python 桥接保持边界。独立 B4 把有限 WorkbenchState 载荷纳入工作区 v3，不将视图混入模型身份。

**Tech Stack:** Python、DearPyGui、已有 PyTorch CPU/CUDA 环境、unittest。

**Spec:** `docs/superpowers/specs/2026-11-17-s4-model-registry-workspace-design.md`，以及其引用的原 A4/B4 规格/计划。

**2026-11-17追加：** 用户在执行中明确取消人为模型参数上限，改由硬件容量决定执行。本文原M1已冻结为历史候选；H1/M2/N1追加任务覆盖其新模型20M限制，旧v1–v4契约仍作为兼容测试保留。

## Global Constraints

- Windows 原生桌面；禁用 OS/浏览器输入自动化。允许应用自身回调及 framebuffer。
- 不安装依赖、下载权重/数据、长训练；只微型合成和短回归。
- 不改变原 `ml_forge/`、旧 S3 EXE/ZIP 或 Git index；GitHub 暂停，无 HEAD，采用新字节基线，不暂存/提交。
- 旧vision-v4的20M/16,777,216/134,217,728阈值保留兼容；新增hardware-v5三项容量无固定天花板，执行时查询所选设备。明确批次契约仍为1..64，图/解析工作量边界保留。
- 当前源码优先；静态导入不能执行 Python，任意 Python 不承诺无损往返。
- 独立任务审查和整体审查，报告输出/梯度/状态及实际验证范围，不列假可用组件。

## Task M1：多模型族图工厂与统一注册

**Files:** 新建 `studio/backend/studio_core/backbone_templates.py`、`model_registry.py`、`backend/tests/test_backbone_templates.py`、`resnet_reference.py`、其他独立模型参考/注册测试；修改 `component_catalog.py` 的模型条目与构建派发。实现者独占以上文件。

**Interfaces:** 保持 `ComponentBundle(entry, definitions, requirements)`。提供原 A4 的 `make_resnet18(input_shape, *, num_classes=None, feature_outputs=('features',), resource_batch_bound=2)`；其他族采用统一 `build_registered_model(model_id, config)`，稳定 ID、有限 JSON 配置和新鲜描述对象。所有模型通过现有 `build_component` 可调用，层级图可展开，无 Torch/DPG 导入副作用。

- [x] 写并运行缺失模型/非法配置/层级/预算红灯；保留日志。
- [x] 实现纯图工厂及注册描述；真实变体逐一预算检查；VGG 不冒称支持原版大 FC；前向和训练均为现有核心路径。
- [x] 原 A4 独立 ResNet 参考的精确映射、标准参数数、stage 形状、224 eval、32 train 梯度/更新/BN 及 Python 往返通过；其他模型按规格验证。
- [x] 运行 `python -m unittest discover -s studio/backend/tests -p test_backbone_templates.py -v` 及注册/目录/原组件回归，保存退出码/摘要。
- [x] 冻结文件清单/哈希；独立任务审查，修复后范围复审。

## Task W1：B4 工作区与导航

**Files:** 修改 `studio/native/workspace_state.py`、`main.py`、`ui.py`、必要的 `workbench_state.py/workbench_panel.py`；新建 `native/tests/test_workbench_persistence.py`、`test_workbench_locations.py` 和应用自身验收脚本。W1 独占以上文件；不改 catalog_panel。

**Interfaces:** `read_workspace_full(path) -> (document, experiment, workbench_payload)`、`load_saved_workspace_full(paths)`；原二元 API 兼容；`workspace_payload/save_workspace(..., workbench=None)`；`NativeWorkbench(..., *, workbench_payload=None)`。精确字段见原 B4，不增加原生 handles/paths。

- [x] 原 B4 测试先红后绿：v2/旧图兼容，v3 读写，视图净化告警，模型/实验身份不变。
- [x] 接入显式保存/打开/启动恢复/无效表单自动保存和退出回退；保护失败读取的源文件，PermissionError 不清空草稿/测量。
- [x] 已知错误/探针 focus_location，替换使旧事件失效；保存恢复/删定义/undo/同名替换回归。
- [x] 真实回调和默认/最小窗口 framebuffer 证明页签/分屏恢复。执行原 B4 两个定向命令和受影响回归并冻结审查。

## Task E1：桌面目录、本地模型扩展和集成验收

**Files:** 修改 `studio/native/catalog_panel.py`、必要的 `chip_library.py`，新建 `native/tests/test_model_catalog.py`、扩展指南和独立验收脚本；不改 W1 的 UI 文件，必须协调其调用接口。

**Interfaces:** 现 `list_components/build_component` 自动包含 M1 模型；本地芯片沿用现有包读取与闭包/策略校验，不执行外部代码。目录增加视觉主干/模型/本地芯片类别，从实际运行时库路径读取，不让核心注册层访问文件系统。

- [x] 写模型类别/端口/放置和本地有效/损坏/冲突包行为测试红灯，保留输出。
- [x] 现有配置/拖放/策略事务直接复用；大型模型只显示图定义，不在 GUI 创建权重。刷新和错误显示清晰，旧过期 callback 保护不退化。
- [x] 至少一个自定义本地模型包无需改平台核心即可刷新、拖放、展开、前向/Python 往返，明确支持的是架构扩展。
- [x] 整合真实模型与保存重开路径，CPU/实际 CUDA 微型模型证据、完整 backend/native 回归，冻结整体审查材料。
- [x] 更新 HANDOFF/使用指南/计划状态/验收 JSON 与设计 QA，旧包身份和 index 哈希核对；源码预览交付，I2 新 EXE 继续独立后续。

## 任务依赖与门槛

M1 与 W1 可并行，E1 消费 M1 注册描述并通过 W1 已确定公开接口协作。根控制者维护基线和报告；不能同时修改共享文件。审查拒绝后由原作者修复，最终完整回归在源码冻结后一次执行；只有新改动、失败或未决疑点才重跑。

## Task H1：硬件容量执行策略

**Files:** core `schema.py/runtime.py/execution_policy.py/devices.py/classification_runtime.py/python_io.py/vision_python.py/experiment_spec.py`、新`memory_admission.py`和独立测试。hardware_core独占；其修改前8源字节基线在`hardware-core-baseline`。

**Interfaces:** `resource_limits('hardware-v5',batch_bound=...)`返回三个数值容量字段None；有限编译/批次契约仍校验。纯内存估算与运行时`devices.memory_snapshot(explicit_device)`分离；`build_model`及两种训练入口在首次分配前准入一次，显式传递报告避免模型已经分配后重新全量计费。`vision-pytorch-v5`支持静态源码往返，旧profile不改。

- [x] 先红：超过旧参数/张量/累计阈值仍能v5校验/导出，旧profile继续拒绝；同图在不同硬件snapshot下身份一致。
- [x] 实现标准库估算、Windows物理/commit及CUDA选定ordinal可用内存；训练模式分账SGD/AdamW、共享、冻结、buffers、实际执行而非编译账本、采集和CPU副本。
- [x] 证明不足/查询失败在模型、合成输入、checkpoint载入前拒绝，足够容量可执行微型图；估计峰值仅预警，真实OOM清晰报错。
- [x] 有/无metadata的v5 Python、源码变更、guard重绑定拒绝、v1–v4摘要和旧回归定向通过；冻结任务审查。

## Task M2：解除固定阈值后的主流模型

**Files:** M1的9份源和测试；model_families独占，不改H1核心。M1最终9源另存m2 baseline。

**Interfaces:** 原生工厂/ModelRegistry默认新hardware-v5；显式legacyvision-v4仍可验证历史。增加ResNet34、ResNet50和VGG19，保持标准通道和命名状态映射，14项默认catalog不分配权重。

- [x] 先红：新增模型结构/count/输出、无固定上限的资源metadata和新profile，旧策略按原阈值拒绝超界。
- [x] 独立Torch参考及命名映射扩展，ResNet50采用3×3 stride/bottleneck，features2048，层级/代码独立实例契约不变。
- [x] 新增大于20M的代表模型32/batch2输出、梯度、SGD、buffers/状态/Python对照，全部默认项纯图/无metadata往返；冻结审查。

## Task N1：原生v5策略接入与最终集成

**Files:** root独占`native/chip_library.py/component_insert.py/document.py/local_components.py/catalog_panel.py`及相关新测试；W1改动结束后协调UI文案，不并行编辑其文件。

- [x] 先红：保存/读取v5芯片、文档/工作区策略、None容量primitive校验，以及同batch的v4芯片加入v5父图保持父策略；不同batch不能静默兼容。
- [x] 接入14模型类别和真实端口，保留旧有限包/内容身份/原子策略、旧回调隔离；不自动修改实验batch或改用CPU。
- [x] 对新增模型和真实CPU/CUDA容量检查重做最终设备/训练/工作区集成；冻结后跑完整backend/native，独立整轮审查和图像对照。
- [x] 更新最新指南/可携带验收/交接，旧M1/原E1/原W1证据作为历史保留，不把此前20M限制写成当前目标。

## 最终执行状态（2026-11-17）

本计划M1/W1/E1与用户追加H1/M2/N1均完成限定源码实现、定向红绿、任务/范围复审及整轮交叉审查；38份Python修改/新增冻结，完整CPU后端319运行318通过1旧权限skip、原生253通过。实际六CPU/六CUDA微型对照、CPU续训、两设备>20M模型真实worker全路径及最终回调/图像通过。新EXE打包I2为独立后续，本计划未执行。GitHub继续暂停，index/旧程序/归档保持，字节证据留供恢复。
