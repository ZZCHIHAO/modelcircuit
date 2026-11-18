# ModelCircuit Studio · 模路：贡献与维护指南

每个验收完成的开发阶段使用独立 PR。同步更新根目录 `VERSION` 和 `CHANGELOG.md`，PR 标题带 `v` 前缀版本号，正文说明具体变化、验证结果、兼容性和未覆盖范围。开发阶段使用 `0.x.y-alpha.N` 等预览版本；`v1.0.0` 留给完整产品通过验收后。产品版本不修改上游 `pyproject.toml` 包版本；开 PR 不等于合并、打标签或发布安装包。

ModelCircuit Studio · 模路处于 **pre-alpha 源码研发阶段**。这是工作台的新产品名，`MLForgeStudio` 等源码标识、启动脚本和历史验收名称暂时保留；更名不改变既有来源与版权声明。当前主要工作对象是 `studio/` 原生桌面工作台，原 `ml_forge/` 作为上游保留入口继续存在。功能范围与路线图见 [README](README.md)，具体行为以[S4 使用指南](studio/native/S4_MODELS_WORKSPACE_GUIDE.md)、[验证说明](studio/native/S4_MODELS_WORKSPACE_VERIFICATION.md)及对应测试为准。

## 仓库访问与协作方式

当前仓库是个人私有仓库，仅仓库所有者访问，不开放协作者邀请。本指南用于约束项目维护，以及将来由所有者明确开放的协作；阅读本文件、提出建议或提交修改不会自动获得仓库访问权或合并权限。

GitHub 对个人仓库提供所有者与协作者两类权限；协作者可向个人仓库写入，因此邀请协作者不等于只授予只读权限。若未来需要“能阅读、可提交建议、由维护者决定合并”的多人流程，可另行评估组织仓库的 Read 角色与私有 fork / Pull Request。此方案取决于套餐及仓库、组织、企业的 fork 策略，Read 权限本身不保证私有 fork 可用；仅在所有者完成配置并明确授权后采用。依据见 [个人仓库权限](https://docs.github.com/en/account-and-profile/reference/permission-levels-for-a-personal-account-repository)、[组织仓库角色](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization)及[私有 fork 策略](https://docs.github.com/en/organizations/managing-organization-settings/managing-the-forking-policy-for-your-organization)。

仓库可见性决定谁能从 GitHub 获取内容，项目许可证规定取得副本后的使用权。MIT 允许接收者依其条款修改自己的副本；访问控制不能保证他人无法修改本地代码。对本仓库受保护分支的修改，仍由仓库所有者决定。

## 提出问题或修改建议

通过与所有者事先约定的渠道反馈；仅在获得仓库访问权且相应功能开放后使用 Issues 或 Pull Requests。报告问题时提供：

1. 所用源码版本或提交号，以及操作系统、Python、PyTorch 和相关依赖版本。
2. 最短复现步骤，预期行为与实际行为。
3. 去除私密信息的错误日志、最小图 JSON 或生成示例。
4. 涉及训练时的数据适配器、张量形状、批次契约和设备信息。

不要把真实受试者数据、私有数据集、访问令牌、本机虚拟环境、运行 checkpoint 或无授权素材加入报告或仓库。功能建议应说明具体使用场景、现有工作流受阻的位置，以及可验证的成功条件。

## 代码结构与扩展边界

| 路径 | 维护重点 |
|---|---|
| `studio/backend/studio_core/` | 图结构、校验、运行、模型目录、静态 Python 桥及实验任务 |
| `studio/native/` | Dear PyGui 原生桌面、文档命令、工作区、本地芯片库及交互 |
| `studio/backend/tests/`、`studio/native/tests/` | 核心回归、数值对照与原生行为检查 |
| `studio/frontend/` | 早期网页实验版及其前端验证 |
| `ml_forge/` | 上游保留实现；修改时单独说明兼容性影响与来源 |
| `docs/` | 项目说明、设计与可携带验收证据 |

模型扩展优先使用受支持算子组成的有限 JSON 芯片包。需要新增目录项时，可编写返回 `ComponentBundle` 的纯图工厂，用 `ModelSpec` 声明端口、配置与能力，通过 `DEFAULT_MODEL_REGISTRY.with_model(spec, factory)` 创建扩展注册表，再传入目录 API 或 `NativeWorkbench(..., model_registry=registry)`。这是受信任的本地源码扩展接口；本地芯片包不夹带权重，也不自动发现或执行任意 Python 插件。

新增算子须同时完成配置/形状校验、运行语义、独立 Python 导出与静态导入，并验证这些路径的一致性。共享参数必须显式区分实例与调用，具名端口不得靠默认猜测接线。未知配置或不支持的结构应清楚报错，不能静默替换算法。

Python 导入只静态解析支持的构造器及显式 `forward`，不执行导入代码；平台生成源码在限定 profile 内支持往返。任意第三方 Python、动态控制流或动态端口选择不保证可导入，更不保证无损还原。结构、布局和 checkpoint 身份也应分别处理。

当前实际任务以 2D 图像/表格及配对输入分类为主，执行为 float32 单 CPU 或单张选定 GPU。增加目录名称不能代替任务后端；Transformer、ViT、LLM、多 GPU、分割、检测、3D 与临床生产能力均不可在尚未实现和验证时宣称支持。内存估算用于执行前检查，不能承诺完全避免 OOM。

## 本地开发与验证

先按 [Studio 说明](studio/README.md)准备自己的开发环境。`studio/启动S4源码预览.cmd` 使用已有的 `studio/.venv`，不会安装依赖；CUDA 源码预览使用独立环境。根目录 `pip install -e .` 安装的是上游入口，不会安装 Studio 前端或 `studio_core`。

在仓库根目录运行以下 PowerShell 命令检查当前原生源码；如使用其他虚拟环境，替换 Python 路径：

```powershell
$env:PYTHONPATH = @(
    (Resolve-Path '.\studio\backend').Path,
    (Resolve-Path '.\studio').Path
) -join [IO.Path]::PathSeparator

.\studio\.venv\Scripts\python.exe -m unittest discover -s studio/backend/tests -v
.\studio\.venv\Scripts\python.exe -m unittest discover -s studio/native/tests -v
```

网页实验版变更还需在 `studio/frontend/` 运行：

```powershell
node --test tests/graph-model.test.mjs tests/editor-session.test.mjs tests/flow-state.test.mjs
npm run test:sites
npm run build
```

根据改动补充能复现问题并验证实际行为的检查。数值路径应使用独立参考核对输出、梯度、状态或训练恢复；图/代码路径检查有无 metadata 的往返；GUI 变更检查最小窗口、错误状态与实际回调。仅对影响的路径执行合理验证，不把截图或通过数量当成科研有效性证明。

2026-11-17 的[源码验收记录](docs/评审附件/2026-11-17-S4-多模型与硬件工作区验收记录.json)记录 backend **318 通过、1 跳过**，native **253 通过**，合计 **571 通过、1 跳过**。跳过项为 Windows symlink 权限限制。这是该日期、该源码和该环境的历史证据，不能作为后续提交已通过的结论；提交者应提供自己实际执行的命令、结果和未验证范围。

## 提交、审查与许可证

在所有者明确开放协作后，先就较大的范围变化达成约定，再使用独立分支提交聚焦的修改。变更说明应交代问题、修改后的行为、验证证据、兼容性或迁移影响；小修订可简短说明。是否合并以及何时发布，由仓库所有者审查决定。

不得在未核查来源时把第三方代码、图标、商业界面截图、权重或数据加入提交。引入依赖和模型时同步更新 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，保留原作者声明与对应许可证。不得删改根目录 [LICENSE](LICENSE) 中的原版权声明 `Copyright (c) 2026 zaina-ml`。

向本项目提交并获接纳的新增/修改源码及文档，按本项目 MIT License 提供，另有明确第三方许可的内容除外。提交者应有权提供相应内容，并说明其中的第三方来源；本指南不要求额外转让版权，也不将第三方材料自动改为 MIT。
