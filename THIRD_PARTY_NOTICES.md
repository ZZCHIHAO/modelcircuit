# ModelCircuit Studio · 模路：第三方来源、许可证与致谢

更新日期：2026-11-19。ModelCircuit Studio · 模路是本工作台的新产品名；`MLForgeStudio` 等源码标识及历史验收名称暂时保留。更名不改变上游、依赖与素材的归属。本文件记录当前源码的上游、技术依赖和设计参考，区分项目贡献与第三方内容。完整项目授权见 [LICENSE](LICENSE)；具体依赖仍适用各自随包提供的许可证及第三方声明。

## 原项目与当前工作的关系

本仓库基于 **[zaina-ml/ml_forge](https://github.com/zaina-ml/ml_forge)** 继续开发。原项目提供可视化拖拽机器学习训练器，是本工作的直接代码基础，而不只是灵感来源。原上游介绍另存于 [ML Forge 1.0.4 README 归档](docs/upstream/README-ML-Forge-1.0.4.md)。根目录 `pyproject.toml` 保留其 `zaina-ml-forge` 包名、`1.0.4` 版本、作者元数据及旧安装入口；这些元数据不代表 Studio 的成熟度或独立发行版本。

原项目采用 [MIT License](https://github.com/zaina-ml/ml_forge/blob/main/LICENSE)。本仓库 [LICENSE](LICENSE) 保留原文及原版权声明：

> Copyright (c) 2026 zaina-ml

`ml_forge/` 保留原业务代码、界面、模板和配套图像资源。2026-11-17 的[验收记录](docs/评审附件/2026-11-17-S4-多模型与硬件工作区验收记录.json)保存了其中 33 份 Python 源文件的 SHA256 基线；该清单只核验所列文件，不代替全部素材的逐项权利审查。保留的原图标、演示图和截图不应被描述为本次开发的原创素材。

`studio/` 是在此项目背景下新增并持续重构的实验工作台，包含独立图核心、芯片/算子与模型目录、静态 Python 桥、实验协调器与 worker、原生桌面编辑器等实质性工作。新增架构和交互并不消除原项目的来源关系，也不表示本仓库所有代码、网络架构、依赖或资产均为原创。

本仓库新增和修改的项目源码及文档同样按根目录 MIT License 提供，已有明确许可的第三方内容继续遵循其原许可。复制或分发项目的全部或实质部分时，应保留原版权声明及 MIT 授权文本；新增贡献不得移除已有归属。

## 模型拓扑与独立对照实现

S4 的 ResNet18/34/50、VGG11/13/16/19 和 SqueezeNet1.0/1.1 结构核查，以官方 **[pytorch/vision 的 `v0.21.0` 标签](https://github.com/pytorch/vision/tree/v0.21.0)** 为固定参考：

| 模型族 | 官方拓扑来源 | 本项目对应位置 |
|---|---|---|
| ResNet | [resnet.py](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py) | `backbone_templates.py`、`tests/resnet_reference.py` |
| VGG | [vgg.py](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vgg.py) | `backbone_templates.py`、`tests/model_family_reference.py` |
| SqueezeNet | [squeezenet.py](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/squeezenet.py) | `backbone_templates.py`、`tests/model_family_reference.py` |

表中相对位置均在 `studio/backend/` 内。Torchvision 使用 [BSD-3-Clause License](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE)，完整原文及版权声明另保存在[本地许可证副本](licenses/torchvision-BSD-3-Clause.txt)。本项目的图工厂和手写 PyTorch 对照实现参照上述官方源码表达网络拓扑；对照实现独立于平台图及导出器，用于减少共同实现错误。“独立”描述测试路径的关系，不表示未参照官方实现，也不是对 ResNet、VGG 或 SqueezeNet 架构原创性的声明。

官方来源的访问日期、固定标签、URL、原始字节 SHA256 和 Git blob SHA，见[验收记录的 `official_topology_sources`](docs/评审附件/2026-11-17-S4-多模型与硬件工作区验收记录.json)。当前模型默认随机初始化，未随项目附带预训练权重；拓扑核查不要求安装 torchvision 或下载权重。VGG 的 GAP512 读出、单通道输入等变体应继续按[使用指南](studio/native/S4_MODELS_WORKSPACE_GUIDE.md)明确标识。

参照、复制或改写官方实现时，应随相关源码与分发包保留对应版权及许可证文本，不能以本项目的 MIT 声明覆盖第三方许可。模型权重、训练数据和论文插图的许可应分别核对。

## 直接依赖与构建工具

下表是当前依赖声明的来源索引，不是完整的软件物料清单。Python 约束见 [pyproject.toml](pyproject.toml)、[Studio 运行依赖](studio/requirements.txt)和[桌面构建依赖](studio/requirements-desktop.txt)；网页实验版的确切解析版本见 [package-lock.json](studio/frontend/package-lock.json)。

| 组件 | 当前用途/声明 | 官方许可证来源 |
|---|---|---|
| PyTorch | Studio 固定 `torch==2.6.0`；旧入口为可选训练依赖 | [BSD-style，含其第三方许可](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE) |
| Torchvision | 旧入口可选训练依赖；S4 以 `v0.21.0` 作拓扑参考 | [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE) |
| Dear PyGui | 原生 GUI；桌面声明 `2.3.1` | [MIT](https://github.com/hoffstadt/DearPyGui/blob/v2.3.1/LICENSE) |
| Pillow | 图像读取/处理；桌面声明 `12.3.0` | [MIT-CMU 及随包第三方声明](https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE) |
| NumPy | 数值处理；桌面声明 `2.3.5` | [BSD-3-Clause 及随包第三方声明](https://github.com/numpy/numpy/blob/v2.3.5/LICENSE.txt) |
| PyInstaller | 桌面打包工具；声明 `6.22.3` | [GPL-2.0-or-later，附分发例外](https://github.com/pyinstaller/pyinstaller/blob/v6.22.3/COPYING.txt) |
| React / React DOM | 网页实验版；锁定 `19.2.0` | [MIT](https://github.com/facebook/react/blob/v19.2.0/LICENSE) |
| React Flow (`@xyflow/react`) | 网页节点画布；锁定 `12.12.1` | [MIT](https://github.com/xyflow/xyflow/blob/main/LICENSE) |
| Phosphor Icons (`@phosphor-icons/react`) | 网页图标；锁定 `2.1.10` | [MIT](https://github.com/phosphor-icons/react/blob/master/LICENSE) |
| Vite | 网页开发与构建；锁定 `6.4.2` | [MIT 及其随包第三方声明](https://github.com/vitejs/vite/blob/v6.4.2/LICENSE) |
| `@vitejs/plugin-react` | 网页 React 构建插件；锁定 `5.0.4` | [MIT](https://github.com/vitejs/vite-plugin-react/blob/main/LICENSE) |

安装包中的传递依赖、二进制库和运行时可能附有额外许可。制作 EXE、ZIP 或网页静态产物时，应根据实际打包内容保留各组件要求的完整 `LICENSE`、`NOTICE` 与版权声明；上表及根目录 MIT License 不能替代这些文件。PyInstaller 的分发例外也不改变被打包依赖自己的授权条件。

## 工作流灵感与名称使用

本工作台将模型视作可连接、可展开、可复用并可测量的工程对象；其概念工作流受到 **NI Multisim、Altium Designer 与 AMD Vivado** 中元件目录、端口连线、层级设计和探针观察方式的启发。这里只致谢交互思路，不声称这些厂商参与、授权、认证或认可本项目。

以上产品名称及其商标属于相应权利人。概念参考不授予商业软件源码、图标、界面截图、字体或其他素材的复制权。第三方商业参考截图不能仅因放入本仓库而适用 MIT；发布文档应使用本项目自己的界面图，或另有明确授权的素材。新增素材必须注明来源及可再分发依据。

## 科研引用与使用范围

使用本软件开展研究或教学时，请记录实际使用的提交号或版本。如果引用 ModelCircuit Studio，可使用 [CITATION.cff](CITATION.cff)。引用上游 ML Forge 的实现、模型架构或第三方素材时，请同时注明其原始来源。

本项目面向非商业科研、教学与个人学习。维护者倡议将本项目用于上述场景，避免用于商业产品、收费服务、付费部署或营利性生产。源码及第三方材料的使用权以各自许可证为准。[MIT License](LICENSE) 允许依其条款进行商业使用。

医学影像功能面向研究和工程验证。使用者应独立验证实验方法和结论，遵守所用数据、权重和素材的相关许可。本项目未获临床诊断用途的验证。

## 后续维护

引入新的依赖、代码片段、模型、权重或素材时，应同步补充来源、确切版本、许可证和修改说明。无法确认再分发依据的内容，先保留在个人参考资料中，待权利明确后再纳入仓库及分发物。发现本文件遗漏或归属错误时，由仓库所有者核对并修订。


## S6 classical machine learning runtime

The table-based SVC and SVR components use scikit-learn and its dependencies. The S6 portable Windows packages include the installed distributions’ license and notice files in their respective `.dist-info` directories.

| Distribution | License documentation |
| --- | --- |
| scikit-learn 1.7.2 | BSD 3-Clause and bundled license notices |
| SciPy 1.15.3 | BSD 3-Clause and binary wheel third-party notices |
| joblib 1.5.2 | BSD 3-Clause |
| threadpoolctl 3.6.0 | BSD 3-Clause |
