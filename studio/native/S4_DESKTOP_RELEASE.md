# 模路 S4 Windows 桌面交付

2026-11-18，ModelCircuit Studio · 模路 `v0.4.0-alpha.1`。此版本号标识已验收的 S4 本地 Windows x64 便携里程碑，已有 EXE 和 ZIP 保留原文件名与构建标识。

| 版本 | 本地启动脚本 | 便携 ZIP 所在目录 | 执行设备 |
|---|---|---|---|
| CPU | [启动模路CPU版.cmd](../启动模路CPU版.cmd) | `studio/dist/s4-cpu/ModelCircuitStudio-Windows-x64-S4-CPU.zip` | CPU |
| CUDA | [启动模路CUDA版.cmd](../启动模路CUDA版.cmd) | `studio/dist/s4-cuda/ModelCircuitStudio-Windows-x64-S4-CUDA.zip` | CPU 或一张兼容的 NVIDIA GPU |

可以直接双击脚本，或解压 ZIP 后运行 `ModelCircuitStudio/ModelCircuitStudio.exe`。保留整个目录和 `_internal`。运行不需要另装 Python 或 Node.js；CUDA 需要兼容的 NVIDIA 驱动。进入右侧「实验」后显式选择 CPU 或 `cuda:0`。

直接启动与脚本启动共用 `%LOCALAPPDATA%\ModelCircuitStudio-S4`，CPU 和 CUDA 可在先后启动时使用同一工作区；避免同时写同一工程。旧 S3 状态及源码预览目录仍保留。已有 Studio 工程可通过「打开 JSON」显式读取，训练前重新检查外部数据路径。历史 `.mlf` 文件是另一种格式。

两种环境各运行 backend 319 项（318 通过，1 项因 Windows symlink 权限跳过）与 native 265 项（全部通过）。两个 EXE 均运行 S4 训练、S4 原生渲染和 S3 配对回归；CUDA EXE 另行以 CPU 完成 S4 训练。每份冻结报告均为退出码 0、通过且 worker 已退出。

S4 的运行诊断采用 6 张程序生成的 PNG：两类 ResNet50 共 23,512,130 参数，执行一轮、一步 AdamW；检查权重更新、独立 checkpoint 推理、内存报告和带/不带 metadata 的 Python 往返。两种冻结界面各记录 22 次应用回调和 4 帧画面，覆盖目录、芯片、分屏、工作区保存恢复和本地芯片实际前向。样本只用于功能验证，不代表科研效果或设备性能。

两 ZIP 的每个条目均核验 CRC 与解压内容，校验值见[可携带验收记录](../../docs/评审附件/2026-11-18-S4-桌面交付验收记录.json)。旧 S3 两个 EXE 和三个 ZIP 的 SHA256 保持不变。包内附项目许可、来源说明和现用运行依赖随安装包提供的许可资料；未做独立的 NVIDIA 再分发合规审计。

本机验收不覆盖洁净 Windows 安装、代码签名、不同驱动或全部显卡。界面验证使用应用自身回调和 framebuffer；未认证物理手势或精确视口恢复。源码分支和便携包均留在本地。本阶段未实现 Transformer、ViT、多任务、分割或多 GPU。
