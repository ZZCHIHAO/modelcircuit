# S1 验证记录与交付边界

后续原生DearPyGui桌面阶段已完成：最新后端62项、原生17项测试，以及冻结EXE真实渲染/回调/worker诊断见 [native/VERIFICATION.md](native/VERIFICATION.md)。下面保留早先网页S1的验收轮次，不把其61项历史数量当作当前桌面阶段结果。

## 最终集成验收（主执行者）

日期：2026-11-14。本节在实现与修复后实际执行，下面的“独立文档复核”是另一轮次的只读核查，不能将其未执行事项误读为本节未执行。

| 实际命令 | 结果 | 覆盖 |
|---|---|---|
| `studio/.venv/Scripts/python.exe -m unittest discover -s studio/backend/tests -v`，PYTHONPATH=studio/backend | 61 tests，OK | 核心23、Python桥25、HTTP/进程13 |
| `node --test tests/graph-model.test.mjs tests/editor-session.test.mjs tests/flow-state.test.mjs` | 26 pass，0 fail | 图编辑、封装、接口、选择状态、源码编辑会话 |
| `npm run test:sites` | 4 pass，0 fail | 原前端模板静态分发与API边界 |
| `npm run build` | exit 0 | 生产构建；保留单chunk超过500kB的优化提示 |

总计91项测试通过。没有把截图、构建或纯函数测试当作端到端训练证明。后端测试实际包含独立手写PyTorch参考的forward/backward/update对照；导出代码在独立子进程中与运行器比较输出、梯度、嵌套共享及冻结；受限Python导入读取当前源码并拒绝不支持语法。进程测试覆盖崩溃、取消隔离、JSON非有限值/超限和持久化失败回滚。

真实Chrome验收经用户明确授权使用Playwright CLI。已操作：CPU运行/取消、按具体callId查看探针、共享参数编辑/校验/过期结果、非法Python拒绝/保留草稿/显式重生、Python改8→10并回写画布、下载JSON/.py、JSON重开和reload草稿恢复、多选封装并展开芯片、实际鼠标拖端口/节点以及删除、1100/760/1440宽度及760宽检查器可达。完整步骤、修复前后证据、组合图及限制在 [frontend/design-qa.md](frontend/design-qa.md)。最终页面控制台检查0错误、0警告；非法Python测试中的预期HTTP422单列说明。

### 实际运行证据

- 最后一次成功的默认实验：`studio/runs/319a8e51f9d94f4abf863f9fc5ae3af2.json`，r1，status=succeeded，exit_code=0，CPU、seed42、固定合成批次16样本、20次SGD更新。
- loss_before=0.9259226322174072，loss_after=0.49365317821502686，同批次accuracy=0.9375。58个参数，共享encoder只构造一组权重。它们是本地合成测量，不是独立测试集或临床效果。
- 浏览器真实取消：`studio/runs/aec3beecdfa343b2aff1a923d405192f.json`；界面等待“实验已取消”后恢复运行按钮。
- 实际下载产物：`output/playwright/studio-edited-project.json`、`studio-edited-model.py`、`studio-custom-chip-project.json`（路径均相对仓库根目录）。最后一份保留用户操作生成的双路融合芯片；交付页面恢复默认示例。
- 截图：`output/playwright/studio-final-{root,chip,parameters}.png`、`studio-python.png`、`studio-layout-{1100,760,1440}.png`等。来源参考及最终实现已置入同一组合图并目视核对。

### 尚未验收或未实现

跨机器/干净环境安装、根项目wheel/sdist安装、原DearPyGui程序全回归、触屏与完整无障碍、大规模图性能均未完成。未接入真实医学队列/3D训练、报告生成、LLM/RAG/传统ML后端、远程GPU、完整实验包导出或训练恢复；Python只支持s1-pytorch-v1静态子集。不得将本S1称为全面科研平台完成。

下面的源码基线对照仍有效；CONTRACT的output_metadata和序列化预算现已由主执行者补齐。

## 独立文档复核（原始轮次记录）

核查日期：2026-11-14（Asia/Shanghai）。本记录对应一次文档一致性复核，不能替代其他轮次的后端、前端或浏览器验收报告。仅修改 `studio/README.md` 和本文件；未修改生产代码、安装依赖、启动服务/浏览器或训练。

## 本次实际执行

- 重新读取 `studio/README.md`、`studio/CONTRACT.md`、`docs/superpowers/plans/2026-11-14-studio-s1.md`、`studio/{setup,start,stop}.ps1`、启动 `.cmd`、根 `pyproject.toml`，并核对当前运行器、协调器和 Vite 配置中的对应实现。
- 使用 SHA-256 对照 `docs/评审附件/2026-11-14-源码核查记录.json` 的 `business_files`。该基线内位于 `ml_forge/` 的40个文件全部存在且哈希一致；当前 `ml_forge/` 中排除 `__pycache__` 后未发现该基线之外的文件。
- 对照该附件中全部45个基底文件，发现的唯一内容变化为根 `pyproject.toml`，见下节。
- 实际运行解释器/依赖探测，得到 Studio 虚拟环境 Python `3.12.14`、PyTorch `2.6.0+cpu`、`torch.cuda.is_available() == False`；Node.js 为 `v24.14.1`，前端 Vite 入口文件存在。这只证明这些依赖可导入或文件存在，不等于所有环境与交互已验收。

本次没有运行新的测试套件，不登记或推断新的测试通过总数。README 的开发验证代码块是可执行命令说明，不能作为成功日志引用。

## 原业务源码与打包例外

基线位置：`docs/评审附件/2026-11-14-源码核查记录.json`，键为相对路径，字段为 `business_files[路径].sha256`。这为“原 `ml_forge/` 文件在该基线之后未改变”提供本地字节级证据；不证明上游来源、提交历史或更早版本一致。本目录没有可用于本轮追溯的 Git checkout，不以虚构 commit 代替哈希证据。

根 `pyproject.toml` 的有意例外是新增包发现配置：

```toml
[tool.setuptools.packages.find]
include = ["ml_forge*"]
exclude = ["studio*"]
```

其目的是在仓库新增 `studio/` 后，保持根包只发现旧 `ml_forge` 包，避免把 Studio 作为同一发行包自动发现。旧项目名称、版本、依赖与 `ml-forge` 启动入口保留；Studio 仍通过独立的 setup/start 脚本准备与运行。该调整不属于“原业务 Python 源码未改”的反例，但也不能把整个仓库表述为完全没有变化。

| 文件 | 基线 SHA-256 | 本次对照时 SHA-256 |
|---|---|---|
| `pyproject.toml` | `d839f1c0cf0df1716b7eef8ac2a41b285c8f1a2197b57314fce884b9cf6ef768` | `773a449e687cf8a02527b44cb4312d6dd25c34f7d822ad5f528d138357412710` |

本次没有构建或隔离安装根 wheel/sdist，因而“发现范围已明确”不能升级为“分发产物已安装验证”。

## 支持范围与测量解释

- 当前是 `s1-pytorch-v1` 的局部实现，不等于计划书的全部阶段、后端或任意 Python 导入。计划中的未勾选任务不是测试结果清单；不能仅据该计划宣称完成或未完成。
- 运行器仅支持合成 CPU 二分类任务：二维根输入、命名 `logits` 二类输出、有实际参与 logits 计算的可训练参数。一般图结构有效与演示训练任务可运行是两项检查。
- 同一次运行使用一个固定 seed 的合成批次反复做 SGD 更新。前后 loss、准确率和最终数据探针来自该批次的 `eval()`；history 来自更新前的 `train()`。不设置独立验证/测试集，不据这些数值声称模型科研或临床效果。
- 探针 `values` 和根 `outputs` 是有界预览，不是完整张量传输；根输出另带 `output_metadata`，说明原形状/数量和截断。完整 tensor 仍由模型 forward 返回。
- 本地运行记录保存 graph/config、事件及结果；不提供训练断点恢复或重启后的历史运行列表恢复。取消请求与进程确认终态不同。

`CONTRACT.md` 的数值是 schema 示例而非测量证据；复核时发现其结果示例漏列 `output_metadata` 和独立序列化预算。该差异随后已由主执行者补齐。示例预览为缩写，不是返回完整无界张量的承诺。

## 启动核对与尚未验证

- README 的相对路径命令以仓库根目录为工作目录；双击 `studio/启动工作台.cmd` 则由脚本自身位置定位。start 使用 Studio 自有 `.venv` 与 frontend/node_modules，通过127.0.0.1上的4173/8765端口提供界面/API；需要Node可从PATH找到。
- setup 固定 CPU PyTorch 2.6.0并用 npm ci 安装前端锁定依赖；它会下载依赖，不下载训练数据或模型权重。本次未执行 setup，未验证其他机器、Python版本或网络安装。
- stop按启动器记录的PID与命令身份停止服务，是进程终止操作；运行中需先取消任务并保存项目，不能把停止服务当作保存或恢复训练。
- 本次未启动浏览器、未运行界面操作、未生成截图，也未执行后端测试、前端构建、导出独立运行、故障/并发或训练验收。发现测试或QA脚本、构建文件存在，不构成这些步骤已执行的证据。
- 浏览器QA需另存测试环境、实际步骤、观察结果、截图和遗留问题；如由其他复核轮次完成，应引用其独立记录，不归入本次文档复核。
