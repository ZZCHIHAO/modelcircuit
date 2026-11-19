# 模路开发交接

## 2026-11-18 · S6 CSV 数值回归

`v0.6.0-alpha.2` 交付单个 Windows EXE，双击 `studio/启动模路.cmd`，训练设备在应用内选择。统一的工作区目录为 `%LOCALAPPDATA%\ModelCircuitStudio-S6`。PyTorch 每次训练使用 CPU 或一张 CUDA GPU。

`v0.6.0-alpha.1` 增加第二个任务适配器 `regression-v1`。可在原生工作台加载 CSV 并训练、评估和续训。选择 CPU 或一张 CUDA 卡。数据配置与任务一起保存在工作区，模型通过受限 Python 接口往返。新增分库与 CPU 版 SVC、SVR。入口、运行范围和验收步骤见 [S6 指南](native/S6_REGRESSION_GUIDE.md)及 [S6 验收](native/S6_VERIFICATION.md)。历史 S4/S5 记录如下。

## 2026-11-18：S5 源码阶段验收

`v0.5.0-alpha.2` 已实现并验证可展开 Attention/Encoder/小型 ViT、实际训练、注意力热图、Python 静态往返和分类任务适配。先读 [S5 使用指南](native/S5_TRANSFORMER_GUIDE.md)及 [S5 验收](native/S5_TRANSFORMER_VERIFICATION.md)。S4 EXE 不包含 S5 功能；在现有 CPU/CUDA 环境使用独立源码入口。下一阶段先实现第二个任务适配器，用回归或多头训练验证任务协议的可扩展性。

## 2026-11-18：S5 张量静态契约首个源码预览

`v0.5.0-alpha.1` 已完成计划 T0 的后端静态范围。schema v2 / `tensor-v1` 接受 float32 和 bool 的显式张量契约，旧 schema v1 保持独立路径。未接入 v2 执行、导出和桌面编辑；下一步 T1 新增有限张量算子与参数源。CPU 环境 backend 327 通过 + 1 权限跳过，native 265 通过。便携版继续使用 S4 的 `v0.4.0-alpha.1`；S4 交付已提交私库 PR #1。本节覆盖下方历史状态。

## 2026-11-18：S4 本地便携交付

先读 [S4 桌面交付说明](native/S4_DESKTOP_RELEASE.md)。CPU/CUDA 新包使用 `studio/dist/s4-cpu/` 和 `studio/dist/s4-cuda/`，保留旧 S3 产物。开发分支为 `codex/s4-desktop-release`，基线 `a4a3315`。本轮本地构建和验收，未推送新分支或创建 GitHub Release。此前基线已在个人私库 `ZZCHIHAO/modelcircuit`；下方关于此前无 Git 提交、旧包仍为唯一可用便携版的文字是历史记录。

后续研发从 [S5 设计](../docs/superpowers/specs/2026-11-18-s5-transformer-task-design.md) 和 [S5 计划](../docs/superpowers/plans/2026-11-18-s5-transformer-task.md) 起步。先验证序列张量、显式掩码与可组合注意力，再做可展开 Encoder/小型 ViT 和最小分类任务接口。尚无 S5 生产代码。

> 本页记录早期开发与验收。当前 GitHub 发布和协作方式见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 历史执行位置：多模型、硬件容量与工作区源码已完成

2026-11-17用户继续推进并要求主流模型可扩展，执行中明确“代码理论上不应该有参数上限，只有硬件上限”。当前以补充规格/计划`docs/superpowers/*/2026-11-17-s4-model-registry-workspace*`为准，不能再使用旧20M决策阻挡新模型。新增hardware-v5三项容量为None，编辑/校验/导出不查询硬件，执行前按所选CPU/GPU查询和分账估算；真实最低存储超容量硬拒绝，保守峰值只预警。旧v1–v4兼容/身份/黄金值保留，旧工程有明确、可撤销的硬件策略升级入口。

当前源码包含14个真实模型入口：ResNet18/34/50主干与标准分类器、VGG11/13/16/19标准非BN卷积特征＋GAP、SqueezeNet1_0/1_1主干与标准分类器。总静态目录36项（另17算子5功能芯片），不包含预训练权重或假占位模型。纯图工厂/注册和有限本地JSON包可扩展，受信任ModelRegistry可直接注入NativeWorkbench；模型执行/Python核心不用为每个模型改分支。工作区v3已接入显式/自动/退出/启动保存恢复，以及真实错误/探针位置导航；损坏源保护和旧API继续生效。

详细操作见[native/S4_MODELS_WORKSPACE_GUIDE.md](native/S4_MODELS_WORKSPACE_GUIDE.md)。源码仍用两份S4预览cmd和本机既有CPU/CUDA环境。旧S3 EXE/ZIP尚未重新打包；本轮没有GitHub、stage/commit/remote、依赖安装、模型/数据下载或长训练。

新账本在`.superpowers/sdd/2026-11-17-s4-model-registry-workspace/progress.md`，不要重做前轮A3/B3，也不要重新实施本轮已接受的W1/E1/H1/M2/N1。原137源字节基线、新155源中38份修改/新增及5256行整轮diff已冻结。W1-fix1/E1-fix1/H1-fix1b/M2/N1-fix1及整轮交叉审查均接受，无剩余P1/P2。W1的深view/非有限指数、E1大文件预算、H1虚假批次下界/完成恢复、N1策略说明矛盾均有原红灯和修复/复审证据。

冻结源完整CPU backend319运行（318通过、1旧Windows symlink权限skip）86.415s/exit0，native253通过82.327s/exit0。实际六组CPU和六组cuda:0微型输出/输入梯度/参数梯度/SGD/完整状态/独立Python对照通过，CUDA条件为固定初始化种子、TF32关闭及cuDNN deterministic；原非确定反向输入梯度差异通过同模型重复实验定位，未改生产训练或放宽容差。CPU生成12PNG的ResNet18两轮与1+恢复的122项状态bit-exact，并保留硬件报告。没有完整CUDA套件或CUDA checkpoint位级恢复/性能声明。

当前最终原生图像：N1-fix1九帧/19callbacks/两精确投放与hardware-v5/v3重开；W1-fix1六布局帧/10callbacks及额外真实CPU探针帧。使用应用自身回调/framebuffer，未用电脑控制；精确pan/zoom、人手手势和小分屏所有节点完整可见均不保证。23,512,130参数ResNet50两类还分别通过实际CPU/cuda:0的RunManager→worker→运行记录与产物全路径，一轮两SGD步，子进程全部关闭。下一任务从I2新CPU/CUDA便携包开始，不要把旧EXE当作当前源码。本轮源关口关闭，详见[可携带验收记录](../docs/评审附件/2026-11-17-S4-多模型与硬件工作区验收记录.json)和[验证说明](native/S4_MODELS_WORKSPACE_VERIFICATION.md)。

最后更新：2026-11-17。项目根目录：`.`。

## 当前执行位置：A3/B3、基础目录与直角连线源码已完成

2026-11-17用户“推进下一步”，并追加可用性优先、参考AD/电路软件补基础库、组件直接拖出及直线/转折和左右精确端口。本轮完成五类纯图工厂、17primitive+5功能芯片可检索/配置/拖放/点击的目录与原子插入、可见原生页签/单左右上下布局/独立侧栏，以及完成连线的正交绘制/命中/删除和真实输入左输出右。A3、目录修复复审、正交helper、B3四项P2修复复审及整轮交叉审查均接受，本轮限定源码任务关闭。

先读本轮[使用与验证说明](native/S4_COMPONENT_WORKBENCH_VERIFICATION.md)及SDD账本最新末尾，不重做A1/A2/B1/B2。119份Python源基线在a3-b3-baseline；本轮各brief/report、原始红绿和冻结审查清单保留。最终完整backend275运行（274通过1旧权限skip）/29.037s，native199通过/63.329s，退出0；source25份修改/新增的最终清单为a3-b3-final-source.json，与整体审查3773行diff对应。CPU/cuda:0五工厂微型输出/梯度/一步SGD/BN buffers与去metadata独立Python比对通过，不宣称完整CUDA/科研性能。

可双击`studio/启动S4源码预览.cmd`或`studio/启动S4-CUDA源码预览.cmd`使用本机已有pythonw环境和独立LOCALAPPDATA预览状态；两个小启动器仅做路径/内容核对，未用OS自动化双击验收。旧固定S3启动器/包未变。最终界面18frame在b3/review-fix1-*目录，原18张/170/192/193候选证据不覆盖；4P2修复49项定向及范围复审通过，同尺寸root/端口/弹窗/小窗口对照QA passed，整体终审a3-b3-final-review-report接受，无新增P1/P2。最终可携带[验收记录](../docs/评审附件/2026-11-17-S4-组件工作台验收记录.json)包含源、测试、设备、审查、截图、文档及旧产物身份。

源目录及固定EXE保护、原生桌面/禁电脑控制、GitHub暂停/无HEAD或remote/不暂存提交、不安装依赖下载模型数据和长训练继续生效。固定CPU/CUDA EXE仍S3，本轮源码预览命令在新验证说明。A4生产ResNet18、B4视图持久化、剩余本地目录索引/集成与I2正式包仍后续，不能把可用目录切片称作完整S4。

下轮从A4与B4、剩余A5/I1恢复，不能重做已经接受的A3/B3及基础目录/直角线。物理手势、非零pan/zoom精确恢复、minimap精确矩形、任意密集布线保证、完整CUDA套件、新EXE未认证；临时拖线预览保持可见原生样式，完成连线为直角。去metadata不保证已删组件版本；任意动态调用独立Scale构造器类型校验边界在文档中保留。

## 历史执行位置：A2+B2限定源码已完成

2026-11-17用户授权“进行下一步”，本轮完成A2新资源策略/受控Python v4、native芯片策略桥接和B2真实常驻CanvasHost/事件归属。A2与B2各自两轮修复复审、桥接修复复审及整体交叉审查后的唯一修复波次/范围复审均通过；本轮已关闭。先读[策略与画布验证](native/S4_POLICY_CANVAS_VERIFICATION.md)，可携带[验收记录](../docs/评审附件/2026-11-17-S4-A2-B2验收记录.json)含测试/源码/设备/审查与旧产物身份；完整本地账本在`.superpowers/sdd/2026-11-15-s4-workbench-components/progress.md`，不要重做已完成任务。

最终源码CPU backend251项运行（250通过、1旧symlink权限跳过，29.590s）、native141项通过（61.875s），退出0。CPU与实际cuda:0的2×3×4×4生成张量验证独立Python/平台输出、梯度、一步SGD、BN buffers、严格state重放及batch上界；不宣称本轮完整CUDA套件、CUDA checkpoint位级恢复或新冻结EXE验收。

新vision-v4要求显式严格int1..64的resource_batch_bound，根/各定义执行预算与有界编译工作分账；不同真实call不能因Tensor对象相同而合并。v4源码显式保留None/classification-v2/vision-v4策略，当前源码优先；vision必须保留精确根guard/helper，活跃符号重绑定静态拒绝，旧v1–v3合法名称兼容。独立Python未复制平台完整资源账本。v2芯片包要求显式匹配策略；workspace仍v2，只保留模型策略，尚无视图持久化。

整体审查的两P2已合并修复：根forward局部结果/别名不能遮蔽活跃guard；仅v4身份将显式null资源策略与缺省等价，保留旧黄金摘要。20项Python定向及最终251/141完整回归通过，最终范围复审接受，无该范围遗留P1/P2。初审3062行/22文件证据保留，最终代码另记录于a2-b2-final-fixed-review（3130行/22文件）；唯一修复波次95行/3文件可单独复核。

B2每view独立常驻容器/editor，生产回调携带view/generation/context/document令牌；上下文/重绘/整体替换及关闭重开令牌失效。Undo移除当前定义、有效非焦点延迟事件、空闲轮询保留未提交输入均有真实回调/控件回归。逻辑选择用高亮，精确pan/zoom及正向原生选择位未验，不宣称真实OS鼠标手势。新算子端口可渲染，左侧旧palette尚未换成新目录。

下一未实施项：A3功能芯片工厂与B3正式页签/分屏/侧栏；之后A4 ResNet18、A5/I1目录、B4工作区v3、I2源码与CPU/CUDA新EXE交付。D1和D3核心关口已过，A4真实ResNet闭包与参考数值关口仍待做。没有新增ResNet/融合目录菜单或正式页签条，不把22份修改/新增Python文件的本轮工作标成完整S4。

固定EXE/启动器仍为S3，旧程序和CPU/CUDA/源码ZIP整体SHA256均匹配历史记录；原ml_forge33份Python源码与本轮110份既有源基线一致。源码预览命令在本轮验证文档中，使用现有独立环境和output独立状态目录。GitHub暂停，无HEAD/remote，不暂存/提交，不询问merge/push菜单；保留SDD基线供后续恢复。

## 历史执行位置：2026-11-16 A1+B1

2026-11-16用户再次“下一步”后，完成A1算子核心与B1纯视图状态/作用域，当前先读[基础核心验证](native/S4_FOUNDATION_VERIFICATION.md)。CPU完整backend219项（218通过、1 symlink权限跳过）、native108项通过且退出0；CPU/cuda:0六算子微型独立前向/梯度/SGD/state_dict检查通过。P1 BN持久buffer分配预算、P2大整数Scale、P2歧义选择/探针已修复并独立复审；最终交叉审查接受。

新增算子为BN2d/Identity/Sigmoid/Multiply/Scale/OneMinus；旧11项保留。BN四组合、冻结/共享/独立实例与buffers规则有验证。WorkbenchState/view_scope无GUI依赖；不会制造实际页签，也不承诺原生选择或视口恢复。Document.add新kind在原生尚未接线时明确ValueError；Python导出仍拒绝六种新kind，A2 v4未做。

下一未实施项：A2预算/Python契约、B2 CanvasHost/事件路由。A3/A4/A5、B3/B4、I1/I2仍后续。正式EXE/ZIP仍S3，本轮未打包；原ml_forge、ui.py和python_io.py未改，历史记录。可携带记录：`docs/评审附件/2026-11-16-S4-A1-B1验收记录.json`。D0/原规划段落是历史追记，不覆盖本段。

## 最新执行追记：2026-11-16 D0调查完成，预算Go关口未过

用户已授权“推进下一步任务”，本轮执行了S4的A0/B0，不再是仅规划。当前正式软件仍S3，没有新的组件菜单、多页签主程序或EXE。入口：[S4 D0验证与交接](native/S4_D0_VERIFICATION.md)；完整账本、API/原生截图、日志在 `output/native-s4-d0/`，可携带记录为 `docs/评审附件/2026-11-16-S4-D0验收记录.json`。

A0新增17项、B0修复后6项定向通过；CPU完整backend运行192项，日志OK（1项symlink权限跳过）；完整native73项通过。backend包裹脚本在测试结束后解码Windows日志失败，子进程退出码未保存，未冒称捕获退出0；原始unittest摘要已核对。B0清理失败误记成功的P2已修复并范围复审。既有回归中的微型合成/CSV/PNG短训练单列，不作为新ResNet能力；无长训练、权重/数据下载或依赖安装。

双editor限定验证可行，但精确pan/zoom、原生正选择与真实鼠标手势未验。候选17定义ResNet18按现计费的条件累计batch1/2为148,151,712 / 296,303,424，均超134,217,728；独立复算接受，**只改batch_bound不可行**。后续可先做独立A1算子及B1视图状态；A2必须先重新明确仅新profile的编译/执行预算口径，不能抬旧档全局阈值、缩窄网络或合并真实调用。S3源核心/旧文件、EXE和ZIP保留，历史记录。

## 已规划的下一阶段：S4（尚未实施）

用户指出基础/成熟组件库不足、主模型与芯片内部无法多页签切换，并要求参考Vivado层级工作台。本轮用户明确选择“本轮只交付计划”，因此S3仍是当前软件交付，不将S4规格当作已实现功能。

下一阶段总入口：[S4实施总计划](../docs/superpowers/plans/2026-11-15-s4-workbench-components.md)，配套[设计规格](../docs/superpowers/specs/2026-11-15-s4-workbench-components-design.md)、[组件库子计划](../docs/superpowers/plans/2026-11-15-s4a-component-library.md)、[多页签子计划](../docs/superpowers/plans/2026-11-15-s4b-tabbed-workbench.md)。优先完成D0能力/预算验证，再并行核心组件与视图状态，最终顺序集成共享文件及实际CPU/CUDA EXE验收。

S4计划包含统一目录/中文检索、首批融合芯片及可展开随机初始化ResNet18、单Document多页签/双分屏、v3工作区恢复、资源/Python兼容。精确pan/zoom跨会话恢复没有公开API证据，不承诺；ResNet34/预训练权重/训练关系图等仍是后续。S3旧包和验收记录不因新增规划重打包或改写身份。GitHub上传继续暂停。

## 用户要求与当前范围

- 保持原 ML Forge 的 **DearPyGui 原生桌面形式**，交付可双击 Windows EXE；React/Vite 网页实现保留作早期 S1 参考。
- 用户已批准推进 S3，本轮增加具名多输出芯片、接口/公开参数/绑定编辑、有限语义撤销/重做、版本化本地芯片库与图片/表格配对分类。数据仍使用 `classification-v2`，保留 S2 CSV/ImageFolder、train/val/test、checkpoint 和 S1 合成演示。S3 源码及 CPU/CUDA EXE 已通过本机真实训练、恢复、Python 往返和原生回调验收。长期的层级 AI 科研平台目标没有因本轮分类扩展而全部完成。
- **调试不使用 Use Computer、OS 或浏览器输入自动化。** 使用源码测试、程序自身 DPG 回调/渲染、独立 worker 和冻结诊断；不要恢复电脑控制调试。
- 用户授权的独立 CUDA 环境已安装，`torch 2.6.0+cu124` / CUDA 12.4 / cuDNN 90100；S3 配对芯片训练已在 RTX 4060 Laptop 的源码与冻结 EXE 中实际完成，独立 best 推理、续训、Python 往返、显式选卡和 DPG 回调均通过。CUDA EXE 显式选 CPU 的同一路径也通过。CUDA 入口 `studio/启动CUDA桌面版.cmd`，EXE 在 `studio/dist/desktop-cuda/MLForgeStudio/`。CPU `2.6.0+cpu` 环境与包保持独立。多 GPU、DDP 与远程训练未实现；S3 证据见 [native/S3_VERIFICATION.md](native/S3_VERIFICATION.md)，S2 历史保留。
- 本轮没有 DICOM/NIfTI、3D、分割、LLM/RAG 或完整临床科研流程承诺；本地 CSV/PNG 测试 fixture 不代表临床性能验证。
- 用户明确要求 **暂缓 GitHub 上传，稍后另开对话继续**。当前无提交、无 remote、无协作者；不要自动重试、创建定时任务、公开发布或添加协作者。

## 实现与操作要点

- 原生入口 `studio/desktop_entry.py`；桌面 `studio/native`；数据/训练位于 `studio/backend/studio_core/{classification_data,paired_data,classification_runtime}.py`，根配置/设备模块为 `experiment_spec.py`、`devices.py`。模型执行与受限 Python 往返继续共用独立核心，协调器为 `server.RunManager`。
- 右侧“实验”选择 CSV、ImageFolder 或“配对图片与表格分类”，打开“数据设置”，配置来源/标签/特征/划分后“检查并冻结数据摘要”；检查在后台执行。单模态任务要求一个根输入；配对任务映射 image/tabular 两路到两个不同根输入。所有任务均要求匹配类别数的命名 logits。
- 从检查结果生成 MLP、CNN 或配对 CNN+clinical MLP→Concat→logits 时必须确认替换当前模型；不会把真实数据隐式复制到旧默认双输入图。数据、实验或模型语义变化使原结果过期，节点移动不影响测量身份。
- CSV 标准化只拟合 train；同一患者多行时设置稳定 `group_column`，显式 split 同样不得让组跨 train/val/test。ImageFolder 没有患者分组字段，多张患者图片应先在外部按患者拆成 `train/val/test` 目录；内容重复检测不能代替患者隔离。
- 配对 CSV 默认必填列角色为 `sample_id`、`patient_id`、`image_path`，另选标签和有限数值特征。每行绑定图片/数值/标签/组，ID 唯一，患者组必选；显式 split 或 group-aware 划分同时作用两路，联合 shuffle 保持样本对应。image_root 空则 CSV 父目录，拒越界/缺模态/坏图/跨 split 患者或图片内容；数值重复本身允许。`input_ports` 默认 `{image:'image',tabular:'clinical'}`，映射精确覆盖两个根输入并绑定恢复身份。
- “接口与芯片”的接口页操作当前画布作用域；定义选择器针对公开参数/库目标。可增删/重命名端口、改 shape、连接输出；被消费接口先断开再删。named 多输出使用 `call:id:port`，单输出裸引用兼容，禁止多输出隐选第一项，旧唯一输出扩展会明确迁移引用。
- 公开参数行包含类型/默认值/局部 `instance.parameter` 绑定，非法路径/类型/悬空绑定原子拒绝；尺寸变化仍可留下可见下游校验错误。Python 具名 profile 为 `ports-pytorch-v3`，当前源码决定实际端口，metadata 不覆盖语义。
- 语义 undo/redo 总计最多32状态、32 MiB；不包含移动/选择/导航，不持久化历史，revision 单调增加。封装支持当前作用域、多输出、内部共享；保留原图并新增独立实例，不自动替换/重接，不捕获跨边界权重共享。
- 本地 `chips/<id>__<version>.mlfchip.json` 保存架构、接口、公开默认值/绑定与依赖闭包，不含权重。相同 ID/版本不能覆盖不同内容；导入 ID 冲突一致改名，完整性/限额错误保持项目和历史不变。
- 每轮训练所有 train minibatch，单独评估 val，仅按 val loss 选 best；完成后加载最佳 checkpoint，最终评估 held-out test。探针来自实际 val 批次；梯度范数来自选定 epoch 的最后训练 minibatch，更新范数比较初始化与选定 checkpoint。
- 模型增加 Conv2d、MaxPool2d、AdaptiveAvgPool2d、Flatten；CNN 图使用明确资源 profile，S1 上限保留。Python 仅静态解析受支持构造器/显式 forward，导入只替换模型并保留实验设置，不执行任意代码。
- 显式 JSON 与自动 `workspace.json` 使用 `{workspace_version:2,model,experiment}`；启动优先恢复 workspace，并兼容旧纯图 JSON / `draft.json`。恢复设置后真实运行需要重新检查文件内容。
- 每个真实运行目录含 `manifest.json`、`run_spec.json`、`last.pt`、`best.pt`、指标/参数文件，成功后有 `predictions.csv`。选择 `last.pt` 并保持图、数据、预处理、优化器、lr、batch、seed 及配对端口映射一致可继续；epochs 是总目标轮数。取消保留最后完整 epoch，第一轮前取消可能没有 checkpoint；不自动恢复未完成进程或提供完整历史实验列表。

## 本地路径与归档

- EXE 固定输出：`studio/dist/desktop/MLForgeStudio/MLForgeStudio.exe`；启动器 `studio/启动桌面版.cmd`。CUDA 入口仍为 `studio/启动CUDA桌面版.cmd` / `studio/dist/desktop-cuda/MLForgeStudio/MLForgeStudio.exe`。分发均带 `_internal`；两个固定路径现为已通过实际冻结验收的 S3 EXE。
- **保留 S1 历史归档**：`studio/dist/desktop/MLForgeStudio-Windows-x64-S1.zip`、`MLForgeStudio-source-S1.zip`。
- **保留 S2 历史归档**：同目录 `MLForgeStudio-Windows-x64-S2-CPU.zip`、`MLForgeStudio-Windows-x64-S2-CUDA.zip`、`MLForgeStudio-source-S2.zip`。内容身份与 CRC/SHA256 以 S2 验证记录为准；源码快照排除依赖、运行日志和构建产物。
- **S3 已生成归档**：`studio/dist/desktop/MLForgeStudio-Windows-x64-S3-CPU.zip`、`MLForgeStudio-Windows-x64-S3-CUDA.zip`、`MLForgeStudio-source-S3.zip`；三个 ZIP CRC 通过，便携包整体哈希和 EXE 内容匹配，源码包逐项匹配当前文件。CPU/CUDA 身份由 S3 记录确认，最终源码包身份在本地 `output/native-s3/archive-source.json`。
- 指南：`studio/README.md`、`studio/native/README.md`。当前结论入口是 [native/S3_VERIFICATION.md](native/S3_VERIFICATION.md)；源码包内的[验收记录](../docs/评审附件/2026-11-15-S3验收记录.json)包含10份原始报告的状态/哈希、64份 Python 源文件哈希、程序/便携包身份和设备边界。原生界面见[同图对照](../docs/images/native-s3-comparison.png)。S2 历史证据保留在 [native/S2_VERIFICATION.md](native/S2_VERIFICATION.md)。
- S1 历史源码/冻结验证保留在 [native/VERIFICATION.md](native/VERIFICATION.md)、[VERIFICATION.md](VERIFICATION.md) 与 `native/design-qa.md`；其中测试数量、截图和构建结论只证明当时 S1。
- 默认用户状态：`%LOCALAPPDATA%\MLForgeStudio\` 下 `workspace.json`、兼容图草稿 `draft.json`、`runs/<run_id>.json`、`runs/<run_id>/` 、架构库 `chips/` 和 `logs/desktop.log`。QA 状态在 `output/`，勿上传；可用 `--data-dir` 指定另一可写目录。
- 构建文件：`studio/build-desktop.ps1`、`studio/requirements-desktop.txt`、`studio/native/MLForgeStudio.spec`。环境包括 CPU torch 2.6.0、DPG 2.3.1、PyInstaller 6.22.3、Pillow、NumPy；便携包用户无需另装 Python/Node。
- 原 `ml_forge/` 业务源码继续保留；根 `pyproject.toml` 的包发现 include `ml_forge*` / exclude `studio*`；旧 `ml-forge` 和 Studio 是不同入口。旧哈希与打包调整证据保留于 S1 验证文档。

## 验证与待续边界

S3 源码测试与独立审查覆盖共享/嵌套具名输出、手写前向/梯度/更新、metadata-free Python、接口引用迁移、public 绑定类型/删除保护、有限历史、版本化库闭包/冲突/限额，以及配对图像数值对齐、患者/图片泄漏、路径根限制、文件变化、联合 shuffle、恢复身份与 CPU 精确续训。每环境 backend 运行175项（174通过，1项因 Windows symlink 权限跳过），native 67项全部通过；10份源码/冻结运行及渲染报告均 passed、退出码0、workers_stopped=true。CPU/CUDA EXE 实际训练通过，CUDA EXE 内 CPU 路径及新 CPU EXE 的 S1 训练/取消回归也通过。详见 [native/S3_VERIFICATION.md](native/S3_VERIFICATION.md)。

S2 CPU/CUDA 的既有记录与 ZIP 保留；CPU 包不可 CUDA，CUDA 包单设备可选 cpu/cuda:N。S3 CPU 续训的模型/优化器/RNG/shuffle/历史/指标逐位一致已验证，CUDA 验证连续性与推理容差，不承诺跨设备或 CUDA 位级一致。未编辑 Python 导回保持测量有效；工作区重开仍需重新检查外部数据。上述均非干净 Windows/全部显卡、代码签名或安装器认证。

下一路线仍包括多 loss/可配置训练协议图、交互式前向/反向/优化器单步与完整训练工程导出；本切片不声称完成这些，也不包含3D/分割、LLM/RAG、多 GPU、DDP 或远程执行。
