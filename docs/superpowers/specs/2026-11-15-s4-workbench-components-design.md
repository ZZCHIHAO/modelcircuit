# S4 组件库与多页签层级工作台设计规格

日期：2026-11-15。状态：规划交付，尚未实施。用户已明确选择“本轮只交付计划”。本文中的新文件、接口、格式、测试和交付物均是设计，不代表已实现。S3 已发布的本地 EXE、ZIP 和验收记录保留为历史快照。

实施追记2026-11-16：用户已授权推进下一任务。D0独立诊断/算例与回归已执行，正式S4组件与工作台尚未实施；候选预算出现阻断项，见第4.4节及 `studio/native/S4_D0_VERIFICATION.md`。上段保留原规划轮状态，不作为新授权之后的禁令。

同日下一轮追记：A1算子计算核心与B1纯视图状态/作用域已完成源码实现、回归和独立审查；不代表目录/多页签UI或S4 EXE完成。详情见 `studio/native/S4_FOUNDATION_VERIFICATION.md`。A2新profile预算/Python v4和B2–B4仍待实施。

2026-11-17追记：A2资源/Python契约、芯片策略桥接和B2真实常驻画布/事件路由完成各自独立范围复审及完整源码回归，整轮交叉审查与交接记录见`studio/native/S4_POLICY_CANVAS_VERIFICATION.md`。正式目录、ResNet18、页签控件、视图持久化及新EXE仍未交付；前文日期状态为历史，不覆盖本追记。

## 1. 问题、目标和依据

目标：用户能从清晰的组件目录添加基础层、融合模块和可展开的 ResNet18，在始终打开的主模型页签与芯片页签间切换，并通过双分屏对照编辑。所有视图编辑同一工程，保持权重身份、Python 往返和当前单设备分类流程。

已核实事实（以下为2026-11-15规划基线，当前实现以执行追记及对应验证记录为准）：

| 证据 | 当前行为 | 对设计的约束 |
|---|---|---|
| `studio/native/ui.py:30,351`，`_search_library` | 11种基础算子，按英文 kind 过滤；项目 definitions 另列；无成熟网络目录 | 改成可检索、有类别和能力说明的目录，不能只增加几个按钮 |
| `studio/native/ui.py:173`，`_build` | 项目树、搜索框和组件列表共用滚动容器 | 分离项目树与目录；搜索固定，结果独立滚动 |
| `studio/native/ui.py:373,604`，`_render_graph/_enter_root` | 单一 `native_editor`，导航后重绘 | 多视图必须拆出独立渲染上下文和事件归属 |
| `studio/native/document.py:166,213,234` | 单 scope/trail/selection；探针由具体调用路径组成 | 图文档共享，视图状态独立；定义与调用位置不能混同 |
| `studio/native/workspace_state.py:101,110` | workspace_version=2，严格限定 model/experiment | 新 UI 状态用显式 v3；继续读取旧格式 |
| `studio/backend/studio_core/schema.py:48,117` | classification-v2 的上限为20M参数、16,777,216单张量元素、134,217,728累计元素；null按64估算 | 新 batch 上界必须贯通静态、运行、导出和组件包，保留旧策略 |
| `studio/native/chip_library.py:38,59` | v1包严格校验，合成根固定 classification-v2 | 新组件需求不能只修改模型根，库校验也要识别 |
| `studio/backend/studio_core/runtime.py:79,85` | 普通算子调用 nn 构造器，frozen 仅关闭参数梯度 | 自定义数学算子单独构造；BN统计冻结不能偷换原 frozen 语义 |

已有轻量核查：Document.add 可新增 Add/Concat，ResNet18 被明确拒绝为不支持的 primitive。这里只确认文档命令，不声称本轮重新完成界面或训练验收。S3 历史实训证据见 `studio/native/S3_VERIFICATION.md`。

外部参照：Vivado UG893 的独立页签组和水平/垂直工作区分屏 [E01]；PyTorch/torchvision 的 BasicBlock、BatchNorm 状态和 ResNet18 拓扑 [E02–E05]；DPG 节点编辑器的回调、端口和选择机制 [E06]。只借鉴工作方式，不移植其产品资产。

## 2. 首批范围与明确边界

S4必须交付：

1. 分类组件目录：基础层/数学与融合/功能芯片/视觉主干/当前工程芯片/本地芯片库；固定搜索，中文和英文别名、简介、端口、默认参数、展开/运行/Python能力说明。
2. 现有11种算子保留；新增 BatchNorm2d、Identity、Sigmoid、Multiply、Scale、OneMinus。
3. 可展开功能芯片：MLPHead、ConcatProjection、WeightedSum、GatedFusion、ResidualBasicBlock；Add/Concat直接入口标注“逐元素相加/沿轴拼接”。
4. 标准通道和 [2,2,2,2] block 数的 ResNet18 图模板，提供主干和带分类头两种目录项；原始大小32和224的代表性形状验证。不是缩窄网络冒充标准网络。
5. 主模型固定页签、芯片定义页签、调用上下文切换；单栏、左右双栏、上下双栏；最多16个打开的图页签、最多2个可见图视图。
6. 同一 Document、全局模型撤销历史；每页独立选择和探针上下文；跨页更新、错误定位、保存重开和旧工程读取。
7. 新组件和运行策略的受限 Python v4 往返，包含去除 visual metadata 的测试；CPU/CUDA EXE 后续重新构建并执行真实验收。

S4不交付：ResNet34/50、ViT、U-Net、交叉注意力、任意第三方模型导入、预训练权重下载/映射、任意代码组件、训练关系画布、交互式 backward 单步、完整实验工程导出、3D/分割/文本任务、多GPU/远程。目录不显示这些为已可用组件。首批主干随机初始化，初始化仍采用平台已声明的逐层策略；不声称与 torchvision 相同 seed 的初始权重相同。

精确视口恢复单独设门槛：本机 DPG 2.3.1 的公开 node_editor 接口无 pan/zoom getter/setter；get_x/y_scroll 属于 item 滚动，plot zoom API 不能代替。S4必须恢复页签、面板、选择、节点布局；会话内尽量通过常驻 editor context 保持视口。跨关闭/跨会话的精确 pan/zoom 在公开接口试验通过前标记“不支持精确恢复”，不以未验证 API、DLL 修改或电脑控制作兜底。若该能力成为必选，再独立评审画布适配器。

## 3. 路线取舍

| 路线 | 收益 | 代价 | 决定 |
|---|---|---|---|
| 在 ui.py 的 KINDS 后面继续添加名字和按钮 | 最快看到条目 | 白名单/默认值/形状/导出易漂移，仍不能展开成熟网络 | 不作为正式方案 |
| 无UI依赖的算子/目录描述＋图模板＋独立视图上下文 | 兼容当前执行核心、用户芯片与原生EXE，模板可展开/往返 | 需一次梳理状态、资源和UI事件边界 | 采用 |
| 直接把全部外部网络作为黑盒并重做前端 | 短期模型名多 | 难以兑现展开、参数身份及现有Python往返，增加依赖 | 后续插件方案，S4不采用 |

## 4. 组件及执行契约

### 4.1 模块边界与接口

下表列为新增的路径按阶段创建，部分已在A1/B1落地；具体完成状态以执行追记和验证记录为准。接口变更先更新本规格，再通知依赖任务。

| 新增模块 | 职责与公开接口 |
|---|---|
| `studio/backend/studio_core/operators.py` | torch-free `OperatorSpec(kind, required, defaults, input_ports, state_kind, python_constructor, editor_defaults)`；defaults为构造器描述，editor_defaults保持旧编辑预设，不能以构造默认填掉旧图required字段。`operator_spec(kind) -> OperatorSpec`、`operator_kinds() -> tuple[str,...]`、`operator_defaults(kind) -> dict`均提供防御性数据。python_constructor不代表当前导出器已支持；禁止任意eval/import |
| `studio/backend/studio_core/operator_rules.py` | `normalize_operator(kind, params) -> dict`、`infer_operator(kind, params, input_shapes) -> (shape, parameter_count)`；覆盖新算子，已有算子逐步接入并保留旧黄金行为 |
| `studio/backend/studio_core/operator_runtime.py` | `make_operator(kind, params) -> nn.Module`；显式构造 Multiply/Scale/OneMinus，不能假定它们存在于 torch.nn |
| `studio/backend/studio_core/component_catalog.py` | `ComponentSpec(id, category, label, aliases, description, required_config, capabilities)`；`list_components(query='', category=None) -> list[ComponentSpec]`；`build_component(component_id, config) -> ComponentBundle` |
| 同上 `ComponentBundle` | `entry: dict` 为待实例化 primitive/chip 规格；`definitions: list[dict]` 为闭包；`requirements: dict` 为算子集/资源策略；不含模型对象或权重 |
| `studio/backend/studio_core/module_templates.py` | `make_module(template_id, config) -> ComponentBundle`，纯图工厂，不导入torch或DPG |
| `studio/backend/studio_core/backbone_templates.py` | `make_resnet18(input_shape, *, num_classes=None, feature_outputs=('features',), resource_batch_bound=2) -> ComponentBundle` |
| `studio/native/catalog_panel.py` | 搜索/分类/配置对话框；组件添加到当前获得焦点的图，不根据悬停位置猜目标 |
| `studio/native/document.py` | 新 `add_component(bundle, position=None) -> str`，返回 `call:<id>`；合并依赖、ID迁移、创建实例/调用为一次模型命令，失败整体回滚 |

目录不得把 availability 与完整支持混在一起：首批仅展示已注册、可构造和有端口契约的条目；capabilities分别说明 expandable/forward/train/python_roundtrip/weights。用户本地包只解析有限JSON，不执行包内Python。相同ID不同内容一致改名；相同定义复用不会自动复用实例权重。

### 4.2 算子与功能模块

| 名称 | 固定契约 |
|---|---|
| BatchNorm2d | x:[N,C,H,W]→同形；num_features=C，eps>0，momentum为null或[0,1]有限值，affine/track_running_stats严格bool；参数计数affine时2C，统计buffers单独记录 |
| Identity / Sigmoid / OneMinus | x→同形；OneMinus计算1-x；不支持inplace，参数数0 |
| Multiply | a、b严格同形→同形；不隐式广播，参数数0 |
| Scale | x→factor*x；factor是有限固定标量，参数数0；不是可学习标量 |
| MLPHead | x:[N,F]→logits:[N,K]；Linear→ReLU→Linear；F/隐藏宽度/K显式配置 |
| ConcatProjection | a:[N,F1]、b:[N,F2]→features:[N,D]；Concat(dim=1)→Linear→ReLU |
| WeightedSum | a、b:[N,F]；y=a+alpha*(b-a)，alpha为任意有限标量；0..1是插值预设，范围外明确允许外推且不截断；用Scale/Add图实现，alpha仅绑定最终Scale，避免1-alpha绑定表达式失效 |
| GatedFusion | a、b:[N,F]；g=sigmoid(Linear(Concat(a,b)))，y=g*a+(1-g)*b；输出features和gate；不隐式投影或广播 |
| ResidualBasicBlock | Conv3×3→BN→ReLU→Conv3×3→BN，加Identity或Conv1×1+BN投影支路，再ReLU；stride仅1/2，形状和通道显式配置 |

改变模板公开的已有结构参数不会自动重写所有下游 shape。结构性变体通过配置工厂生成新定义；手工进入内部修改后遵循原有显式校验。禁止把有效但不适配的变更静默修成另一种模型。

**2026-11-17 A3工厂配置细化：**`make_module`只接受有限dict，未知字段明确报错，所有尺寸为严格正整数（不接受bool）。共用可选`resource_batch_bound`默认2，范围沿vision-v4契约；首批bundle均声明该显式策略。`ComponentBundle`现阶段只实现entry/definitions/requirements数据类型，完整检索目录留A5，不用空函数冒充目录。

| 工厂ID | 必需配置 | 可选配置及接口 |
|---|---|---|
| `head.mlp` | `in_features, hidden_features, num_classes` | 输入x，输出logits |
| `fusion.concat_projection` | `a_features, b_features, out_features` | 输入a/b，输出features |
| `fusion.weighted_sum` | `features` | alpha默认0.5且任意有限数；输入a/b，输出features；公开float alpha只绑定最终Scale.factor |
| `fusion.gated` | `features` | 输入a/b，输出features/gate |
| `block.residual_basic` | `input_shape, in_channels, out_channels` | stride默认1，仅1/2；input_shape为显式[N,C,H,W]且C匹配，N可未知或不超过bound；输入x，输出features |

以上均另接受共用batch字段。定义版本1.0.0，各次调用返回独立JSON数据，不缓存可变定义或模型权重。结构配置固定在新定义，除alpha外不宣称公开参数可自动重写整个结构。提供合理的内部节点初始布局，便于Python导回后展开检查；布局不参与模型语义身份。

**本轮用户体验约束（用户2026-11-17明确要求）：**优先保证真实操作可达、焦点清楚、编辑草稿不丢失、失败可恢复。B3使用现有原生桌面样式增加可直接操作的页签/单栏/左右/上下入口；项目树与组件结果独立滚动、搜索常驻；长名称/最多页签时仍能选择目标，空分屏提供有效打开入口。关闭图只关闭视图，主模型可直接返回。默认及现有最小窗口均测量关键控件可见性，不用截图中的文字存在替代可点击和真实回调。A3本轮交付工厂及受控导入验证，不把尚未接入的A5/I1目录按钮标为可用。

**同轮新增用户要求，覆盖上一段目录延期边界：**用户要求补充基础库，并提供组件工具栏、仪器栏及选择器截图，明确“元器件能直接拖出来”。本轮增加A5/I1的可用基础切片：17种已验证primitive与5种通过A3验收的工厂，分类/中文英文检索/参数与端口详情/点击和原生拖放添加；当前工程芯片也可拖放为独立实例。ResNet18仍等待A4，不显示为可放置条目。已有本地包导入保留，不把未来全部科研组件或多任务能力标为完成。

`ComponentBundle`移至`component_types.py`，`component_catalog.py`重导出该类型并实现描述/检索/构建，避免工厂与目录循环依赖。ComponentSpec原字段保留，required_config为键名tuple，增加config_defaults:dict、input_ports/output_ports:tuple供参数表单与预览。分类ID为layers、activation、normalization、math_fusion、shape_pool、functional；project/local由原生面板动态补充。capabilities分别描述expandable/forward/train/python_roundtrip/weights，首批weights为无预训练权重。所有描述/默认值返回独立数据；只注册可构建并通过验收的组件。

primitive bundle沿用当前工程策略，requirements只声明operator_set；芯片工厂bundle声明完整vision策略。`Document.add_component(bundle, position=None, *, apply_policy=False)`完成有限校验、定义闭包冲突迁移、新独立instance/call及位置作为一次可撤销命令。策略不匹配时默认拒绝且原子无变更；只有原生UI展示具体新策略后用户点击“应用策略并添加”，才以apply_policy=True在同一命令中修改策略及添加。不得自动修改实验batch或旧checkpoint；取消不改变项目。

原生拖放使用DPG drag_payload/drop_callback：payload携带有限组件身份/配置，目标canvas携带版本令牌；丢到非焦点但可见pane时显式激活该目标，不能按最后渲染页猜测。落点转换必须基于真实画布/节点几何并测试，不能把屏幕坐标直接当模型坐标。过期目标、越界/非法payload、取消策略提示均不产生半个组件。保留点击添加作为可达性后备，放置后选中新调用并可立即编辑/连线/展开。应用回调/framebuffer证明与真实OS拖拽手势验收分开记录。

**同轮连线/端口要求：**用户随后要求直线与转折的平整连线、按接口数量在左侧显示输入连接点/右侧显示输出连接点。本轮增加原生直角折线渲染与实际声明端口布局；线段只水平/垂直、90°转折，稳定出线与入线方向，多个分支保持真实连接关系。芯片按定义具名inputs/outputs创建真实连接点，primitive按OperatorSpec创建，不因装饰或探针增加伪端口。数据输入/输出边界节点分别保留其合理的0入1出/1入0出。以真实矩形核对端口左右对齐、计数及连线端点，不能只改截图。保留连接、选线/删除、探针、页签/分屏/移动和旧工程；仅连线样式/布局不改变模型身份。公开DPG能力不足时用有界独立绘图/命中适配层，不用私有DLL或Web替换；独立纯路由测试与实际framebuffer/注册回调均须验证。全局最优避障不作首版承诺，明显穿过源/目标模块与跨pane串线必须避免。手势输入仍禁用OS自动化，必须如实区分应用回调证明与未做的人手手势验收。

### 4.3 BatchNorm、共享和恢复

`frozen=True`继续只冻结可训练参数，BN统计按train/eval更新；UI写明“冻结参数（不冻结BN运行统计）”。若以后增加冻结统计策略，使用独立字段与源码语义。共享实例多次call共用参数和buffers，num_batches_tracked随真实调用增长；不同实例即便来自同一BasicBlock定义也有独立状态。[E05]

track_running_stats=True时state_dict严格保存/加载running_mean、running_var、num_batches_tracked；False时这些buffer键不存在，eval仍用批次统计，不能断言计数增长。affine=False时不产生weight/bias参数。buffers不得算成可训练参数或梯度更新。Python数值比较先显式装载同一组参数/buffers，不以相同随机种子代替相同权重。实际使用批次统计时若N*H*W≤1，明确失败并定位，不暗中丢弃末批或改eval。四种affine×track_running_stats组合均需验证。

A1补充的分配前状态限制：持久buffers单独计费，按模块实例树（含未调用实例）检查每个向量C不超过已有单张量上限、所有BN状态2C+1之和不超过已有累计上限；根与各definition默认验证独立建立该账本，不按call次数重复计持久状态。参数计数仍为0/2C，不调整旧激活计费或全局阈值。低层make_operator没有图资源上下文，处理图文档必须走prepare_graph/build_model。

### 4.4 资源策略与Python

新根图 `resource_profile='vision-v4'`、`resource_batch_bound: int`，范围1..64，默认工厂2。新档参数/单张量/累计执行元素硬上限与classification-v2保持相同：20_000_000 / 16_777_216 / 134_217_728。未知batch替代数由64改为显式bound，并按下文A2细化的编译/执行账本计费；传播到所有嵌套预算与输入校验。实际batch及配置batch不得超过bound。维持现有首轴之外不允许None的规则，不借新档开放动态空间/特征维。

旧无profile/classification-v2继续按64估算，但这不是给旧模型新增实际batch≤64的运行硬限制；新actual≤bound检查仅应用于vision-v4，旧模型继续原有元素预算行为，旧model_digest/checkpoint黄金值不变。新profile/bound计入模型及checkpoint身份，改变后保守拒绝直接续训。先以纯图累计预算和224/batch2实测证明ResNet18及一条配对融合路径；若超限，不擅自全局抬阈值，回到架构调整关口。

**D0证据带来的阻断修订（2026-11-16）：**按声明的17定义、69 primitive、各实例单call候选树，沿现有 `_infer` 重复遍历并验证全部definitions，batch1/2的条件累计估计分别148,151,712 / 296,303,424，均超134,217,728；独立复算一致。此为纯算例而非已运行的生产ResNet。仅更换null batch替代数不足以证明新档可行。

因此，A2实施前必须先明确仅新profile的预算计费契约，候选方向为区分有界编译/定义验证工作与真实根调用路径的执行累计元素，或对纯shape/参数计算使用有界缓存；明确缓存键含形状、公开参数、dtype和策略，不得去重真实runtime调用或合并BN/Dropout效果。全部定义仍须验证合法与有界，不能靠跳过未使用定义、无限缓存或提升全局阈值过关。旧档计费/黄金值不变。2026-11-17的A2已按下文细化实现分账并通过有限嵌套算例及独立审查；这不等于ResNet闭包已通过。A4及主干接入仍必须等真实闭包、芯片合成根及Python无metadata的预算测试通过。

新增Python profile `vision-pytorch-v4`，继承具名输出/typed public能力，扩展静态受限构造器和精确helper。根源码包含可解析字面量：

```python
MLFORGE_EXECUTION_POLICY = {
    'resource_profile': 'vision-v4',
    'resource_batch_bound': 2,
}
```

它是受支持源码契约，不是visual metadata；导出独立Model执行同样的batch上界检查。当前源码优先，删除visual metadata不能抹掉策略，源码非法/动态策略明确拒绝。旧profile继续原样导出，不能给所有旧图强制升级。

**2026-11-17 A2契约细化：**凡输出`vision-pytorch-v4`，均显式声明`MLFORGE_EXECUTION_POLICY`；含新算子但沿用旧资源策略时，`resource_profile`为`None`或`classification-v2`，且不带batch字段。旧v1–v3图不强制升级。声明vision-v4的受支持源码必须同时保留精确helper和根forward的批次检查；删除、移动或改写后明确拒绝导入，不能凭元数据或策略字面量重建源码已删除的行为。独立Python的数值、BN及batch契约需要一致；复制平台完整资源账本不在A2独立导出的声明范围。

新vision策略区分编译与执行：保留原有resolved/inference节点次数、单张量、参数和BN持久状态的上限；编译累计元素单列为账本观察值，不把编译重复推导当作重复执行。根图和每个definition默认调用路径各自受累计执行预算约束，全部定义仍验证。运行时只消除同一call内由规范化IR来源标明的嵌套输出别名；不同call即使返回同一个Tensor对象也分别计费，BN/Dropout仍按真实调用执行。旧策略维持原计费分支。有限嵌套图通过不等于ResNet18闭包已验证，后者仍是A4关口。

芯片包v2增加 `requirements={'operator_set':'vision-ops-v4','resource_profile':'vision-v4','resource_batch_bound':2}`，仍仅含架构闭包；库合成验证根使用声明策略。v1继续可读；新需求写v2，旧读者应明确拒绝。导入不自动改变当前工程策略：显示所需能力，由用户显式应用策略；任何不兼容保留原工程。实例配置值保留在工程，不被包内默认值覆盖。

### 4.5 ResNet18范围

标准stem、四阶段[2,2,2,2]、通道64/128/256/512；每个block为独立实例，定义可复用。支持1或3输入通道并明确1通道为结构变体。主干输出features:[N,512]，按配置可增加stage1..stage4具名输出；带分类头输出logits:[N,K]。用户可展开stem/stage/block并保存为自定义芯片。[E02–E04]

首批不提供预训练权重按钮，不下载torchvision；官方源码用作架构依据，独立手写参考用于验证。若未来引入torchvision用于开发侧对照，必须单独锁定兼容版本及许可证，不加入本轮运行依赖。ResNet34标准权重配置21,797,672参数超过旧20M上限，明确留后批；新增目录不得列作可用。[E04]

## 5. 多页签、分屏与身份

### 5.1 数据模型

新增 `studio/native/workbench_state.py`，不导入DPG或torch：

```python
@dataclass(frozen=True)
class ScopeTarget:
    definition_id: str | None   # None为根图

@dataclass
class ViewState:
    view_id: str
    target: ScopeTarget
    call_path: list[str]        # 从根开始，逐级call_id；定义无调用时为空
    selected_refs: list[str]
    probe_port: str | None
    pane: str                  # primary / secondary
    context_epoch: int = 0      # 瞬时事件版本，不持久化

class WorkbenchState:
    def open_root(self) -> str: ...
    def open_definition(self, definition_id, call_path=(), pane='primary') -> str: ...
    def activate(self, view_id) -> None: ...
    def close(self, view_id) -> None: ...
    def set_layout(self, mode) -> None: ...  # single / columns / rows
    def move_to_pane(self, view_id, pane) -> None: ...
    def reconcile(self, document, *, model_replaced=False) -> list[str]: ...
    def to_payload(self) -> dict: ...
    @classmethod
    def from_payload(cls, payload, document): ...
```

以上省略号为接口签名记法，不是让实施者复制的实现。公开字段为views、layout、active_by_pane、focused_pane、warnings及只读focused_view_id，详细类型见B计划。状态类不持有Document，只验证字段/布局/数量；实际图引用由WorkbenchPanel和view_scope验证，载入及模型变更后调用reconcile(document)。call_path长度不得超过现有MAX_CHIP_DEPTH，页面引用不得携带graph对象。构造器 `WorkbenchState()` 自动创建primary根页签，ID固定root。每个definition最多一个页签；再次打开激活原页签并切换所选调用上下文，保留原pane（移动另用move_to_pane），因此首版不能左右同时比较同一定义的两个调用视图。没有有效调用的定义可编辑，不伪造探针或实例覆盖。根页签固定primary且不可关闭，根视图仍可在single模式直接激活；其他页签可移至secondary，不复制图对象。触达16页签时提示用户关闭页签，不偷偷销毁页面。

初始active_by_pane为primary=root、secondary=null。空secondary不能持有焦点；关闭当前页、reconcile移除当前页、切回single时，焦点统一回退primary当前有效页，否则root。focused_view_id始终返回存在的页面。页签选择恢复是逻辑selected_refs与主题高亮，不声称恢复ImNodes内部选择位：公开API只有读取/清除，没有setter；激活或重绘后的空原生选择不能立刻覆盖逻辑选择，只有明确选择/清除操作才更新它。

### 5.2 命令与渲染隔离

新增 `view_document.py` 提供 `view_scope(document, view)` 上下文管理器：验证definition和call_path，从共享图临时解析scope/选择，执行后始终恢复原Document cursor；异常也恢复。需要在Document引入显式scope override而非构造虚假call路径；新增 `scope_override` 私有状态，只存definition ID而非graph dict指针，使用独立sentinel表示“未覆盖”，None明确表示根图，Document.scope/scope_id均在上下文期间读取override。若原目标因命令变更而不存在，恢复到root，避免finally错误遮住原异常。定义编辑允许无调用，probe_source在该情形明确不可用。现有直接Document导航API保持兼容。

UI仅主线程调用该适配器，worker仍接收深拷贝提交快照。所有编辑命令和选择/端口/探针操作携带view_id；不使用“最后渲染的scope”决定命令目标。

新增 `canvas_view.py` 持有 `CanvasView(view_id, editor_tag, node_refs, ref_nodes, source_attributes, target_attributes, link_targets, content_token)`；每个editor和节点tag以view_id命名，模型ID不变。建立 `CanvasHost` 管理创建/隐藏/刷新/销毁。单纯切页只隐藏和显示，不delete_item重建；验证时比较原生item ID和节点句柄，不能只比较可被重复使用的tag字符串。同一scope被编辑时刷新该定义、全部传递祖先和根图，保留无关常驻editor；依赖遍历带visited边界。任何原生视口保存能力由B0报告决定。

事件令牌同时包含view_id、generation、context_epoch、document_epoch。generation用于图控件重绘；context_epoch是瞬时视图字段，同定义页切换call_path时也递增，即使没有重绘。参数、冻结、探针及模态对话框回调同样携带令牌；A调用排队的回调不能在切到B后修改B。过期命令明确丢弃并提示重新操作。

单一全局撤销历史：撤销的是最近一次模型语义命令，与当前页签无关；完成后reconcile并刷新所有受影响视图。导航/页签/分屏/选择不增加模型revision、不进入模型undo、不令测量过期。节点布局修改不进入语义身份；同定义内部布局在所有引用中相同，不能伪装成每实例独立结构。

### 5.3 失效和调用语义

| 事件 | 规定行为 |
|---|---|
| 打开同定义的另一调用 | 重用定义页签；显式显示当前call_path和“修改定义影响全部引用”；实例参数只在选定上下文修改 |
| 关闭芯片页签 | 仅关闭视图，芯片及权重不删除；工程仍自动保存 |
| 删除定义/调用、undo/redo | reconcile清掉失效选择；定义不存在则关闭页签，定义存在但调用失效则保留定义页并清空调用上下文，探针显示未绑定 |
| Python导入、打开工程、替换起始模型 | model_replaced=True，旧页签全部重置为root；有v3视图载荷时再验证恢复，避免相同ID指向不同工程 |
| rename接口/连线变化 | 重绘引用处端口；非法尺寸定位调用路径，不抹掉草稿 |
| 运行中切页/分屏 | 不停止或重建worker，不重置权重；真正模型语义修改仍使旧测量过期 |
| 深层来源未在有界采集中 | 显示“本次未采集”，不得复制其他调用的测量或用零占位 |

### 5.4 保存格式

workspace v3：`{workspace_version:3, model, experiment, workbench:{layout,views,active_by_pane,focused_pane}}`。views固定为有序list，每项为 `{view_id,target:{definition_id},call_path,selected_refs,probe_port,pane}`；内存仍以view_id为key的dict维护。context_epoch、warnings和原生控件信息均不保存。只保存有限JSON与稳定ID，无DPG tag、Python对象、测量张量或原生句柄。最多16视图，分屏首版采用1:1，比例拖动调节留后续；非法引用净化回root并记录警告。未知workspace版本拒绝且备份原文件，不能覆盖读取失败文件。

WorkbenchState.from_payload返回状态对象，诊断通过瞬时warnings:list[str]交给UI日志；reconcile返回本次警告并追加warnings。read_workspace_full返回第三项原始workbench载荷，由NativeWorkbench调用from_payload，避免净化后丢失警告。旧格式第三项为None。workbench字段自身结构损坏时仅恢复默认root并发出警告，保留有效model/experiment；非法JSON、非有限数、未知workspace版本仍按文件失败/备份路径处理。

旧纯图和v2读入默认root；保留原 `read_workspace(path)->(document,experiment)` 与旧调用方。新增 `read_workspace_full(path)->(document,experiment,workbench_payload)`，`workspace_payload(...,workbench=None)`、`save_workspace(...,workbench=None)`在无workbench参数时仍按旧v2写，以维持旧测试和外部调用；新UI在所有正常、表单无效回退和退出保存路径始终提供视图载荷写v3，不能因回退experiment而降级成v2。`load_saved_workspace(paths)`保留二元返回，新增`load_saved_workspace_full(paths)`供启动三元恢复；NativeWorkbench构造器在原experiment之后增加仅关键字`workbench_payload=None`。启动恢复与显式打开使用同一路径。workbench不进入model_digest/checkpoint_signature，重开后的数据仍需重新检查。

## 6. 交付、验证和调整条件

固定6个交付批次：D0能力/预算验证；D1统一算子及资源/Python契约；D2功能芯片与ResNet18；D3多页签渲染核心；D4目录、分屏、持久化和定位集成；D5独立验收、文档及CPU/CUDA便携包。任务详见两个子计划，依赖图见总计划。

S4完成至少同时满足：

- 空白工程从目录添加Add/Concat、一个融合芯片和ResNet18，中文/英文检索均可达；项目树增长不遮挡搜索。
- ResNet18可逐层展开，主模型保持打开；16页签边界和两个可见视图事件不串scope。
- 同一实例共享、同定义不同实例、BN统计、冻结、梯度、更新、checkpoint/Python状态均与独立参考核对。
- 224/batch2主干和小型配对任务真实执行；资源限制超界提前失败，不静默改网络/输入/batch。
- 页签切换/移动/关闭/保存不改变模型语义或运行；语义编辑和stale状态按原规则工作。
- v1芯片/v2工作区和旧Python/旧checkpoint回归通过；v3工作区/v2芯片包新格式通过。
- 源码与冻结CPU/CUDA EXE各运行声明的代表性检查；测试fixture不是科研性能证据。不能用按钮、截图或构建成功替代运行。

若DPG隐藏/分屏无法保留正确独立上下文，先修CanvasHost或停止承诺精确视口，不换Web前端。若ResNet预算或模型格式不能表达正确状态，先修核心契约，不把成熟网络换成同名小CNN。若范围扩张到多任务训练/任意Python，拆为后续阶段。

## 7. 角色、投入与不确定性

需要核心/模型、原生UI/交互、测试/打包三类能力；同一工程师可兼任。核心组件与视图核心可并行；ui.py、document.py、python_io.py等共享入口由集成人统一修改，避免并发覆盖。主要工作量分别为：算子/状态/导出大，模板与独立参考大，多视图/失效/迁移大，目录呈现中，冻结验收中至大。这里是相对规模，不是人日估算。

D0后记录实际图规模、修改点和DPG行为，再给日历工期；当前不编造完成百分比或固定天数。未计后续模型权重、临床数据、大模型/服务器、签名证书、安装器和云成本。GitHub上传继续暂停。

## 8. 官方资料索引与核查边界

核查日期均为2026-11-15。固定版本资料用作本地torch2.6语义依据，不声称是最新版本。

- E01：AMD Vivado UG893 2025.2《Splitting the Workspace》；独立页签组和双向分屏。
  `https://docs.amd.com/r/2025.2-English/ug893-vivado-ide/Splitting-the-Workspace`
- E02：torchvision0.21《resnet18》；架构入口、权重参数与默认无预训练。
  `https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html`
- E03：torchvision v0.21.0 ResNet源文件；BasicBlock/stem/stages结构与初始化差异。
  `https://raw.githubusercontent.com/pytorch/vision/v0.21.0/torchvision/models/resnet.py`
- E04：torchvision0.21《resnet34》；标准权重配置参数数。
  `https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet34.html`
- E05：PyTorch v2.6.0 batchnorm.py；参数、统计buffers和train/eval行为。
  `https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/nn/modules/batchnorm.py`
- E06：Dear PyGui《Node Editor》；节点/端口回调、选中查询。pan/zoom限制来自本机2.3.1公开签名核查，不能由网页未列出API单独推断。
  `https://dearpygui.readthedocs.io/en/latest/documentation/node-editor.html`

本轮增补核查日期2026-11-17，仅借鉴交互机制，不复制产品图标/资产：

- E07：Altium Designer《Searching for & Placing Components》；分类检索、详情以及一次拖放放置一个实例。
  `https://www.altium.com/documentation/altium-designer/schematic/searching-placing-components`
- E08：Altium Designer《Building & Maintaining Your Components and Libraries》；组件面板整合库来源与参数/模型详情。
  `https://www.altium.com/documentation/altium-designer/components-libraries`
- E09：Dear PyGui《Item Callbacks》；drag_payload、payload_type与drop_callback接口；实际投放和坐标仍以本机公开API及原生渲染测试核实。
  `https://dearpygui.readthedocs.io/en/latest/documentation/item-callbacks.html`

AMD单独Window Tabs及Packaging-a-Block-Design页面本轮访问失败，不将其列为本轮重新核实的证据；不影响E01已经明确描述的工作区分组与分屏机制。PyTorch2.6生成文档页访问失败，使用精确tag源码E05替代。
