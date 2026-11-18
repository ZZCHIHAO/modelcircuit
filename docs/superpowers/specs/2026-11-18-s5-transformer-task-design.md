# S5 Transformer、可展开 ViT 与最小 TaskAdapter 设计

日期：2026-11-18。产品：**ModelCircuit Studio｜模路**。状态：后续阶段设计，**尚未实现**。承接用户已批准的 S5 方向及 `docs/superpowers/plans/2026-11-18-s4-desktop-release.md` Task D；S4 CPU/CUDA 桌面包先完成交付，本文件不改变 S4 的发布退出关口。

这是历史设计基线。`v0.5.0-alpha.2` 的落地范围和实测结果以 [S5 验收记录](../../../studio/native/S5_TRANSFORMER_VERIFICATION.md) 为准。

## 1. 目标与阶段边界

用户能够在原生桌面左侧组件库放置 Attention、Encoder、小型 ViT，展开芯片看到真实张量运算，修改参数与连线，在现有页签/分屏里定位类型错误和观测结果，完成本地图片分类短训练，并导出、编辑、重新导入受支持的静态 Python。成功标准是结构、计算、梯度、参数共享、状态、训练与桌面操作共同成立；增加菜单名称不算完成。

S5 分成依赖有序的三个交付段：S5-A 建立有限张量契约、显式注意力和独立计算证据；S5-B 建立可展开 Encoder/小型 ViT，并复用既有图片分类闭环；S5-C 仅从已证明的分类路径提取 TaskAdapter 接缝。回归/多头任务另设后续证明阶段，未通过该阶段不得称为通用任务平台。

不在首批承诺：任意 Python 导入、完整训练关系图、动态图控制流、动态序列长度、tokenizer/文本语料工作流、预训练权重、生成/KV cache、Decoder、Swin、3D、分割/检测、多 GPU、AMP、任意 dtype、通用插件执行。`Embedding` 需要 int64 端口、索引越界、padding/state/数据管线的单独设计与验证；小型 ViT 的可学习位置向量不依赖词表 Embedding，不能借此宣称文本模型已支持。

## 2. 当前代码证据与需要跨越的接口

以下为本轮读取的真实源码位置，行号只定位当前基线；实施前重新定位，不机械复制行号。

| 当前证据 | 对 S5 的约束 |
|---|---|
| `studio/backend/studio_core/operators.py:41`、`:53` | 有限注册表共 17 项；没有 Reshape、Permute、MatMul、Softmax、Embedding 或参数源对象。不能把新增模板视为已可执行。 |
| `studio/backend/studio_core/operator_rules.py:95`、`:108` | 形状推导独立于 Torch；Add/Multiply 要求等形，拒绝隐式广播。保留这个规则。 |
| `studio/backend/studio_core/schema.py:178`、`:211`、`:434` | 只允许首维为 null、输入 float32、schema_version=1。新增 bool mask 与无 batch 参数源需要显式新契约。 |
| `studio/backend/studio_core/schema.py:318`、`:347` | 实例决定参数计数；稳定逐调用拓扑序关系到 Dropout RNG，不可随意批处理 ready 集合。 |
| `studio/backend/studio_core/runtime.py:64`、`:94`、`:108` | ModuleDict 按 instance 建模块，call 复用模块；输入固定 float32；非芯片运行分支按 a 或 x 派发，尚不支持零输入和 q/k/v。 |
| `studio/backend/studio_core/runtime.py:100`、`:135` | capture 当前会 detach/clone 张量，结果阶段截短不能阻止先克隆巨大注意力矩阵。 |
| `studio/backend/studio_core/python_io.py:129`、`:342`、`:715` | 现有多个 Python profile、有限字面输入注解、AST 静态导入及 2 MB 源码边界；不能执行上传源码。 |
| `studio/backend/studio_core/python_io.py:683` | visual metadata 仅恢复匹配对象的位置/名称等显示信息，不能覆盖新的 dtype、轴、mask、共享或源代码语义。 |
| `studio/backend/studio_core/classification_runtime.py:247`、`:262`、`:278`、`:303` | 实际训练要求 logits=[B,class]、交叉熵和单个优化器。ViT 首先输出该接口，避免同时重建工作流。 |
| `studio/backend/studio_core/classification_runtime.py:325`、`:364`、`:401` | 梯度来自选定 checkpoint epoch 的最后训练 minibatch；最终张量探针来自该 checkpoint 的验证集 eval 前向，并非交互式即时梯度。 |
| `studio/backend/studio_core/memory_admission.py:90`、`:95`、`:123` | 当前按 float32 和累计执行元素估算；已区分最低存储与预测峰值。新 dtype/视图/attention workspace 必须纳入估算。 |
| `studio/backend/worker.py:18`、`studio/native/ui.py:999` | RunManager→独立 worker 已存在，GUI 提交不可变快照；延续进程隔离和取消，不在 GUI 分配权重。 |
| `studio/native/catalog_panel.py:166`、`studio/native/workbench_panel.py:136`、`studio/native/chip_panel.py:154` | 可复用左库、调用/定义页签和接口编辑；新功能必须接到真实回调与保存恢复。 |

`hardware-v5` 取消人为参数/单张量/累计元素天花板的用户决策继续有效。旧 v1–v4 资源策略及 Python/摘要/checkpoint 契约保留原行为；解析文件、图节点、递归和推导工作量边界与硬件容量是两回事。

## 3. 方案选择

| 方案 | 收益 | 代价与结论 |
|---|---|---|
| **有限张量原语 + 可展开芯片 + 分类优先** | 算子可组合、共享可见、Python 可静态还原；沿用已可用桌面与训练 | 需要一次清晰的类型/执行桥接；采用。 |
| 把 nn.TransformerEncoder/torchvision ViT 当不透明单节点 | 初次接通较快 | 不能证明展开、内部修改和真实 attention 探针；不作为本轮产品实现。可作为独立参考。 |
| 一次引入完整任务关系图、任意模型导入/执行框架 | 远期范围广 | 同时改变图、执行、安全、数据、loss、checkpoint、UI，难以独立验收；分解到后续。 |

新模板生成普通 definitions/instances/calls，无隐藏执行支路。运行、Python、本地芯片与 UI 使用同一有限算子契约。

## 4. 文件格式与 TensorSpec

新增图采用 `schema_version: 2`、`capability_profile: "tensor-v1"`、`resource_profile: "hardware-v5"`。三者分别表示格式、张量语义和执行容量策略，不能用一个 profile 同时表达三类含义。`resource_batch_bound` 保留 1..64。旧图不自动升级；显式升级生成新修订，保存前可撤销。v2 读取器能只在内部把 v1 float32 输入规范化为 TensorSpec，禁止静默改写原文件或旧模型摘要。

TensorSpec 是有限 JSON 对象，使用现有 shape/dtype 字段并新增 axes/batch_axis：

```json
{"shape": [null, 65, 64], "dtype": "float32",
 "axes": ["batch", "token", "feature"], "batch_axis": 0}
```

- 首批 dtype 仅 float32、bool；bool 不可训练、不参加 Linear/MatMul/loss。int64/Embedding 不接受为占位能力。分类标签仍是数据适配层的 int64，不因此成为模型图支持的输入 dtype。
- 模型数据输入有 batch 时 `batch_axis=0`；只有该轴可以为 null。序列长度、图像几何、头数及特征宽度均为正整数且静态固定。固定 batch 的端口也必须标记 batch_axis=0。运行时所有带 batch 的输入实际 B 相等。
- 参数/常量/固定 causal mask 的 `batch_axis=null`，所有维度固定；没有把它们首维的 1 误认为真实 batch。rank 继续 1..8，不新增标量张量。
- axes 是同 rank、互异的有限标识符列表，保留 `batch`；非 batch 标签如 token/query/key/feature/head/channel/height/width 描述位置意义。标签进入图语义、导出和诊断，不能从 visual metadata 恢复。标签不是自动单位推导系统：维度兼容仍由算子规则验证；明确改变轴意义只通过 Reshape 的 target_axes 或 AxisRename。
- Chip 边界校验 dtype、rank、固定维度、batch_axis 与 axes；输出沿 source 推导。报告新增 `tensor_specs` 映射，同时保留原 `shapes` 投影供旧 UI 使用。bool 的显存/内存按 1 byte，float32 按 4 bytes 计。

初始序列长由构图配置决定。改变 T/D/heads/patch 需要新修订、重新验证和重新建立实验身份；不在训练中插值位置向量，不以 null 代替任意 token 长度。

## 5. 首批张量算子、参数对象与状态

旧 17 项保持原数学行为。在 tensor-v1 内为 Linear、LayerNorm 添加有限可选构造参数：Linear.bias（默认 true）、LayerNorm.eps（默认 1e-5）、elementwise_affine（默认 true）。旧 profile 的省略形式和导出黄金契约不改变。Encoder 显式指定 eps=1e-6。需要非默认旧参数的图须显式升级。

| 新 kind / 输入端口 | 参数及推导 | 执行/状态约束 |
|---|---|---|
| Reshape / x | `shape` 为正整数列表，可用唯一字符串 `"B"` 保留首位 batch；`axes` 为目标轴列表 | 每个样本元素数严格相等；B 不可合并/拆分，不支持 -1、0、任意算术或 shape 变量；使用 reshape 语义，非 contiguous 输入允许复制。 |
| Permute / x | `dims` 必须是 0..rank-1 的排列 | 有 batch 必须 dims[0]=0，轴标签同步排列；不把视图当零成本保证。 |
| AxisRename / x | `axes` | 只改标签，shape/dtype/batch 不改；用于把 token 标为 query/key 等，不能绕过尺寸检查。 |
| BroadcastTo / x, like | `shape` 为固定维度和首位 `"B"`，`axes` 为输出轴 | like 只提供同设备 B，须带 batch；右对齐各源维只能相同或 1，rank 只增不减。批次来源必须显式连线；输出=expand 的只读视图语义，禁止原地写入，反向沿广播维求和。 |
| MatMul / a,b | 无参数 | float32、rank≥2；所有前缀维完全相同，最后收缩维相同；**不隐式广播**。输出保留左侧行轴、右侧列轴，轴名冲突需先 AxisRename。 |
| Softmax / x | `dim`（含负轴） | float32；轴可定位且不能为 batch；沿该轴规范化。mask 检查见下节。 |
| Select / x | `dim,index` 为字面整数 | 静态 index 在范围内，禁止选择 batch；移除指定轴，保持正 rank；ViT 读取 cls token。 |
| ParameterTensor / 无 | `spec` 为无 batch float32 TensorSpec；`init` 为 zeros 或 normal(mean,std) | 每实例注册一个 `nn.Parameter`，只在构造时初始化，call 不重新采样；freeze 与既有实例规则一致。cls=[1,1,D]，pos=[1,T,D]。 |
| CausalMask / 无 | `length` 为正整数 | 注册持久 bool buffer [1,1,T,T]；含对角的下三角为 true；不是可训练参数，不允许动态图生成代码。 |
| MaskApply / scores,keep | 两者严格同形；scores float32、keep bool | true=允许参与；false→-inf。只有显式 BroadcastTo 处理 [B,T] 或 [1,1,T,T] 的扩展。 |

首批不提供任意张量字面大数组、用户指定文件加载、forward 内随机张量工厂、任意函数/原地操作。图 JSON 只保存初始化策略，权重保存于已有 checkpoint/state_dict；加载同配置的图不等于加载已训练权重。ParameterTensor 状态键固定 `components.<instance_id>.value`，嵌套沿现有 components 路径扩展；同实例多次调用只计一次状态和参数，复制实例才产生独立参数。

Runtime v2 统一通过 `component(**arguments)` 派发有限注册端口（含空字典）；现有算子的 wrapper 适配 x/a/b，不再根据是否存在 a 猜接口。不能因此开放任意关键字或 getattr。一个 call 只保存其自身观察路径；共享实例跨 call 的结果不能互相覆盖。参数对象多 call 返回同一个 Parameter，观察快照按 call 标识；捕获不能改变 autograd 图。

## 6. 掩码、数值与随机性

平台统一 bool mask 的 **true=keep**，明确标在端口和导出 helper 中。PyTorch 2.6 SDPA 的 bool attn_mask 也是该含义；不能把其他接口的 mask 原样猜成兼容。浮点 additive mask 首批拒绝。`MaskApply` 只允许供紧邻的 Softmax(dim=-1) 使用的 masked-score 输出；验证器不把含 -inf 的该中间结果当普通全有限 logits。运行时检查每个 query 行至少保留一个 key；全遮蔽行在 Softmax 前报带 call 路径的错误，禁止 NaN、静默零向量或改写 mask。普通算子非有限检查保持有效。

padding mask 只定义 key 可见性；不承诺自动清零 padded query 输出，也不承诺跨样本不同长度的数据管线。一个 bool keep 输入可用于有限单元测试；首批图片 ViT 不需要 padding mask。causal mask 与外部 keep mask 不提供隐式合并；需要组合时使用后续独立逻辑算子扩展，不在输入解释中暗藏规则。

Dropout 使用已有模块 train/eval 开关；每个 call 按稳定拓扑序消耗 RNG。同一 Dropout 实例连续两次调用不复用同一个 mask。构造顺序、seed、参数初始化、CPU/CUDA RNG 与 shuffle state 要进入复现/恢复证据；参数 sharing 不等于随机 mask sharing。仅同设备、同执行策略的恢复测试要求逐步一致，不宣称跨设备 bitwise 一致。

初期注意力只用显式 MatMul→Scale(1/sqrt(head_dim))→可选 MaskApply→Softmax→Dropout→MatMul。**首批不启用 SDPA/fused 替代路径**。后续 SDPA 执行策略必须独立版本化，eval 显式 dropout_p=0；记录请求策略、实际可确认的实现与未知项，不根据设备名宣称 FlashAttention。不能把显式路径另算的 attention 权重标为优化内核的“实测中间张量”，也不能要求两种内核/dropout 路径逐元素相同。

## 7. 可展开 Attention、Encoder 与小型 ViT

`make_attention(*, sequence_length, hidden_dim, num_heads, dropout=0.0, masked=False, resource_batch_bound=2)` 返回普通 ComponentBundle。要求 D%H=0。输入 x=[B,T,D]；三个独立 Linear(Q/K/V)→Reshape [B,T,H,d]→Permute [B,H,T,d]；K 转置最后两轴，MatMul 得 [B,H,query,key]，缩放/可选 mask/softmax/dropout 后乘 V，转回 [B,T,D]→输出 Linear。构造出的 Q、K、V 不是共享参数，除非用户显式重新连到同一个实例。

芯片具名输出 `features`；需要 weights 的配置显式声明输出 `attention`，指向 Softmax 后、Dropout 前的实际张量。Dropout 后量称 `attention_dropped`，不混称概率分布。输出 aliases 和采集路径按已有 named-port 机制推导，模板不为了 UI 重算注意力。

`make_encoder(*, sequence_length, hidden_dim, num_heads, mlp_dim, dropout=0.0, attention_dropout=0.0, depth=1, resource_batch_bound=2)` 生成 pre-norm Encoder：LN→Attention→Dropout→Add residual→LN→Linear(D,M)→GELU→Dropout→Linear(M,D)→Dropout→Add；最外层最终 LN。depth 个 block 默认不同实例，允许用户把两个 call 指向同一 block 实例进行有意权重共享。轴/形状完全相同也不能自动合并实例。

`make_small_vit(*, image_size=32, patch_size=4, channels=3, hidden_dim=64, num_heads=4, mlp_dim=128, depth=2, num_classes=2, resource_batch_bound=2)` 用 Conv2d(kernel=stride=patch)→Reshape→Permute 得 64 个 patch，显式 BroadcastTo(cls, image) 后 Concat 得 T=65，显式 BroadcastTo(pos,tokens)→Add，Encoder→Select(token,0)→Linear 得 logits=[B,C]。图片尺寸首批为正方形，须整除 patch；界面显示配置所得 T，不自动裁切或改变 patch。所有 dropout 默认 0；支持显式非零配置与独立随机性测试。

目录稳定 ID：`tensor.attention`、`tensor.encoder`、`vit.small_classifier`。产品名为“小型 ViT 分类器（随机初始化）”，不称标准 ViT-B/16，不把 32×32 验证推广成 ImageNet 效果。cls 初始化 zeros，pos normal(0,0.02)，普通 Linear/Conv2d 使用明确记录的 PyTorch 默认初始化；本变体分类头沿默认初始化，便于短训练观察主干梯度，不冒称完全复刻 torchvision 的零初始化分类头。数值验证通过显式复制同一权重排除初始化分歧。

## 8. Python 往返与语义优先级

新增 `tensor-pytorch-v1` profile，只对 schema v2 图选择。继续使用普通 nn.Module、ModuleDict、显式 forward calls 和可独立运行的有限 helper 源码；导出不依赖 studio_core。v1–v5 原 Python profile 保留。不是任意 Python→图编译器。

输入注解扩展为有限字符串，例如 `'float32[?,65,64];axes=batch,token,feature;batch=0'`、固定 mask `'bool[1,1,65,65];axes=mask_batch,mask_head,query,key;batch=none'`。新 profile 强制完整注解，不从 metadata 或猜测补类型；旧 profile 原注解规则不动。ParameterTensor 的 spec/init、Reshape/BroadcastTo 目标及轴、mask helper 语义和执行策略均在真实构造源码里，属于可审查程序语义。

导入顺序：先限源码字节/AST 工作量→识别允许的 imports/helper 精确 AST/全局绑定→解析类与实例构造→按源码 forward 恢复 calls/共享/端口→纯验证→恢复匹配对象的显示信息。拒绝 helper 篡改、别名遮蔽、动态控制流、外部读取、任意 torch 调用；诊断含源码行号和可定位 call。视觉信息缺失只丢位置/名称，不丢 dtype/轴/参数共享/初始化/mask/批次策略。

必测四条路径：有 metadata 往返；删除整段 metadata 后往返；保留过期 metadata 但编辑受支持的 Linear/Reshape/keep 连线/共享调用源码后导入；编辑成不支持表达式或类型不匹配时拒绝。成功导入必须服从源码，不能恢复旧图覆盖修改。比较语义摘要、输出、梯度、state_dict 及一步 SGD；不只比较 JSON 文本。

## 9. 采集与硬件容量

attention 空间随 B×H×T×T 增长。图编辑/导入只做纯推导，不探测硬件、不构造参数、不分配 attention 张量。hardware-v5 在 worker 查询所选 CPU/CUDA 容量；区分参数/梯度/优化器/持久 bool buffers/视图/reshape 可能复制/attention logits 与 probabilities/dropout/autograd/workspace/host staging/checkpoint/probe transfer。最低必要存储超可用容量则分配前拒绝；预测峰值与余量只告警，真实 OOM 可读失败，不更换设备、不减少 T/B/D、不重新引入固定参数天花板。

首批新增 **capture-before-copy** 策略，应用于 tensor-v1 图：只采集用户请求的有限 source，默认最多 16 个来源，单次结果总采集预算 1 MiB、attention slice 最多 64×64 个 float32（16 KiB），参数摘要沿旧上限；这些是观察/UI 响应边界，不限制训练模型大小。每个来源先按预算选 slice/reduction，再 detach/clone/transfer，禁止先全量 clone 再截断。未请求张量不留观察副本；预算不足标明 omitted 与理由，不能显示旧采样。

ProbeRequest 含 call 路径、port、sample_index、head_index、query/key 范围、统计选择。训练前把请求冻结为 run snapshot；运行中改变选择不会改变已运行模型。首批观察是选定 checkpoint 的 validation eval，显示实际 sample_ids、设备、epoch、模式、形状、切片坐标、是否截断和采样点。梯度沿用“该 checkpoint epoch 最后训练 minibatch”；参数更新为 selected checkpoint minus initial，不显示成验证样本的局部梯度。实时单样本 backward/点击重运行另设功能，不在本轮声明。

后续如新增优化 attention，只能标记实际可观测 outputs；内部 weights 无法得到时明确 unavailable。另开显式诊断重放属于新的测量，需要单列模式、seed、输入及策略，不能混在同一次性能/显存记录。

## 10. TaskAdapter 的最小接缝

先让小型 ViT 在现有 `classification-v2`、image_folder、RunManager→worker→run_classification 路径通过。之后才提取 `task_adapters.py`（有限内置注册）和 `classification_task.py`（分类语义），不在 S5 重写通用训练循环或开放自定义 Python 插件。

```python
# 拟定接口，属于设计；当前仓库尚无这些对象。
Batch = {"features": dict[str, Tensor], "targets": Tensor,
         "sample_ids": list[str]}  # TypedDict；此处说明字段
BatchStats = {"loss_sum": float, "correct": int, "count": int}

class TaskAdapter(Protocol):
    id: str
    version: int
    def validate_contract(self, tensor_specs: dict, manifest: dict) -> None: ...
    def bind_batch(self, raw_batch: object, device: str) -> Batch: ...
    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> Tensor: ...
    def summarize(self, outputs: dict[str, Tensor], batch: Batch,
                  loss: Tensor) -> BatchStats: ...
    def finalize(self, totals: BatchStats) -> dict[str, float]: ...
```

唯一可选 adapter 是 `classification`/version=1：float32 logits=[B,C]；targets 是 [B] int64，类索引/类数/样本 ID 来自已检查 manifest；loss 是 CrossEntropy 默认 mean，统计累计按样本数加权。`bind_batch` 复用现有单图像/数值/paired 数据映射，不改数据拆分、预处理、loader 顺序或标识规则。

runtime 仍负责优化器、zero_grad/backward/step、train/eval、取消、进度、checkpoint 原子写入及 RNG；adapter 不持有模型/优化器，不保存结果文件，不产生第二训练循环，不读 GUI 或任意文件。预测表、混淆矩阵及当前 classification 结果协议保持原调用流程。`get_task_adapter("classification",1)` 采用静态注册，未知 ID/version 在 worker 分配前失败；UI 不增加看似可选的空任务项。

旧 schema v1 分类 checkpoint/version=2 和签名原样保持。schema v2 新运行使用 checkpoint/version=3，仍属于 classification-v2 workflow，identity 增加 `adapter={id,version}` 与完整 tensor-v1 图语义；读取器显式区分两种版本，严格拒绝旧/新图互换恢复和不匹配任务。格式新增字段不得用默认猜测令旧签名失效。回归/多头证明需另立规格：定义输出/targets/loss 加权、分割/归一化/指标、schema/checkpoint 身份、数据和实际原生 UI，再实现第二 adapter；接口存在本身不是通用性证据。

## 11. 原生桌面可用流程

保持 DearPyGui 原生桌面，不迁移浏览器。左库新增 Tensor/Attention 分类与三个真实目录项；配置面板先展示输入 dtype/轴、T、D、heads、patch 和生成形状，非法整除/轴/掩码在放置前定位。每个目录项沿现有 `ComponentBundle` 插入/撤销/冲突改名机制；不新增独立模型编辑器。

用户从 image_folder 检查→在目录配置小型 ViT→放置/命名 logits→校验→在原生实验面板选 CPU 或 cuda:N→运行→查看 loss/参数更新/attention slice→取消或查看 checkpoint→保存工作区→重启恢复。任何一步失败都保留可编辑图和 Python 草稿。芯片可双击展开到调用页签，定义页签修改共享定义；面板明确“共享实例 × 调用次数”与“复制为独立实例”。接口编辑增加 dtype/axes/batch，错误能导航到定义或具体 call 路径。

探针面板显示二维 heatmap/数值 slice 与 head/sample/range 选择，使用 framebuffer 的真实注册回调验证；只在训练前冻结请求。保存 v3 工作区时保存有限 ProbeRequest，禁存张量/DPG handle；旧工作区缺该字段可读。探针选择恢复后无对应实测结果显示“尚无本次结果”，避免把模型改变前旧矩阵当新结果。默认窗口和最小支持窗口检查库、页签、配置、运行/取消、错误和探针可访问。

## 12. 验收矩阵与独立证据

| Gate | 必须保存的证据 | 失败时不解锁 |
|---|---|---|
| G0 兼容/格式 | v1–v5 黄金契约不变、v2 dtype/axis/batch/空输入源错误、旧包/工作区只读保护 | 新图执行与目录 |
| G1 张量/状态 | 每新算子独立 Torch 数值；非 contiguous reshape；广播梯度求和；bool/all-masked 错误；共享/复制/冻结状态 | Attention 模板 |
| G2 Attention | 独立手写 Torch attention，显式权重映射，无从平台图生成参考；前向、输入及 Q/K/V/out 梯度、一步 SGD、state_dict；无 mask/causal/外部 keep；T=1 和小 T、heads=1/多头 | Encoder/ViT 对外可用 |
| G3 Encoder/ViT | 完整结构/参数数、独立参考输出/输入及全部参数梯度/SGD/状态/复用；image=32、patch=4、D=64、heads=4、depth=2；真实分类 worker 一轮与 resume/取消 | 分类闭环完成 |
| G4 Python | metadata 有/无、受支持源码改动优先、过期 metadata、防 helper 篡改、共享/独立/冻结、状态键映射；独立导出可运行 | Python 往返声明 |
| G5 观测/容量 | copy 前预算、实测 slice 对照、来源/epoch/grad 文案；请求增长预算稳定；CPU/CUDA 容量拒绝/未知/OOM/不改设备 | 大 T 观测及桌面完成 |
| G6 原生/适配 | 真实回调+framebuffer、页签/左库/保存恢复、运行取消；提取前后分类结果/签名/RNG 等价；v2/v3 checkpoint 对照 | S5 阶段关闭 |

独立参考不得导入平台图工厂、算子 runtime、shape 推导或从图遍历拼层。手写参考使用 Torch nn/functional，明确 Q/K/V 拆分与参考权重字典；state 映射逐项证明名称、shape、唯一性和无遗漏，不能按迭代顺序 zip。通过共享实例注入、state 删项/错形、mask 翻转等负向测试证明验证器能发现错误。

数学对照统一 dropout=0、复制相同权重，CPU/CUDA 分开报告。初始容差为 CPU 前向 atol=1e-6/rtol=1e-5、梯度/一步更新 atol=2e-6/rtol=1e-4；CUDA float32 前向 atol=1e-5/rtol=1e-4、梯度/更新 atol=2e-5/rtol=2e-4。禁 TF32 与非确定算法用于该组对照；比较每个有限张量及近零值的绝对误差。超差先定位算法/映射，不静默放宽。离散结构、dtype、mask、共享 identity、state key/shape 必须精确一致。非零 dropout 另测 train/eval、seed、稳定调用序与 checkpoint 同设备连续/恢复，不强求不同实现的随机掩码逐值相同。

CPU 必须实际通过；CUDA 有现有可用环境时实际通过并记录设备/版本，不可用则记明确阻断而非 pass。文档编写阶段不运行训练、安装依赖、下载数据或模型。未来验收只用本地生成小样本与现有环境；短训练验证正确性，不提供科研准确率/吞吐结论。每 Gate 记录命令、退出码、实际设备/版本、输入/config、失败修复和产物哈希，不使用估算替代测量。

## 13. 迁移、风险和关闭条件

- 格式迁移：新 schema/version 与 capability 显式拒绝不兼容客户端；旧图升级可撤销。新芯片包采用 chip_format_version=3，并在 requirements 指定 schema_version=2、capability_profile=tensor-v1、operator_set 与 hardware-v5；现有 v1/v2 包保持读写协议及内容。v2 工程纳入 v1 芯片使用内部规范化；v1 工程放置新芯片先出现明确升级结果，确认应用现有策略的交互后单次原子撤销。不能只改包版本不改 importer。
- 训练兼容：TaskAdapter 最后提取，旧 profile/模型摘要/checkpoint 黄金测试先锁定；新 v3 checkpoint 不误读旧数据。adapter 不能扩大原分类任务输入承诺。
- 计算风险：reshape 可能复制、expand 别名、Softmax mask 的 -inf 和 RNG 顺序分别测试；参数源与共享调用计数去重，观测按 call 隔离；bool 字节估算不再沿用全 *4。
- 容量风险：attention 平方成本、后台进程共享显存、分配器碎片和 workspace 使预测不精确；界面区分估算、实测及未知。参数无固定上限不等于图解析工作量无边界，也不等于任意硬件可运行。
- 版本风险：验证基线采用现有 PyTorch 2.6 / torchvision 0.21 官方契约，实施时记录实际安装版本；不安装替换环境来制造通过，不混用 stable/latest 文档。
- 关闭条件：G0–G6 全部证据齐备、未通过能力不上可用目录、原生实际流程完成、遗留范围明确、S4 旧包受保护；S5 打包/发布另受当次交付计划约束。本文件不表示 S5 已实现，也不解锁后续非分类/生成/通用代码功能。

## 14. 官方语义参考

以下是版本固定的官方依据，查阅日期 2026-11-18；并非当前项目已兼容这些库所有接口。PyTorch 2.6 SDPA 文档说明 bool true=参与、eval 要显式 dropout_p=0、内核选择可能影响浮点结果；这些事实用于定义未来优化边界，首批仍采用显式路径。torchvision 0.21 `vision_transformer` 的 EncoderBlock/Encoder/VisionTransformer 用于核对 pre-norm、位置/cls 参数和 patch 流程；本设计的小型变体初始化/尺寸已单独标明。

```text
https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html
  定位：文档公式、dropout warning、attn_mask 参数、implementation notes。
https://docs.pytorch.org/vision/0.21/_modules/torchvision/models/vision_transformer.html
  定位：EncoderBlock.forward、Encoder.pos_embedding、VisionTransformer._process_input/forward。
```
