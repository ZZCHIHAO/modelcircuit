# S5 Transformer and Minimal TaskAdapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 只有用户或当次适用指令明确授权时才使用 subagent-driven-development；本次仅编写文档，不派发实施、不修改生产代码。

**Goal:** 在 ModelCircuit Studio｜模路 中提供可展开 Attention/Encoder/小型 ViT、静态 Python 往返及可观察的现有图片分类闭环，最后提取最小 TaskAdapter。

**Architecture:** 新 schema v2/tensor-v1 把张量、轴、mask、参数源与资源策略分离；有限算子驱动普通芯片、独立 Torch 参考和原生目录。ViT 先走 classification-v2；adapter 最后提取，不引入通用工作流引擎。

**Tech Stack:** Python、DearPyGui、现有 PyTorch CPU/CUDA 环境、unittest；官方语义参考固定 PyTorch 2.6 / torchvision 0.21。

**Spec:** `docs/superpowers/specs/2026-11-18-s5-transformer-task-design.md`。承接 S4 发布计划 Task D；`v0.5.0-alpha.1` 已交付 T0 的后端静态范围，后续从 T1 开始。

**2026-11-18 执行结果：** `v0.5.0-alpha.2` 交付 S5 源码阶段。实际通过的命令、用例和未覆盖范围见 [S5 验收记录](../../../studio/native/S5_TRANSFORMER_VERIFICATION.md)；下文复选框保留原规划，不作为实测报告。本轮将算子声明、推导与构造集中在 `tensor_ops.py`，以现有分类配置承载运行快照中的采样请求，分类适配接口仍为内部接口。未重新冻结 EXE。

## Global Constraints

- Windows 原生 DearPyGui 桌面；禁用 OS/浏览器输入自动化。允许应用自身回调及 framebuffer。
- 不安装依赖、下载权重/外部数据或长训练；未来验收只用本地生成小样本和现有 CPU/CUDA 环境。当前 Task D 不训练、不构建、不执行本计划。
- 保留原 `ml_forge/`、根 LICENSE、旧 S3/S4 EXE/ZIP；本计划不授权 Git 暂存/提交/分支/推送或发布。执行期由当次用户指令决定集成方式。
- 新图 `schema_version=2`、`capability_profile=tensor-v1`、`resource_profile=hardware-v5`；`resource_batch_bound` 仍为 1..64；旧 v1–v5 profiles 原行为保持。
- hardware-v5 没有固定参数/单张量/累计执行元素上限；硬件容量 admission 与文件/节点/递归/解析工作量边界独立。预测峰值不等于精确硬件上限，不静默改设备或缩模型。
- 首批只有 float32/bool，固定序列长；Embedding/int64 模型端口、生成、3D、多 GPU、AMP、完整任务关系图与任意 Python 导入不在本期。
- Python 只接受有限静态 AST，源码优先，metadata 不覆盖计算语义；不执行用户导入源码。
- 新模板默认独立实例；显式共享保持一个状态/参数集合、多 call 观测路径。实测、估算和未知必须清楚区分。
- S4 冻结交付先通过。S5-A/B/C 各 Gate 保存真实证据，未通过的能力不标为可用。

## 文件与接口地图

下列为未来实施的预定文件，当前均不在 Task D 修改范围。已有文件跟随职责扩展，不把新子系统塞进 ui.py 或 schema.py 的大分支。

| 文件 | 单一职责 |
|---|---|
| 新 `studio/backend/studio_core/tensor_specs.py` | 无 Torch TensorSpec、轴/batch/dtype 纯验证、字节宽度；v1 内部规范化。 |
| 新 `tensor_operators.py`、`tensor_rules.py`、`tensor_runtime.py` | 有限新算子声明、纯类型推导、Torch 执行与状态；不互相反向依赖。 |
| 修改 `schema.py`、`operators.py`、`operator_rules.py`、`runtime.py` | 显式版本分发/旧契约保留、v2 registered kwargs 派发与兼容 shapes 投影。 |
| 新 `tensor_python.py`；修改 `python_io.py` | 独立 helper 源码、有限注解/构造解析与 tensor-pytorch-v1 分发。 |
| 新 `transformer_templates.py` | 无 Torch 纯图 Attention/Encoder/小型 ViT 工厂。 |
| 修改 `model_registry.py`、`component_catalog.py`、`component_types.py` | 新稳定 ID 与 capability/schema requirements；保持 ComponentBundle。 |
| 新 `probe_capture.py`；修改 `memory_admission.py`、`execution_policy.py`、`classification_runtime.py` | copy 前预算、requested sources、dtype/storage/workspace 与观察上下文。 |
| 新 `task_adapters.py`、`classification_task.py` | 有限协议/内置注册、现有分类 loss/batch/metric 职责。 |
| 修改 `experiment_spec.py`、`worker.py` | 新图完整身份与 checkpoint v3/任务版本；保持旧图签名和 worker 生命周期。 |
| 修改 `studio/native/document.py`、`document_editing.py`、`chip_library.py`、`local_components.py`、`component_insert.py`、`workspace_state.py` | v2 图/格式3芯片包、原子升级/撤销、有限探针选择保存。 |
| 新 `studio/native/tensor_panel.py`、`attention_probe_panel.py`；修改 `catalog_panel.py`、`chip_panel.py`、`ui.py`、`canvas_view.py` | 原生有限类型/轴配置与 attention slice；现页签/分屏和库保持。 |
| 新 `studio/backend/tests/tensor_fixture.py`、`transformer_reference.py`、`transformer_acceptance.py` | 测试图工厂、无平台依赖的独立 Torch 参考、CPU/CUDA 证据入口。 |
| 新相应 `test_tensor_specs.py`、`test_tensor_operators.py`、`test_transformer_templates.py`、`test_tensor_python.py`、`test_attention_capture.py`、`test_task_adapters.py` 及原生测试 | 每阶段行为回归；新旧协议负向路径。 |

## 公共拟定类型

以下代码是待实现的接口契约，**不是本轮已存在 API**。所有后续任务使用这些名称，不能各自发明近似字段。

```python
# tensor_specs.py: JSON TypedDict；dtype 字符串不导入 Torch。
class TensorSpec(TypedDict):
    shape: list[int | None]
    dtype: Literal['float32', 'bool']
    axes: list[str]
    batch_axis: Literal[0] | None

def normalize_tensor_spec(value: dict, *, legacy: bool = False) -> TensorSpec: ...
def tensor_nbytes(spec: TensorSpec, *, batch_size: int) -> int: ...
def infer_tensor_operator(kind: str, params: dict,
                          inputs: dict[str, TensorSpec]) -> TensorSpec: ...
def make_tensor_operator(kind: str, params: dict) -> nn.Module: ...

# probe_capture.py：均为有限 JSON；不序列化 Tensor。
class ProbeRequest(TypedDict):
    source: str
    sample_index: int
    head_index: int
    query_range: list[int]
    key_range: list[int]
    statistics: list[str]  # 有限 mean/min/max/norm

class CapturePolicy(TypedDict):
    requests: list[ProbeRequest]
    max_sources: int       # 16
    max_bytes: int         # 1_048_576
    max_attention_side: int  # 64

def capture_tensor(value: Tensor, spec: TensorSpec, request: ProbeRequest,
                   *, remaining_bytes: int) -> dict: ...
```

`capture_tensor` 返回 `shape,dtype,axes,selection,values,statistics,captured_bytes,omitted,truncated`；omitted 时 values=[]。`capture_tensor` 自身先索引/归约再复制，CPU/GPU 转移均在预算之后；sample/head 越界报可定位错误，不截到“最近有效值”。`infer_tensor_operator` 数值参数计数由对应注册声明单独处理，与原 infer_operator 适配时保留旧返回形状/参数计数协议。

## 阶段依赖与 Gate

`T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9`。T4 的采集/容量必须先于 T5 的真实训练；Python/状态在上目录前通过。阶段 S5-A=T0–T3，S5-B=T4–T8，S5-C=T9。后续回归/多头另行规格，不由本计划自动开始。

### T0：锁定兼容基线并引入 v2 TensorSpec（G0）

执行记录：复用 S4 已冻结的兼容夹具；新测试使用本文件夹内 `test_tensor_specs.py` 的小图构造器，避免提前增加未使用的算子夹具。完整 CPU backend 跑 328 项（327 通过、1 权限跳过），native 265 项通过。`tensor-v1` 此时只承诺静态检查；执行、Python 转换与桌面编辑等待后续对应关口。

**Files:** Create `tensor_specs.py`、`tests/tensor_fixture.py`、`tests/test_tensor_specs.py`；Modify `schema.py`、`experiment_spec.py`。基线证据位置 `schema.py:178/:211/:434`、`python_io.py:129`。

**Interfaces:** 输入为现有 v1 graph 或显式 v2 graph；产生 `normalize_tensor_spec`、`tensor_nbytes` 与 validate_graph 报告的 `tensor_specs`，原 `shapes` 保留。`tensor_fixture.operator_graph(kind, params, inputs)` 返回 schema2/tensor-v1/hardware-v5/batch_bound2、一个 op 实例/一个 apply call/一个 out 输出，inputs 为 `{id:TensorSpec}`；`legacy_case()` 返回既有合法 float32 图。

- [x] 保留 S4 的 profile Python、model_digest、classification checkpoint/signature 和 workspace 夹具，完整回归旧核心、命名端口和硬件策略；原黄金结果未改动。
- [x] 在 `test_tensor_specs.py` 添加入参和图契约正负例，先运行确认缺失 API 的预期红灯。

```python
def test_spec_and_boolean_bytes(self):
    spec = normalize_tensor_spec({'shape': [None, 65, 64],
        'dtype': 'float32', 'axes': ['batch','token','feature'], 'batch_axis': 0})
    self.assertEqual(tensor_nbytes(spec, batch_size=2), 2*65*64*4)
    mask = normalize_tensor_spec({'shape': [1,1,65,65], 'dtype': 'bool',
        'axes': ['mask_batch','head','query','key'], 'batch_axis': None})
    self.assertEqual(tensor_nbytes(mask, batch_size=2), 65*65)

def test_dynamic_token_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'batch'):
        normalize_tensor_spec({'shape': [None,None,64], 'dtype':'float32',
            'axes':['batch','token','feature'], 'batch_axis':0})
```

- [x] 实现纯校验：dtype 白名单、axes 唯一/rank、batch 0/null 与 None 位置、固定 B 标记、rank 1..8；布尔输入不进入浮点算子，无 batch 源不得接受动态维。保留原图校验工作量限制。
- [x] 保留 v1 prepare/摘要路径；v2 只接受 tensor-v1/hardware-v5。旧 graph 不自动迁移，新增 `tensor_specs` 为派生报告；未支持的执行与导出入口明确拒绝 v2。
- [x] 定向测试 9 项通过；完整 CPU 后端回归覆盖旧黄金，完整桌面测试通过。

### T1：有限张量算子与参数源（G1）

**Files:** Create `tensor_operators.py`、`tensor_rules.py`、`tensor_runtime.py`、`tests/test_tensor_operators.py`；Modify `operators.py`、`operator_rules.py`、`runtime.py`、`schema.py`。依据 `operator_rules.py:108`、`runtime.py:108`。

**Interfaces:** 产出公共 `infer_tensor_operator`、`make_tensor_operator`；新 kind 和端口精确遵循规格第5节。ParameterTensor 的参数键 `.value`，CausalMask 的持久 buffer 键 `.keep`。v2 runtime 使用注册端口校验后的 `component(**arguments)`，v1 路径兼容保留。

- [ ] 用 T0 夹具添加每个新 kind 的形状/dtype/非法参数断言，覆盖 Reshape 保 B/元素守恒、非 contiguous、Permute 重复/移动 B、MatMul 前缀不等、Softmax batch 轴、Select 越界、参数源零输入、bool 无梯度。分别运行失败测试，确认不是环境导入错误。
- [ ] 加入广播反向的具体判据：

```python
value = torch.randn(1, 3, 4, requires_grad=True)
anchor = torch.empty(2, 3, 4)
module = make_tensor_operator('BroadcastTo', {'shape':['B',3,4],
                             'axes':['batch','token','feature']})
actual = module(x=value, like=anchor)
actual.sum().backward()
torch.testing.assert_close(actual, value.detach().expand(2,3,4))
torch.testing.assert_close(value.grad, torch.full_like(value, 2.0))
```

- [ ] 实现有限声明/类型推导和 Torch wrapper；添加 tensor-v1 独占 Linear.bias、LayerNorm.eps/elementwise_affine 参数。零输入源按实例构造一次，forward 不初始化；冻结保留 `requires_grad` 与 state，不从 optimizer 删除非冻结共享参数两次。
- [ ] 验证实例/调用：一个 ParameterTensor 连两个 call 保持 Python object identity 与单次参数计数；复制实例不同对象；各 call 输出观测独立。MaskApply 的 -inf 类型流仅允许连接 Softmax(-1)，全遮蔽 query 行前向明确失败；普通非有限检查不放宽。
- [ ] 将实际结果与测试内独立 Torch 表达式比较，前向/输入和参数梯度/SGD/strict state reload；不使用 infer 函数生成参考。运行 `test_tensor_operators.py`、`test_vision_operators.py`、`test_public_types.py`、`test_paired_runtime.py`、`test_vision_budget.py`。记录 G1。

### T2：独立 Attention 参考与纯图模板（G2）

**Files:** Create `transformer_templates.py`、`tests/transformer_reference.py`、`tests/test_transformer_templates.py`、`tests/transformer_acceptance.py`；Modify `schema.py` 仅在定位真实跨芯片类型错误时。

**Interfaces:** 生产 `make_attention` 精确签名见规格第7节，返回 ComponentBundle，具名 features/可选 attention；测试参考 `AttentionReference(hidden_dim, num_heads, dropout=0.0)` 的 `forward(x, keep=None)->dict` 返回 features/attention。参考不 import studio_core；`reference_state_map(graph, model, reference)` 由测试显式配置键路径，禁止遍历图造层或 zip 状态。

- [ ] 先写独立 reference：Q/K/V/out 四个 nn.Linear，手写 head reshape/transposition，scores=q@k.T/sqrt(d)，keep false→-inf，softmax/dropout/@v。参考 forward 同样检查每行 keep，但不调用平台 helper。
- [ ] 为 T=1/5、D=8、heads=1/2/4，B=1/2 构造无 mask、causal keep、外部 keep 的闭包测试。对以下数值断言先运行红灯：

```python
# test 函数内已按显式 state map 复制相同权重；dropout=0。
x_a = torch.randn(2,5,8, device=device, requires_grad=True)
x_b = x_a.detach().clone().requires_grad_(True)
out_a = model(x=x_a)['features']
out_b = reference(x_b)['features']
torch.testing.assert_close(out_a, out_b, atol=atol, rtol=rtol)
out_a.square().sum().backward(); out_b.square().sum().backward()
torch.testing.assert_close(x_a.grad, x_b.grad, atol=grad_atol, rtol=grad_rtol)
# 对显式 state map 的每一对参数再逐项比较 grad；随后相同 SGD(lr=.01) step。
```

- [ ] 实现普通图工厂，Q/K/V 默认独立；所有 shape/axes/Scale、masked 开关与 bool 接口可纯验证，expected output axes 用 AxisRename 明示。Attention 权重输出指实际 Softmax，不重算。
- [ ] 扩展测试为全部参数梯度/SGD 后参数、state reload、共享 block/独立 copy、错 state key/shape、mask polarity 负向；非零 Dropout train/eval 和稳定调用序测试与数值对照分开。
- [ ] `transformer_acceptance.py` 定义 CLI `--device cpu|cuda:N --section attention|vit|all --output PATH`，按规格容差保存设备/版本、配置、最大绝对相对误差、每参数映射与通过数；CUDA 不可用返回非成功并标阻断，不隐式 CPU fallback。先运行 `test_transformer_templates.py`，CPU 通过后顺序跑现有可用 CUDA；当前 Task D 不执行这些命令。

### T3：静态 Python tensor-pytorch-v1 往返（G4 的基础）

**Files:** Create `tensor_python.py`、`tests/test_tensor_python.py`；Modify `python_io.py`。依据 `python_io.py:342/:715/:865`。

**Interfaces:** 保持 `export_python(graph)->str`、`import_python(code)->dict`；内部 v2 由 `tensor_python` 提供 helpers、有限注解 parser 与精确 helper AST。新 profile 名为 tensor-pytorch-v1；导出 standalone 不 import studio_core。

- [ ] 写失败测试：attention 图有/无 STUDIO_VISUAL_METADATA 导入都保完整 dtype/axes/B/ParameterTensor/init/state/share；接受规格第8节的新注解，拒绝缺 axes、动态 token、bool 算术、任意函数调用。
- [ ] 实现普通 ModuleDict/helper 源码与 v2 根 execution guard；语义参数来自构造代码和输入注解。延续 2 MB、AST/文档工作量边界；精确 helper 检查同时验证真实全局绑定，不通过白名单名字猜安全。
- [ ] 加入源码优先测试：保留旧 metadata，编辑 Linear out_features 并同步合法 Reshape；改变两个 call 的 instance_id、修改 masked keep 连线；重新导入与执行应产生编辑后的行为。过期 metadata 无权回滚计算图；不支持的动态编辑须含行号拒绝。
- [ ] 对 source helper 篡改/别名遮蔽/重复全局、state missing/shape mismatch、metadata 无效字段进行负向测试；保留用户源码草稿直到纯验证成功，不半导入。
- [ ] 在独立子进程仅提供导出文件+Torch，构造模型、strict load state、比较前向/输入与参数梯度/SGD；运行 `test_tensor_python.py`、`test_python_io.py`、`test_vision_python.py`、`test_vision_policy.py`。记录 G4 基础，不把 Attention 通过扩称 ViT 已通过。

### T4：copy 前探针预算与 attention 硬件估算（G5）

**Files:** Create `probe_capture.py`、`tests/test_attention_capture.py`；Modify `runtime.py`、`classification_runtime.py`、`memory_admission.py`、`execution_policy.py`。依据 runtime.py:100/:135 与 memory_admission.py:90。

**Interfaces:** 产出公共 ProbeRequest/CapturePolicy/capture_tensor；`build_model(graph, device='cpu', capture=True, *, memory_report=None, capture_policy=None)` 新增 keyword-only 参数；v1 capture 缺省行为不变，v2 capture=True 缺省使用有限策略。`estimate_memory` 保持原 signature/results，v2 附加 dtype/attention/probe 细项；`run_classification(..., capture_policy=None)` 接收已冻结请求。

- [ ] 写预算/错误测试：attention 请求≤16，slice≤64×64，累计≤1MiB；使用大 T 的**小测试张量**和 mock clone/CPU transfer 记录调用顺序，证明未请求源零克隆、slice 先于 clone、remaining_bytes=0 返回 omitted，不用真实大分配证明预算。
- [ ] 实现请求来源解析与预算扣减；只对实际运行中产生的张量切片，禁止重算 attention。统计采用在线/受控归约，统计临时成本纳入估算；重复 alias 不重复复制，同实例不同 call 保留来源路径。
- [ ] 将 float32/bool、ParameterTensor/buffer、expand alias、reshape 可能副本、BHT² scores/probs/dropout/autograd/capture/传输及现有 checkpoint 副本分项纳入估算；保守项清楚标估计，不把 sum-elements 当真实峰值。最低存储只含可证明必要项，不能把保守 workspace 下界说成必需。
- [ ] 模拟设备查询失败/容量不足/并发变动 OOM，断言分配前错误和不切设备；能推导大参数图而不触发构造，确保无旧 20M cap 复活。保留文件/节点/解析边界测试。
- [ ] 对照实际切片值、样本/head/range、mode/checkpoint epoch/last-training-minibatch 文案、取消无半结果和 stale 结果清除。运行 `test_attention_capture.py`、`test_hardware_policy.py`、`test_classification_runtime.py`、`test_paired_runtime.py`；CPU 后顺序 CUDA，记录真实峰值若测量可用，缺失时标 unknown。

### T5：Encoder 与小型 ViT，先通过现有分类训练（G3）

**Files:** Modify `transformer_templates.py`、`tests/transformer_reference.py`、`tests/test_transformer_templates.py`、`tests/transformer_acceptance.py`；新增 `tests/test_vit_classification.py`。此任务先不改 TaskAdapter。

**Interfaces:** `make_encoder`、`make_small_vit` 与规格第7节相同；`EncoderReference(...)`、`SmallViTReference(...)` 手写 Torch 层，forward 返回 features/logits；`transformer_acceptance --section vit` 新增 Encoder/ViT 全部测试。现 `run_classification`/manifest/config 接口保持。

- [ ] 编写结构测试：patch/正方形/整除、D%heads、depth/MLP/class 合法；默认 32×32、patch4、T65、D64、heads4、depth2；cls/pos 是训练参数；各 Encoder block 默认不同实例，显式同实例两 call 参数只计一次。
- [ ] 手写独立 Encoder/ViT reference，pre-norm 与最终 LN、cls/pos、Select 和读出明确；显式 state mapping（包括 Q/K/V、LayerNorm eps、classifier）逐项严格。数值前向、输入/全部参数梯度、一步 SGD、reload 先红后绿；不从模板反推 reference。
- [ ] 实现普通图工厂，输出 logits=[B,C]，默认初始化差异在模型信息记录；非默认 Dropout 独立验证 train/eval/RNG，不把零初始化 classifier 的梯度现象误报模板错误。
- [ ] 用已有生成 image_folder fixture 创建本地极小 train/val/test；通过实际 RunManager→worker 一轮分类，证明参数更新、结果、prediction、checkpoint、probe context 和 worker exit 成功；同设备一轮+resume 对比连续训练与 seed/shuffle/state。
- [ ] 通过 T3 导出整个 ViT，有/无 metadata 与源编辑路径重复 G4；执行 `test_vit_classification.py`、完整 `test_transformer_templates.py`、CPU/CUDA `transformer_acceptance --section vit`。保存测试/版本/误差和结果哈希，短训练不用于效果宣传。

### T6：格式3本地芯片、注册目录与原子迁移（G0 扩展）

**Files:** Modify `component_types.py`、`model_registry.py`、`component_catalog.py`、`studio/native/chip_library.py`、`local_components.py`、`component_insert.py`、`document.py`、`document_editing.py`；新增 `tests/test_tensor_components.py`、`studio/native/tests/test_tensor_component_insert.py`。

**Interfaces:** `ComponentBundle(entry,definitions,requirements)` 不变；requirements 新字段 schema_version=2/capability_profile=tensor-v1/operator_set/resource_profile=hardware-v5/resource_batch_bound。稳定 ID 精确为 tensor.attention/tensor.encoder/vit.small_classifier；chip_format_version=3 新格式包含完整新需求。旧1/2包读写协议不改。

- [ ] 先测旧目录数量/既有14模型 stable ID 未被替换、三个新条目只在功能已验后返回、配置字段与图闭包一致；注册模块 import 不导入 Torch/DPG、不构造权重。
- [ ] 实现注册工厂/配置/能力说明，参数非法在放置前失败；component requirements 描述功能与容量，两种能力不得混成旧资源 profile 自动降级。
- [ ] 写并实现 v2 工程接纳旧 float32 芯片的内部规范化；旧工程放置新芯片需显式升级交互，升级+插入作为一次可撤销动作。共享/独立、名称冲突、损坏包、缺需求、未知格式、失败回滚测试齐全。
- [ ] 图2新包导出/重新本地扫描/放置/展开/Python 往返通过；旧包原文件哈希不变。无法解析新格式的旧读取器应失败而非覆盖源件；保护失败读取的 workspace 自动保存。
- [ ] 运行 `test_tensor_components.py`、`test_component_catalog.py`、`test_backbone_templates.py`、原生 `test_tensor_component_insert.py`、`test_local_components.py`、`test_component_insert.py`、`test_storage_retry.py`。

### T7：真实原生配置、连线与 attention 探针（G6 的桌面部分）

**Files:** Create `studio/native/tensor_panel.py`、`attention_probe_panel.py`、`tests/test_tensor_panel.py`、`tests/test_attention_probe_panel.py`；Modify `catalog_panel.py`、`chip_panel.py`、`canvas_view.py`、`ui.py`、`workspace_state.py`、`workbench_state.py`；仅必要时修改 experiment_panel.py 接入已冻结请求。

**Interfaces:** `TensorPanel(ui)` 管理有限 TensorSpec 输入/轴校验回调；`AttentionProbePanel(ui)` 生成 ProbeRequest 并显示 capture_tensor 结果；二者只通过 document/manager 和现有 guarded callback，不持有模型。workspace v3 新增可选 `attention_probe_requests` 有限 JSON；旧文件无字段可读。

- [ ] 回调单测先红：新 dtype/axes/batch 面板可编辑且无效输入保留草稿；目录配置 heads/patch 不合法能定位；source port 的类型不兼容拒绝连线且不破坏旧图/undo。
- [ ] 接入左库真实分类/搜索/配置/放置，双击调用展开在现 workbench 页签；共享实例显示调用数量，复制独立实例可测参数 identity。不要再建第二套导航。
- [ ] 实现 attention head/sample/range 与 heatmap/数值 slice、原始 shape/截断/epoch/mode/sample_ids/设备和梯度出处；点击已有实测结果只改变显示选择，训练前 ProbeRequest 冻结，未采集区显示 omitted/重新运行需要，不能伪造当前值。
- [ ] 保存/重开 workspace v3：页签/分屏/接口/有限请求恢复，Tensor/DPG handles 不进 JSON；已删 source 和无结果显示尚无本次结果，旧结果 stale 标记正确。实验界面不出现空的回归/多头/文本任务项。
- [ ] 运行新原生单测和既有 `test_workspace_state.py`、`test_workbench_persistence.py`、`test_workbench_locations.py`、`test_chip_editing.py`、`test_native_experiment.py`；真实 framebuffer 在 T8 执行，mock 通过不等于手势或可见布局已验。

### T8：原生端到端 Gate 与交接证据（G0–G6 汇总）

**Files:** Create `studio/native/s5_acceptance.py`、`s5_render_acceptance.py`、`S5_TRANSFORMER_VERIFICATION.md`、`S5_TRANSFORMER_GUIDE.md`；新增相应 `native/tests/test_s5_acceptance.py`，修改项目 README 仅在阶段实际通过后声明能力。

**Interfaces:** 诊断入口沿 S4 自身 callback/framebuffer/run manager 模式；`python -m studio.native.s5_acceptance --device cpu|cuda:N --output PATH`，`python -m studio.native.s5_render_acceptance --output PATH`。新增 flags 仅在实现后可用，本计划不表示当前可调用。

- [ ] 设计 fixture 与预期快照：image_folder 检查→配置 ViT→放置→展开 Encoder/Attention→类型错误定位→修复→Python 有/无 metadata 导入→实际 worker 分类→查看 attention slice→保存→关闭→重启→恢复；另测取消/损坏输入保护。
- [ ] 用程序实际注册的回调完成上述路径，framebuffer 检查默认和最小支持窗口的左库、页签、配置、运行取消、错误、slice 可见；不使用 OS 输入自动化，不声称物理拖拽已验。
- [ ] 顺序执行 CPU 与已存在 CUDA 环境原生/后端 Gate；有 CUDA 构建仍需显式 CPU 选择路径。CUDA 不可用、query failure 或渲染不可用写精确阻断；不只看测试汇总 pass。
- [ ] 在现有解释器、正确 `PYTHONPATH=studio/backend` 下运行后端完整 unittest discovery 与 native 完整 discovery；同一 Gate 无新改动不反复扩大测试。收集退出码、错误上下文、实际 worker exit、checkpoint 读取、PNG、命令和哈希。
- [ ] 更新指南/验证表，能力分列图编辑/执行/反向/状态/Python/桌面/设备证据；只有全部必要 Gate 通过才在左库和 README 标记完成。S5 打包/发布由当次交付计划另行处理，不覆盖 S4 产物。

### T9：从分类提取最小 TaskAdapter（S5-C；G6 的语义部分）

**Files:** Create `task_adapters.py`、`classification_task.py`、`tests/test_task_adapters.py`；Modify `classification_runtime.py`、`experiment_spec.py`、`worker.py`、`tests/test_classification_runtime.py`、`tests/test_experiment_spec.py`、`tests/test_vit_classification.py`。

**Interfaces:** 定义 `Batch` TypedDict（features:dict[str,Tensor]、targets:Tensor、sample_ids:list[str]），`BatchStats` TypedDict（loss_sum:float、correct:int、count:int）与规格第10节完整 TaskAdapter Protocol；`get_task_adapter(identifier: str, version: int)->TaskAdapter` 只接受 classification/1。`ClassificationTask` 的构造接收已验证 manifest/config，validate_contract/bind_batch/loss/summarize/finalize 签名与规格一致。优化器/循环/文件留 runtime。

- [ ] 先冻结提取前分类行为：csv/image_folder/paired，CPU/CUDA 参数与 RNG、metrics/predictions/probe_context、旧 model_digest/checkpoint version2/signature 与恢复；这些是抽取回归 oracle，不能替代独立 Torch oracle。
- [ ] 写 loss/metrics 测试并先红：

```python
adapter = get_task_adapter('classification', 1)
logits = torch.tensor([[2.0,0.0],[0.0,2.0]], requires_grad=True)
batch = {'features': {}, 'targets': torch.tensor([0,1]), 'sample_ids':['a','b']}
loss = adapter.loss({'logits':logits}, batch)
torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, batch['targets']))
stats = adapter.summarize({'logits':logits}, batch, loss)
self.assertEqual(stats['correct'], 2)
self.assertEqual(adapter.finalize(stats)['accuracy'], 1.0)
```

- [ ] 提取 batch 映射、logits/target 校验、CE 和加权统计，保持现有 loader、预测输出和异常上下文；未知 id/version、bool logits、class 不符、target rank/dtype/范围错误在 optimizer.step 前失败。adapter 不调 optimizer、不读文件或 GUI。
- [ ] 旧 schema1/checkpoint v2 path 原签名不变；tensor-v1 新 checkpoint version3 的 identity 纳入 adapter id/version 与 tensor 图语义，仍 classification-v2 workflow。写旧/新独立恢复、交叉恢复拒绝、resume 不一致 task/data/graph 拒绝测试。
- [ ] 以冻结 oracle 对比提取前后同设备训练/恢复逐步结果；重复 `test_task_adapters.py`、`test_experiment_spec.py`、`test_classification_runtime.py`、`test_paired_runtime.py`、`test_vit_classification.py` 和相关 worker/native 状态测试。实际 ViT 分类再短跑一轮只为验证抽取未改变循环。
- [ ] 在 S5 验证文档标明“仅 classification adapter 已证明”；后续回归/多头先写独立规格，至少完成非 CE loss、不同 targets/outputs、实际数据/metric/checkpoint/UI 和独立 Torch 对照，才允许宣称跨任务扩展。本计划到此结束。

## 执行命令与证据约定

仓库测试导入 `studio_core`，执行前在**本次进程**设置 `PYTHONPATH`，使用 S4 已验证解释器，禁止裸 `python` 意外换环境。PowerShell 示例（实施时把解释器路径替换为当次已核实的实际路径，不据此安装）：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'studio/backend')
& $VerifiedPython -m unittest discover -s studio/backend/tests -p test_tensor_specs.py -v
& $VerifiedPython -m unittest discover -s studio/backend/tests -p test_transformer_templates.py -v
& $VerifiedPython -m unittest discover -s studio/native/tests -p test_tensor_panel.py -v
# 完成各阶段实现后才存在：
& $VerifiedPython studio/backend/tests/transformer_acceptance.py --device cpu --section all --output $CpuEvidence
& $VerifiedPython -m studio.native.s5_acceptance --device cpu --output $NativeEvidence
```

执行器先从 S4 交接记录核实并赋值 `$VerifiedPython`、`$CpuEvidence`、`$NativeEvidence` 为真实绝对路径，记录进报告；示例变量不是待交付产物名称。每条运行记录 exit code，非零即 Gate 未通过；独立命令不要用输出分隔符拼成长串。CUDA 命令仅替换显式 device，顺序运行，保留独立日志。测试源码可集中 helper，但 oracle 禁止依赖生产图/规则/runtime。

证据目录由实施时创建 `.superpowers/sdd/2026-11-18-s5-transformer-task/`，按 T0..T9 分报告；本轮不创建其实现证据。每任务报告列实际文件/命令/退出码/设备/误差/未验证项，Gate 对照规格矩阵。无实际运行证据不得勾选完成；无固定日期工期估算，不把未知硬件性能写成承诺。

## 文档自检与实施起点

- [ ] 执行前重新核对 S4 发布完成状态、现工作目录和最新接口；父阶段真实改动如改变本文定位，只修复相关引用，不重写已批准范围。
- [ ] 每任务完成后做对应 Gate 的代码/测试审查，再推进依赖任务；只有明确授权才并行代理或集成 Git。当前文档任务无此动作。
- [ ] 依次核对 TensorSpec/参数/掩码/随机状态、Python 源码优先、copy 前预算、hardware admission、原生流程、适配提取和兼容测试，缺项添加具体任务/断言后再执行。

本计划及设计文档已形成可审阅的后续工作定义；尚未创建任何 S5 生产模块、算子、训练结果或发布包。
