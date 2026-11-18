# S4-A 基础组件、融合芯片与ResNet18实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可检索、可展开、数值和状态可核对的首批科研组件库。

**Architecture:** torch-free算子/目录描述供schema、文档和UI使用；运行构造单独注册；功能与ResNet18由普通图闭包生成。新profile把资源策略与受控源码往返连接，旧图兼容行为保持。

**Tech Stack:** 现有标准库、torch2.6和unittest；图工厂无torch/DPG依赖；不安装torchvision或下载权重。

**Spec:** [S4规格](../specs/2026-11-15-s4-workbench-components-design.md)，第4节为本子计划权威契约。

2026-11-17执行追记：A2源码、策略桥接与两轮独立范围复审已通过，整体审查两P2已合并修复，最终CPU backend251项运行（250通过、1旧权限跳过），退出0；Python v4去metadata、当前源码优先、活跃符号冲突拒绝及新资源账本已验证。A3/A4/A5未实施，ResNet18真实闭包仍为A4关口，不从有限预算算例推断模型已可训练。整轮交叉审查和交付记录见[本轮验证](../../../studio/native/S4_POLICY_CANVAS_VERIFICATION.md)。

同日下一轮用户追加基础组件库可拖放，现已实现A3五工厂并通过17项定向/独立审查，A5提前交付17primitive+5工厂的纯目录、有限配置、原子插入/策略opt-in及闭包冲突处理；两目录P2已修复复审接受。A5完整交付仍缺A4 ResNet18及后续全库集成，不将本切片勾成完整A5。ComponentBundle迁至component_types，由component_catalog重导出，原公开接口保持。

## Global Constraints

- 2026-11-15仅交付规划；2026-11-16已完成A0调查及A1限定算子核心实现/审查，A2–A5仍未实施。完成状态以实际证据勾选，按总计划约束执行。
- 旧S1/S2/S3图、checkpoint、model_digest和Python profile回归必须保留。
- 组件必须具备声明的形状、执行、Python和状态能力，不以目录名字代替实现。
- Multiply严格同形；Scale.factor固定有限标量；BN frozen仅冻结参数，统计量服从train/eval。
- vision-v4默认batch上界2、范围1..64；旧profile仍按64；不全局放宽预算。
- ResNet18标准宽度与[2,2,2,2]拓扑必须真实生成；首版无预训练权重功能。
- 原生UI、无电脑控制、GitHub暂停；不自动提交/创建worktree。

## 文件结构与依赖

新增 `operators.py/operator_rules.py/operator_runtime.py` 分离声明、数学验证和torch执行；`module_templates.py/backbone_templates.py` 只生成JSON；`component_catalog.py` 负责描述、检索和工厂分派。精确接口均在规格4.1。

A0→A1→A2→A3→A4→A5；A5的纯目录基础可在A1后先写，但完整目录只能注册已完成的组件。UI接线由总计划I1负责，不在A线同时修改ui.py。

2026-11-16：A0限定调查与17项定向测试已完成，见 `docs/评审附件/S4-D0核心验证.md`。候选新策略可行性未通过；A2首个动作须处理重复推导与definition默认校验计费，不得仅实现batch_bound后声称可接入ResNet。A1可独立继续。

A1同日已完成，27项定向与旧回归、CPU/CUDA微型参考检查、独立复审接受；分配前BN持久buffer账本补齐，Python v4/新算子导出仍未实施。证据见 `studio/native/S4_FOUNDATION_VERIFICATION.md`。

## Task A0：预算、状态和架构的前置报告

**Files:**
- Create: `studio/backend/tests/test_vision_budget.py`、`docs/评审附件/S4-D0核心验证.md`（执行后才创建）。
- Read: `schema.py:_elements/_Budget/_infer`，`runtime.py:_GraphModule`，`python_io.py:python_profile`，`chip_library.py:validate_bundle`。
- Produces: 新档是否可行的算例、闭包/节点规模、BN状态清单和限制表；不对CPU/GPU性能作预测。

- [x] 记录当前旧档的参数、单张量、累计元素、节点/定义/嵌套/文件大小上限，以及保存/导入的严格字段。
- [x] 写固定算例并保留未来API红灯，明确当前失败原因（红灯隔离在SDD，正式17项测试全绿）：

```python
def test_explicit_batch_bound_changes_only_unknown_batch_budget(self):
    from studio_core.schema import resource_limits, bounded_elements
    limits = resource_limits('vision-v4', batch_bound=2)
    self.assertEqual(bounded_elements([None, 64, 112, 112], limits), 1_605_632)
    self.assertEqual(resource_limits('classification-v2').max_parameters, 20_000_000)
    self.assertEqual(bounded_elements([None, 64, 112, 112], resource_limits('classification-v2')), 51_380_224)
```

- [x] 将 `bounded_elements(shape, limits)` 与 `resource_limits(profile=None, *, batch_bound=None)` 接口加入后续A2契约；确认定向失败是接口缺失，保存红灯证据。本项不是接口已实现。
- [x] 对照规格/官方资料列出ResNet18各stage输出公式与标准参数计数，设计每个block不同instance的命名清单。独立审查者已复算状态和预算。
- [x] 将224输入、bound2、累计嵌套计费和芯片合成验证根列为A2/A4的必过测试；报告明确纸面算例不是模型已执行。

未来测试命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_vision_budget.py -v
```

## Task A1：统一算子描述、BatchNorm及基础数学算子

**Files:**
- Create: `studio/backend/studio_core/operators.py`、`operator_rules.py`、`operator_runtime.py`。
- Create tests: `studio/backend/tests/test_vision_operators.py`。
- Modify: `schema.py`白名单/参数/端口/推导，`runtime.py`构造/分配前预算；native默认值入口待A5接入，UI入口待I1。
- Consumes: A0限制清单和规格4.2/4.3。
- Produces: `operator_spec/operator_kinds/operator_defaults/normalize_operator/infer_operator/make_operator`，精确签名见规格4.1。

- [x] 写独立数值/梯度测试，至少包含以下实际断言：

```python
def test_scale_multiply_and_complement_have_real_gradients(self):
    import torch
    from studio_core.operator_runtime import make_operator
    x = torch.tensor([[2., -1.]], requires_grad=True)
    y = torch.tensor([[3., 4.]], requires_grad=True)
    product = make_operator('Multiply', {})(x, y)
    value = make_operator('Scale', {'factor': 0.25})(product)
    complement = make_operator('OneMinus', {})(x)
    torch.testing.assert_close(complement, torch.tensor([[-1., 2.]]))
    value.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0.75, 1.]]))
    torch.testing.assert_close(y.grad, torch.tensor([[0.5, -0.25]]))
    self.assertEqual(sum(p.numel() for p in make_operator('Scale', {'factor': 0.25}).parameters()), 0)
```

- [x] 再写BN与原生 `nn.BatchNorm2d` 同权重的train/eval输出、梯度比较，覆盖affine×track_running_stats四组合；track=True时比较running_mean/var/num_batches_tracked，False时断言这些键不在state_dict且eval使用批次统计。共享同实例调用两次与不同实例各调用一次分别核对；frozen仅使affine梯度停止，track=True的train下buffers仍更新。
- [x] 写非法用例：Multiply广播形状拒绝、Scale NaN/Inf/bool拒绝、BN通道不符/eps非正/momentum非法、N*H*W≤1训练失败且不改eval、不跳样本。
- [x] 运行定向测试观察预期失败；实现torch-free描述和数学规则，显式runtime构造；schema按实例对象计数，BN仅affine计2C。
- [x] 更新runtime `_output_elements` 的新同形算子预算；在分配/运行前后均保留限制，不用先计算巨大张量后再报超限。
- [x] 跑定向与原核心/CNN/具名端口回归，核查独立模块中不引入DPG；独立审查状态和共享后记录A1通过。

未来命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_vision_operators.py -v
```

## Task A2：新资源策略、Python v4和组件包兼容

**Files:**
- Modify: `schema.py`、`runtime.py`、`python_io.py`、`experiment_spec.py`、`classification_runtime.py`、`native/chip_library.py`、`native/workspace_state.py`、`native/document.py:_structure`（新策略字段的存储类型检查）。
- Create tests: `backend/tests/test_vision_python.py`、`backend/tests/test_vision_policy.py`、`native/tests/test_chip_requirements.py`。
- Consumes: A1算子语义；A0接口。
- Produces: vision-v4/resource_batch_bound、vision-pytorch-v4源码字面量、芯片包v2 requirements；所有旧格式继续按旧策略。

- [x] 先完成新profile预算口径的修订与失败行为测试：记录编译工作、全部定义校验和根调用路径执行累计元素的独立账本；无界/超限仍拒绝，旧档精确不变。缓存如采用必须有界且仅复用纯shape/参数计算，真实共享调用次数及BN/Dropout效果不能合并。以D0候选树及不同调用次数反例核对；此关口未过，不把后面batch_bound测试的绿灯当作ResNet可行。

- [x] 写以下资源策略往返测试，fixture直接构造一个Identity根图，避免测试依赖尚未完成的ResNet：

```python
def test_policy_survives_python_without_visual_metadata(self):
    from studio_core.python_io import export_python, import_python
    graph = {'schema_version': 1, 'id': 'policy', 'revision': 1,
             'resource_profile': 'vision-v4', 'resource_batch_bound': 2,
             'inputs': [{'id': 'x', 'shape': [None, 3, 224, 224], 'dtype': 'float32'}],
             'instances': [{'id': 'identity', 'kind': 'Identity', 'params': {}}],
             'calls': [{'id': 'identity', 'instance_id': 'identity', 'inputs': {'x': 'input:x'}}],
             'outputs': [{'id': 'out', 'source': 'call:identity'}], 'definitions': []}
    code = export_python(graph)
    # 只移除生成器精确定义的visual metadata语句；源码策略声明必须保留。
    import ast
    tree = ast.parse(code)
    visual_names = {'STUDIO_VISUAL_METADATA'}
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id in visual_names for t in n.targets))]
    restored = import_python(ast.unparse(tree))
    self.assertEqual(restored['resource_profile'], 'vision-v4')
    self.assertEqual(restored['resource_batch_bound'], 2)
```

当前生成器确实使用 `STUDIO_VISUAL_METADATA`（python_io.py:147）。只删除这一语句，不做宽泛文本删除；另断言删后AST中不存在该赋值。另需当前源码把bound2改为3后导入得到3、非法动态值拒绝、旧metadata不得覆盖新源码。

- [x] 增加bound=1/64合法、0/65/bool/float拒绝；vision-v4实际batch超过bound拒绝、末批小于bound允许；配置batch超界在模型构建前失败。另用旧无profile小型Linear图实际batch65证明没有新增硬上限，仍遵循原有元素预算；不把旧分类配置本身的batch上限与任意模型前向混淆。
- [x] 实现ResourceLimits增加默认batch_bound=64用于静态估算；旧构造3个位置参数兼容；新档显式bound只替换未知batch轴，参数/张量/累计硬上限不变，非batch轴None继续拒绝。runtime根输入仅对vision-v4执行actual≤bound检查，导出独立Model遵守相同profile条件，不能给旧profile新增该硬限制。
- [x] Python v4允许受支持的新构造器及精确数学helper；helper源码经AST校验，恶意替换、动态策略、未知调用均拒绝。BN参数和冻结语义由当前源码决定，state_dict键/alias不能重排丢失。
- [x] 新profile/bound加入新模型及checkpoint身份，旧黄金摘要不变。用带BN的小图实测checkpoint完整epoch恢复、严格buffers加载；不以返回成功代替状态核对。
- [x] 实现v2包有限requirements及合成验证根；v1兼容，未知算子集/档位拒绝；导入不自动提高当前工程限制，失败无局部定义/历史写入。
- [x] 运行新测试与旧Python/公共参数/配对恢复/库用例；独立审查“去metadata仍可运行、改变bound拒绝续训、旧图不变”。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_vision_python.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_vision_policy.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_chip_requirements.py -v
```

## Task A3：功能芯片工厂

**Files:**
- Create: `studio/backend/studio_core/module_templates.py`、`studio/backend/tests/test_module_templates.py`。
- Consumes: `ComponentBundle`（在component_catalog.py先定义类型，目录条目晚于工厂注册）、A1/A2。
- Produces: `make_module(template_id, config) -> ComponentBundle`；ID为 `head.mlp`、`fusion.concat_projection`、`fusion.weighted_sum`、`fusion.gated`、`block.residual_basic`。

- [x] 写WeightedSum独立公式测试。测试夹具 `bundle_model(bundle, input_shapes, policy)` 在 `backend/tests/component_fixture.py` 创建：构造单根instance id=`entry`、call id=`entry`、按定义端口映射全部根输入，根输出逐项引用entry具名端口，调用现有build_model；不通过待测工厂计算期望。

```python
def test_weighted_sum_alpha_changes_complement_consistently(self):
    import torch
    from studio_core.module_templates import make_module
    from component_fixture import bundle_model
    a = torch.tensor([[1., 3.]])
    b = torch.tensor([[5., 7.]])
    for alpha in (0., 0.25, 1., 1.25):
        bundle = make_module('fusion.weighted_sum', {'features': 2, 'alpha': alpha})
        model = bundle_model(bundle, {'a': [None, 2], 'b': [None, 2]}, {'resource_profile': 'vision-v4', 'resource_batch_bound': 2})
        torch.testing.assert_close(model(a=a, b=b)['features'], a + alpha * (b - a))
```

- [x] 为其余工厂写独立参考：MLP/投影用原生nn.Linear等组成；GatedFusion比较gate与features、a/b梯度和门控参数更新；BasicBlock分别测试恒等和降采样投影。所有参考先装载同一权重/buffers。
- [x] 实现普通图定义，公开绑定只引用局部参数，不新增表达式求值器；WeightedSum采用规格公式，工厂/实例覆盖/源码三条路径均接受任意有限alpha，范围外允许外推，不截断。改变alpha不遗留互补常量；NaN/Inf/bool均拒绝。
- [x] 生成闭包重入/版本/接口非法测试，验证图→Python→图、去metadata、共享同实例和不同实例均保留声明的行为。
- [x] 跑定向测试，审查工厂只生成有限JSON、不导入torch或GUI；记录通过后才允许目录注册条目。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_module_templates.py -v
```

## Task A4：可展开的标准ResNet18

**Files:**
- Create: `studio/backend/studio_core/backbone_templates.py`、`studio/backend/tests/test_backbone_templates.py`、`studio/backend/tests/resnet_reference.py`。
- Consumes: BasicBlock及A2策略，`make_resnet18(...)`签名见规格4.1。
- Produces: 主干/分类器ComponentBundle；features与配置stage输出；普通芯片定义闭包。

- [ ] 写结构测试：标准3通道、K=1000时参数数11,689,512；stage block数[2,2,2,2]，输出通道64/128/256/512，224输入stage空间56/28/14/7，features512。1通道变体另验收，不套用3通道参数数。
- [ ] 明确稳定测试映射：entry组件内部名称stem/stage1..stage4/fc；stem含conv/bn/relu/pool；stage含block0/block1；block含conv1/bn1/relu1/conv2/bn2/add/relu2，可选projection，projection含conv/bn。这些是工厂生成规则，不要求用户不能重命名。
- [ ] 在resnet_reference.py写独立原生PyTorch参考；核心block必须按下面方式执行，而非调用平台图：

```python
import torch

class ReferenceBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        nn = torch.nn
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                           nn.BatchNorm2d(out_channels))
                           if stride != 1 or in_channels != out_channels else nn.Identity())
    def forward(self, x):
        residual = self.projection(x)
        hidden = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(hidden)) + residual)
```

完整参考 `ReferenceResNet18(input_channels, num_classes=None)` 采用7×7 stride2 padding3 Conv→BN→ReLU→3×3 stride2 padding1 MaxPool，四个stage各两个ReferenceBlock，stage2..4首块stride2，AdaptiveAvgPool2d(1)→Flatten→可选fc。`forward`返回features及按测试要求的stage/logits字典。`copy_reference_state(reference, graph_model)`依据上述稳定路径逐项strict拷贝卷积、BN及fc状态；必须断言目标对象全部映射且无重复目标。不能用zip两棵模块遍历顺序猜映射。

- [ ] 实现工厂，每个block/BN实例独立，定义复用不等于权重复用；工厂只负责已配置形状，修改结构参数需重新显式生成或内部手工校验。
- [ ] 224/batch2测试比较eval前向、32/batch2比较train输出、输入/参数梯度、单步更新、BN全部buffers。CPU设置数值容差并记录；CUDA另按后端设置容差，不能为过测盲目放宽。
- [ ] 核对层级深度、definition/node/JSON大小及累计预算均在限制内；完整网络支持带/不带visual metadata的Python往返、独立Model加载同state_dict。
- [ ] 实施时先运行定向参考测试，再跑小型真实训练/恢复；没有数值和状态证据不得把目录标为train可用。未做预训练，不显示权重下载/加载成功状态。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_backbone_templates.py -v
```

## Task A5：组件目录与文档原子插入

**Files:**
- Create: `studio/backend/studio_core/component_catalog.py`（若A3已建立类型则补齐条目）、`backend/tests/test_component_catalog.py`、`native/tests/test_component_insert.py`。
- Modify: `native/document.py`、`native/document_editing.py`、`native/chip_library.py`。
- Consumes: 已通过的工厂、算子默认值、现有不可变库闭包/冲突规则。
- Produces: 规格4.1目录接口与 `Document.add_component(bundle, position=None) -> str`。

- [ ] 写无需GUI的搜索测试，中文英文都匹配且结果稳定：

```python
def test_catalog_search_has_chinese_aliases_and_no_unimplemented_backbones(self):
    from studio_core.component_catalog import list_components
    self.assertIn('primitive.Add', [item.id for item in list_components('相加')])
    self.assertIn('primitive.Concat', [item.id for item in list_components('拼接')])
    self.assertIn('backbone.resnet18', [item.id for item in list_components('RESNET18')])
    self.assertEqual(list_components('resnet34'), [])
```

- [ ] 建立条目唯一ID：现有/新增primitive使用 `primitive.<kind>`；功能模块使用A3 ID；主干 `backbone.resnet18`，分类器 `model.resnet18_classifier`。label/aliases中文清晰，required_config列明形状/维数/类别等必需字段，未支持项不生成占位条目。
- [ ] native默认值从算子描述获取，移除重复_DEFAULTS来源但保留旧API返回等价值；不批量重构无关数据/训练模块。
- [ ] 原子插入测试：依赖闭包冲突一致改名，创建独立instance+call一次revision/undo；相同definitions复用但权重对象独立；缺依赖/资源不兼容/非法配置/递归均保持图及历史原样；未连线新调用是允许的草稿。
- [ ] 实现add_component，仅接受有类型的有限bundle，不执行动态工厂名；位置只改布局。不要在核心目录里读取用户目录，native把本地JSON芯片作为另一个有限来源合并显示。
- [ ] 跑backend定向、native定向、旧Document/芯片库回归，交付给总计划I1。A线不接管多视图渲染。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_component_catalog.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_component_insert.py -v
```

## 完成检查与预期证据

各任务按“失败行为测试→最小实现→定向通过→必要回归→独立审查”执行；本文件测试代码是未来验收要求，本轮没有运行。对红灯、通过数、跳过数、源码状态和原始报告分别记录。

A线完成不等于S4已完成；仍需B线、I1目录焦点接线和I2真实/冻结验收。ResNet18输出、BN状态或Python策略任一不一致，不能用缩小网络或保留旧metadata绕过。
