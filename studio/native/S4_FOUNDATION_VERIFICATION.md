# S4 A1/B1 基础核心实现与验证

日期：2026-11-16。本轮完成A1计算算子核心与B1纯视图状态/作用域，不代表整个S4已完成。正式便携程序仍为S3，没有新增页签/组件菜单的EXE，旧包未覆盖；GitHub继续暂停。

## 已完成的源代码能力

### A1 算子核心

有限算子声明由原11项扩为17项，新增BatchNorm2d、Identity、Sigmoid、Multiply、Scale、OneMinus。实现分为无torch/GUI依赖的 `studio_core/operators.py`、`operator_rules.py`，以及显式构造的 `operator_runtime.py`，接入schema/runtime。

- Multiply双输入严格同形，不隐式广播；Scale为固定有限标量，无可训练Parameter，执行时将合法大整数转float；OneMinus为1-x。
- BN核对affine×track_running_stats四组合、train/eval、momentum=None、frozen只冻结参数、同实例多调用和同定义不同实例、严格state_dict。批次统计时N*H*W≤1明确拒绝并定位调用，不切eval或丢弃末批。
- BN可选affine仍仅计0/2C参数。持久running buffers另作构造前预算：含未调用实例/未使用定义；共享call不重复计模块状态，不同实例分别计。保持原单张量/累计阈值，未借此实现A2的新激活计费或缓存。
- 旧11算子的允许参数、省略规则与诊断定位保留。Dropout构造描述缺省0.5、编辑预设0.1、旧图required p三者分开；注册描述返回防御性数据。
- 低层 `make_operator` 无整个图的资源上下文；外部图必须走prepare_graph/build_model，不能绕过模型分配前检查。

新算子当前仅在源码计算核心注册。原生目录/端口渲染尚未接线，Document.add对新kind明确报“编辑器尚未接入”，避免_DEFAULTS缺项变成KeyError。python_constructor字段只描述构造符号；当前Python导出器仍明确拒绝这六种新kind，Python v4未实现。

### B1 多视图状态与作用域基础

新增 `native/workbench_state.py` 与 `view_document.py`；修改Document的临时scope解析，不改变旧导航调用接口。

- 同一工程的root固定页、每definition单页、最多16页、single/columns/rows两pane逻辑状态；有效焦点回退与调用上下文epoch。
- 有限有序JSON载荷、瞬时警告、失效定义/调用/选择/端口净化。非法草稿可保留，但歧义node、instance、definition或输出端口不能保留为可用探针来源。
- `view_scope(document, view)`只保存ID、独立sentinel和游标状态，无图对象副本；当前graph换对象后按ID重新定位。无真实调用链的定义可编辑，但不会伪造根探针路径。
- 导航/页签/分屏状态不改模型revision、语义身份或统一undo。实际模型命令仍走原事务与历史；异常、嵌套、rollback/undo/replace后恢复合法游标。
- 跨任务集成证明：定义中的BN eps修改不改变已构建运行快照；重建时两个实例读取新定义，但参数及统计buffer独立。不同调用探针路径不同，失败命令替换graph后后续编辑仍落到正确定义。

这里是状态与命令适配基础，**没有**CanvasHost、实际页签/分屏控件、过期事件令牌消费、workspace v3文件接线或精确pan/zoom恢复。不能把状态类的layout字段当作新界面已经显示。

## 实际测试证据

| 检查 | 结果 | 本地原始证据 |
|---|---|---|
| A1返修后的定向＋旧兼容 | 27新项＋105旧项，共132通过 | `output/native-s4-a1-b1/a1/review-fixes-regression.log` |
| B1返修后的定向＋旧兼容 | 20状态＋13作用域＋24旧项，共57通过 | `output/native-s4-a1-b1/b1/green-review-ambiguity-and-regression.log` |
| 跨任务集成 | 2项通过，纳入完整native | `test_operator_view_integration.py` |
| CPU完整backend | 运行219项：218通过、1项Windows symlink权限跳过；退出0 | `output/native-s4-a1-b1/backend-cpu-full.log/json/exitcode.txt` |
| CPU完整native | 108项全部通过，退出0 | `output/native-s4-a1-b1/native-cpu-full.log/json/exitcode.txt` |
| CPU微型算子链 | 六新算子前向、输入/参数梯度、SGD一步、buffers与严格state_dict重放一致，退出0 | `output/native-s4-a1-b1/device-cpu.json` |
| CUDA微型算子链 | 同一独立参考核对通过，状态真实位于cuda:0，退出0 | `output/native-s4-a1-b1/device-cuda.json` |

设备核对使用生成的2×3×4×4张量及一次优化，不下载数据/权重，不是科研性能或完整CUDA suite的验收。CPU torch2.6.0+cpu、CUDA torch2.6.0+cu124；未安装新依赖。旧分类/配对/checkpoint/Python身份的完整CPU回归通过，不将其外推为新算子的分类checkpoint或Python v4已完成。

D0正式测试中的“BN尚不支持”原是历史现状断言；本轮先保存其真实失败，再移除已过时的BN拒绝断言，保留vision-v4仍不支持的检查。BN正向/状态由A1新测试承担，不能为了保留旧拒绝结论而阻止新能力。

## 独立审查与修复

首审发现并修复三项：

1. P1：未调用、无affine的巨大BN可越过参数计数并分配统计buffer。补独立持久buffer账本和构造前反例，保留参数计数及旧激活预算。
2. P2：合法有限整数Scale可能在PyTorch整数标量转换时溢出。仅执行构造转float，保留原图/归一化中的整数与身份。
3. P2：reconcile折叠重复ID后保留歧义选择/探针。改为kind-local唯一解析；原探针首选失效则清空，不转移到下一个选择。

各项修复先红后绿、范围复审通过；最终跨任务审查无新增实质发现。审查接受的是本轮限定核心，不是完整S4。原始审查与实现报告留在 `.superpowers/sdd/2026-11-15-s4-workbench-components/`，可携带状态/哈希摘要见[本轮核查记录](../../docs/评审附件/2026-11-16-S4-A1-B1验收记录.json)。

## 复现和下一步

使用项目已有环境，在项目根目录执行：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/backend/tests -p test_vision_operators.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_workbench_state.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_view_document.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_operator_view_integration.py -v
```

下一任务是A2的新profile预算/受控Python契约与B2的CanvasHost/事件路由。ResNet18模板、功能融合头、组件目录、正式多页签与分屏、工作区v3及新EXE分别继续按计划验收；不能跳过[D0预算阻断](S4_D0_VERIFICATION.md)，也不以放大旧阈值或缩窄网络代替解决。
