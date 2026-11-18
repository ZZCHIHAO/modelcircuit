# S4-B 多页签、分屏与层级工作台实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 主模型和芯片内部可以保持打开并通过页签/双分屏编辑，保存恢复可验证的工作区状态。

**Architecture:** 单一Document保存模型和统一历史；WorkbenchState保存有限视图状态；view_scope为现有文档命令提供明确上下文；CanvasHost持有每页独立DPG映射与事件令牌。切页不复制图或重新提交worker。

**Tech Stack:** 现有DearPyGui2.3.1、标准库dataclass/contextmanager/unittest；无Web、无OS输入自动化。

**Spec:** [S4规格](../specs/2026-11-15-s4-workbench-components-design.md)，第5节为权威身份和保存契约。

2026-11-17执行追记：B2真实常驻CanvasHost、事件路由及两轮范围复审已通过。最终23项CanvasView及完整native141项通过，退出0；包括Undo关闭当前定义、有效非焦点延迟事件后控件归属、空闲轮询保留未提交输入。B3正式页签/分屏控件、B4视图持久化仍未实施，精确pan/zoom边界不变。整轮交叉审查和交付记录见[本轮验证](../../../studio/native/S4_POLICY_CANVAS_VERIFICATION.md)。

2026-11-17本轮B3已通过独立修复复审与完整native199项，真实页签、独立侧栏、基础目录拖放和直角完成连线均接入。四项P2的请求/目标归属及离屏端点已闭合，整轮交叉审查与最终可携带记录另列。B4视图持久化仍未实施，临时拖线预览保持可见原生样式。

## Global Constraints

- 2026-11-15仅交付规划；2026-11-16已完成B0限定验证与B1纯状态/作用域，B2–B4仍未实施，正式多页签控件未交付。
- 主root页签固定primary不可关闭；每个definition最多一个页签，最多16页签、2个可见图；布局single/columns/rows。
- 所有页签共享模型和全局语义undo；view_id不能变成instance_id或参数对象ID。
- 定义、实例覆盖、调用路径探针三者分开；无调用上下文的定义允许编辑但不伪造测量。
- 导航、选择、分屏和标签关闭不改变模型revision/身份/权重/训练状态。
- 精确pan/zoom恢复未经公开API证明不承诺；不能使用item scroll或plot axis API冒充。
- 用户禁用电脑控制；后续UI验收使用应用回调与framebuffer。GitHub和原ml_forge业务代码保持原边界。

## 文件结构与接口

新增 `native/workbench_state.py`（纯状态）、`view_document.py`（显式文档上下文）、`canvas_view.py`（每页渲染/映射）、`workbench_panel.py`（页签/布局/定位控制）。共享的ui.py、document.py、workspace_state.py、main.py按任务顺序修改，不并发改写。

规格中的WorkbenchState公开方法不变；新增明确字段：`views:dict[str,ViewState]`、`layout:str`、`active_by_pane:dict[str,str|None]`、`focused_pane:str`、`warnings:list[str]`、只读属性`focused_view_id:str`。根视图ID固定root。`activate(view_id)`更新该视图所属pane的active以及focused_pane；single布局激活secondary视图时将其移回primary，root仍在primary标签组。重复打开已有定义保留其pane，移动显式调用move_to_pane。空secondary不得持有焦点：关闭当前页、reconcile移除当前页、转single均回退primary有效活动页，否则root。

`view_scope(document, view)`为contextmanager；`CanvasHost(document, workbench)`提供`sync(parent_by_pane) -> None`、`refresh_scope(definition_id) -> None`、`apply_selection(view_id, refs, *, intentional=False) -> None`、`dispose(view_id) -> None`、`dispose_all() -> None`、`get(view_id) -> CanvasView`；`CanvasView`字段按规格5.2，并保存瞬时`generation:int`。ViewState新增瞬时context_epoch；所有事件必须同时校验view_id、generation、context_epoch和document_epoch；工作区模型整体替换时递增document_epoch，同页调用上下文变更也递增context_epoch。

## Task B0：公开API和双editor可行性关口

**Files:**
- Create: `studio/native/tests/test_canvas_capabilities.py`、`docs/评审附件/S4-D0画布验证.md`（执行后才创建）。
- Read: `ui.py:_build/_resize/_render_graph/tick`，当前DPG公开签名及已安装版本。
- Produces: 明确支持/不支持/待实测的矩阵；创建/隐藏/显示editor的真实回调与原生状态证据。

- [x] 记录本机 `inspect.signature(dpg.add_node_editor)`、节点/连线/选择函数的公开签名；未使用私有native binding或改动DPG二进制。
- [x] 应用自有测试创建两个真实editor及同名模型引用/不同原生ID；限定连线/删线回调隔离、逻辑选择独立，隐藏/显示后句柄保持。
- [x] 同一UI线程切换分屏/resize，真实minimap已渲染；通过公开位置API与注册按钮回调核对Document布局归属。真实鼠标拖动/连线/双击/minimap导航仍未验证，不能把回调模拟外推成手势通过。
- [x] 记录常驻context的原生ID保持及公开API边界；没有将scroll或plot zoom当作pan/zoom，不承诺精确视口恢复。
- [x] 双editor限定验证没有观察到污染，允许继续B1/B2；本轮不替换前端，生产CanvasHost及事件版本仍待实施。

2026-11-16证据：6项定向测试通过，独立诊断CLI正常退出；销毁失败错误记成功的问题已修复并经范围复审。原生正选择、真实手势、非零pan后的保持及生产工作台集成未验证。详见 `docs/评审附件/S4-D0画布验证.md`。

未来命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_canvas_capabilities.py -v
```

## Task B1：纯视图状态与文档作用域适配

2026-11-16完成限定源码实现：20项状态、13项作用域测试及旧回归通过；歧义节点/instance/端口清理经修复复审；跨任务BN/快照/回滚2项与完整native108项通过。事件令牌消费、CanvasHost、实际页签和v3文件接线仍属后续。

**Files:**
- Create: `studio/native/workbench_state.py`、`view_document.py`、`native/tests/test_workbench_state.py`、`test_view_document.py`。
- Modify: `native/document.py`的scope属性，增加仅上下文使用的scope override；必要时调整`document_editing.py`的状态恢复。
- Consumes: 当前Document命令/原子undo、规格5.1/5.3。
- Produces: WorkbenchState全部公开方法/字段，view_scope的明确作用域/恢复语义。

- [x] 写纯Python失败测试，不启动DPG：

```python
def test_opening_shared_definition_reuses_view_without_cloning_model(self):
    from native.document import Document
    from native.workbench_state import WorkbenchState
    doc = Document()
    workbench = WorkbenchState()
    before_key, before_revision = doc.semantic_key(), doc.graph['revision']
    first = workbench.open_definition('encoder_chip', ['encode_a'])
    second = workbench.open_definition('encoder_chip', ['encode_b'])
    self.assertEqual(first, second)
    self.assertEqual(workbench.views[first].call_path, ['encode_b'])
    self.assertEqual(len(workbench.views), 2)
    self.assertEqual(doc.semantic_key(), before_key)
    self.assertEqual(doc.graph['revision'], before_revision)
```

- [x] 写异常恢复测试，显式证明对芯片内部操作不会误落到root：

```python
def test_scope_context_restores_cursor_even_when_command_fails(self):
    from native.document import Document
    from native.workbench_state import WorkbenchState
    from native.view_document import view_scope
    doc, workbench = Document(), WorkbenchState()
    view_id = workbench.open_definition('encoder_chip', ['encode_a'])
    with self.assertRaises(ValueError):
        with view_scope(doc, workbench.views[view_id]):
            self.assertEqual(doc.scope_id, 'encoder_chip')
            doc.set_input('missing_call', 'x', None)
    self.assertIsNone(doc.scope_id)
    self.assertEqual(doc.trail, [])
```

- [x] 实现有限状态校验、root保护、16页签边界、pane移动/布局变化、selection局部保存。超过上限原子拒绝；同definition重新打开重用页签，不克隆graph。
- [x] 补三条焦点回退测试：关闭secondary最后一页、reconcile关闭当前定义页、双栏切single后focused_view_id始终存在且落在primary。to_payload的views为有序list，from_payload返回对象，warnings为瞬时字符串列表；不把图对象藏入视图状态。
- [x] 实现显式scope_override上下文：验证definition存在，验证call_path逐级确实到达该definition；用独立sentinel区别未覆盖与根图None；只存definition ID，不存graph指针。call_path为空时可编辑定义但probe不可用；从芯片cursor临时进入root视图后应恢复芯片cursor。恢复时若旧目标已不存在，回root并清空选择，避免finally异常掩盖原错误。上下文不得跨异步回调或worker。
- [x] 补reconcile测试：调用删失但定义存在时清call_path，定义消失时关闭页签，模型替换回root；接口重命名/undo后移除无效选择；未知工作区字段和非有限数拒绝或按载荷策略净化。
- [x] 测试同实例多call与不同实例的selected parameter路径，探针输出严格含完整call_path；定义默认值和某实例覆盖值不能互写。
- [x] 跑定向与旧Document/芯片编辑回归，独立审查无假路径、无revision增加、无历史污染。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_workbench_state.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_view_document.py -v
```

## Task B2：独立CanvasView与事件路由

**Files:**
- Create: `studio/native/canvas_view.py`、`native/tests/test_canvas_view.py`。
- Modify: `native/ui.py:_build/_render_graph/_connect/_disconnect/tick`，通过小适配器保留旧接受路径，不把全部代码挤进ui.py。
- Consumes: B0报告、B1状态和view_scope。
- Produces: CanvasView/CanvasHost及带view_id/generation/document_epoch的回调；每视图独立node/port/link映射。

- [x] 写实际DPG对象隔离测试。构造方式与现有原生测试一致：

```python
def test_switching_tabs_keeps_editor_identity(self):
    import tempfile
    from native.document import Document
    from native.paths import resource_root
    from native.ui import NativeWorkbench
    with tempfile.TemporaryDirectory() as directory:
        ui = NativeWorkbench(Document(), None, directory, resource_root())
        try:
            ui.build()
            root = ui.canvas_host.get('root')
            import dearpygui.dearpygui as dpg
            native_id = lambda item: dpg.get_alias_id(item) if isinstance(item, str) else item
            original = native_id(root.editor_tag)
            original_nodes = {ref: native_id(item) for ref, item in root.ref_nodes.items()}
            self.assertGreater(original, 0)
            chip = ui.workbench.open_definition('encoder_chip', ['encode_a'])
            ui.workbench.activate(chip)
            ui.canvas_host.sync(ui.canvas_parent_by_pane)
            ui.workbench.activate('root')
            ui.canvas_host.sync(ui.canvas_parent_by_pane)
            self.assertEqual(native_id(ui.canvas_host.get('root').editor_tag), original)
            self.assertEqual({ref: native_id(item) for ref, item in ui.canvas_host.get('root').ref_nodes.items()}, original_nodes)
            self.assertIn('call:encode_a', ui.canvas_host.get('root').ref_nodes)
        finally:
            ui.shutdown()
```

NativeWorkbench._build在B2创建workbench/canvas_host及 `canvas_parent_by_pane={'primary':'native_canvas_primary','secondary':'native_canvas_secondary'}` 两个真实父容器；本任务以CanvasHost.sync驱动状态变化，B3再接页签控件。不得为过测引入只在测试里成立的editor_tag。

- [x] 将原单组映射移入CanvasView；tag前缀包括view_id和generation，模型ref仍是原ID。纯切页隐藏/显示现有editor，不能delete整个editor；结构变更时刷新受影响scope，generation递增。
- [x] 所有选中、拖动、连线、删线、双击、探针端口、属性/冻结和模态对话框事件先校验事件令牌；旧节点排队回调、已关闭页签和已替换工程的回调丢弃并记录，不触发模型命令。同定义A调用排队的参数回调在切换到B后必须失效，即使图未重绘。
- [x] 选择恢复采用ViewState.selected_refs与主题高亮，不能声称恢复原生ImNodes选择位。激活/重绘后的空get_selected_nodes结果以intentional=False传入apply_selection，保留逻辑选择；真实选择/空白点击清除或显式清除按钮才以intentional=True更新，并调用公开clear API同步清除。测试这两种空列表路径。
- [x] 用两个pane内相同局部call_id的测试验证事件归属。只有当前焦点图接受Delete，且输入框活动时不删除；芯片内拖动只更新该定义节点布局。
- [x] Document命令成功后统一reconcile；refresh_scope沿定义引用反向关系刷新本定义、全部传递祖先和root端口，不抢另一个pane的焦点，visited防止不完整草稿造成重复遍历。三级嵌套改输出shape时祖先正确失效，无关常驻editor的原生ID不变。主图上多个同定义实例仍是独立权重对象。
- [x] shutdown按确定顺序清CanvasHost→DPG上下文→既有worker管理；保留原关闭/取消回归。测试销毁页签仅销毁视图，不调用删除定义或训练取消。
- [x] 跑B1/B2、旧原生生命周期与端口测试，报告会话内视口实际等级，不宣称跨重建精确恢复。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_canvas_view.py -v
```

## Task B3：页签、双分屏与独立侧栏

**Files:**
- Create: `studio/native/workbench_panel.py`、`native/tests/test_workbench_panel.py`。
- Modify: `native/ui.py:_build/_resize/_render_tree/_render_inspector`、`native/chip_panel.py`事件归属。
- Consumes: B2 CanvasHost、WorkbenchState；组件目录位置保留I1的catalog_panel占位父容器，不做假组件。
- Produces: `WorkbenchPanel.activate_view(view_id)`、`open_call(call_path,pane='primary')->str`、`open_definition(definition_id)->str`、`set_layout(mode)`、`close_view(view_id)`、`focus_location(location)`。

`location`为有限dict：`{'definition_id':str|None, 'call_path':list[str], 'node_ref':str|None, 'field':str|None}`；字段必须来自已验证图引用，非法项提示“目标已变化”，不猜测同名节点。

- [x] 写主模型页签始终存在、双击调用打开chip页签、重复打开激活原页签、root不能关闭、左右/上下切换后的激活/焦点稳定测试。
- [x] 拆分“项目层级”和“组件目录”，搜索固定，层级树和组件结果各自滚动。沿用现有色彩/字体/面板风格，不扩展成全新视觉主题。
- [x] 采用明确的“单栏/左右/上下”布局入口和页签“移到另一栏”，首版分屏1:1；不在首批依赖复杂tab拖拽docking。root固定primary，另一pane无页签时显示可操作的打开提示，不显示假图。
- [x] 属性面板跟随focused_view_id，芯片页显示definition及所选调用路径，注明修改影响全部引用；实例参数仅在有效上下文修改。无上下文只显示定义配置，不伪造实例。
- [x] 修复所有旧 `_enter/_root/_back/_crumb` 入口：root激活root页；enter打开页；back/crumb改为激活目标层级对应页，不关闭其他页签。保留原Document导航API给旧调用者。
- [x] 应用拥有的framebuffer同时捕获single/columns/rows、长项目树和深层芯片；核对关键矩形不相交/不裁切。截图之外必须跑添加/编辑/连线/返回/关闭回调。
- [x] 跑B线和旧原生回归，交总计划I1连接真实目录面板。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_workbench_panel.py -v
```

## Task B4：工作区v3、定位和模型替换

**Files:**
- Modify: `native/workspace_state.py`、`native/main.py:load_saved_workspace`、`native/ui.py:_open_json/_save_json/_import_python/_save_draft`及主干替换入口。
- Create tests: `native/tests/test_workbench_persistence.py`、`test_workbench_locations.py`。
- Consumes: B1 `to_payload/from_payload`、B3定位接口、原保存错误保护。
- Produces: `read_workspace_full(path)->(document,experiment,workbench_payload)`、`load_saved_workspace_full(paths)`三元启动恢复；workspace_payload/save_workspace可选workbench参数；NativeWorkbench原构造参数后新增仅关键字workbench_payload=None；v3明确格式，旧二元API继续可用。

- [ ] 写v2→root默认、纯图→root默认、v3→视图/布局/选择恢复测试；view状态变化不改model_digest/checkpoint_signature。

```python
def test_workbench_layout_does_not_change_model_identity(self):
    from native.document import Document
    from native.workbench_state import WorkbenchState
    from native.workspace_state import workspace_payload, synthetic_config
    document, workbench = Document(), WorkbenchState()
    before = document.semantic_key()
    view_id = workbench.open_definition('encoder_chip', ['encode_b'], pane='secondary')
    workbench.set_layout('columns')
    workbench.activate(view_id)
    payload = workspace_payload(document, synthetic_config(), workbench=workbench.to_payload())
    self.assertEqual(payload['workspace_version'], 3)
    self.assertEqual(document.semantic_key(), before)
    self.assertEqual(payload['model'], document.graph)
```

- [ ] 实现新API而不改变旧read_workspace/load_saved_workspace的返回数量；启动主路径使用full函数，旧NativeWorkbench调用不传新参数仍成立。无workbench参数的写入仍为v2，新UI总传入v3载荷。strict有限JSON检查、8MiB等原文件上限和atomic写保护保留。
- [ ] 固定JSON views为有序list，逐项字段为view_id/target/call_path/selected_refs/probe_port/pane；from_payload把结构损坏的workbench重置root并保留有效模型/实验，引用失效则净化；warnings交UI日志而不持久化。read_workspace_full返回原始载荷供UI净化，避免丢失告警。旧格式返回第三项None。
- [ ] 写专门自动保存回归：“特征JSON无效→使用上次有效experiment→自动保存/退出保存→读取后仍为v3且页签/分屏保留”；所有异常回退分支同样传入workbench，不能成功降级成v2。
- [ ] 载荷无效引用净化到root并报告；未知workspace版本/损坏JSON保留备份且不覆盖原件；视图状态不存DPG句柄、模型对象、测量数组、任意文件路径或代码。
- [ ] 显式打开工程/Python导入/起始模型替换递增document_epoch并重置旧页签；恢复v3时先载入模型，再对全部目标reconcile，不能用旧view同名ID接上新图。运行快照保持既有隔离，真正模型变化显示stale。
- [ ] 让结构错误和已知探针来源调用focus_location：展开正确定义、设置真实call_path、选中对应node_ref。深层未采集来源只显示未采集，不伪造波形/梯度；该导航不触发训练。
- [ ] 测试删除/undo/redo、接口重命名、同名工程替换、版本升级失败和自动保存PermissionError；重要草稿和旧测量不能被异常处理清空。
- [ ] B线全部回归通过并独立审查后，移交总计划I1/I2。指南必须披露精确pan/zoom恢复实际支持等级。

未来定向命令：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\studio').Path+';'+(Resolve-Path '.\studio\backend').Path
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_workbench_persistence.py -v
& '.\studio\.venv\Scripts\python.exe' -m unittest discover -s studio/native/tests -p test_workbench_locations.py -v
```

## 退出关口与范围保护

B线完成须能同时保持root与chip页签，两个pane独立事件、选择与命令目标正确，关闭页签不删模型，运行中切页不取消/重建worker，保存恢复有限工作区状态。需要至少一次应用回调/framebuffer证据；仅检查状态类或tab数量不算完成。

跨会话精确视口、任意dock布局、多个独立工程同时打开、任意源码编辑器标签同步留在独立后续任务。S4中的Python仍使用已有代码窗口；不能为了页签统一而丢失代码草稿保护。训练关系画布尚未实现，不创建功能性假页签。
