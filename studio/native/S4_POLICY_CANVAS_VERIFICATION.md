# S4 A2+B2：资源/Python契约与常驻画布验收

日期：2026-11-17。A2、B2及两者最小集成的源码里程碑已通过完整回归与独立交叉审查。本文只证明这一限定范围，不代表整个S4或新的EXE完成。可携带[验收记录](../../docs/评审附件/2026-11-17-S4-A2-B2验收记录.json)已生成。

## 本轮能力与使用边界

| 能力 | 本轮源码位置及效果 |
|---|---|
| 新资源策略 | `backend/studio_core/execution_policy.py`、`schema.py`、`runtime.py`；vision-v4要求显式batch bound，旧资源档位保持既有语义 |
| 受控Python v4 | `backend/studio_core/python_io.py`、`vision_python.py`；支持六新算子和资源策略的静态往返，当前源码优先于布局元数据 |
| 芯片包兼容 | `native/chip_library.py`、`document_editing.py`；v2包显式声明算子集与策略，不能暗中升级当前工程 |
| 常驻多画布 | `native/canvas_view.py`、`canvas_commands.py`、`ui.py`；每个view保留真实editor及独立节点/端口/连线映射，命令携带版本化owner |
| 旧工程保护 | workspace仍为v2，保存根策略；原`ml_forge/`与历史S3包保留 |

这不是完整S4产品交付。左侧完整分类目录、功能芯片工厂、ResNet18模板、正式页签条与分屏菜单、workspace v3视图恢复及新EXE尚未交付。新算子可经受支持Python/图工程进入核心及画布；旧palette还没有全部新算子的添加入口。不要因CanvasHost具备分屏机制就把完整页签交互标为已完成。

当前启动器默认打开的固定CPU/CUDA EXE仍是S3。需要验证本轮源码时使用项目现有Python环境从源码入口启动；不要把旧EXE里的菜单当作本轮源码界面。

从项目根目录手动查看源码版时可使用独立状态目录：

```powershell
& '.\studio\.venv\Scripts\python.exe' '.\studio\desktop_entry.py' --data-dir '.\output\s4-source-preview'
# CUDA环境入口；在实验设置中仍需显式选择设备。
& '.\studio\.venv-cuda\Scripts\python.exe' '.\studio\desktop_entry.py' --data-dir '.\output\s4-source-preview-cuda'
```

## 资源契约

新根图声明：

```json
{"resource_profile":"vision-v4","resource_batch_bound":2}
```

bound必须是1..64的严格整数；只允许首轴为未知batch，实际及实验配置batch均受该显式上界约束。旧无profile/classification-v2的静态估算仍按64，这没有给旧模型新增实际batch≤64的硬限制。

新策略区分编译重复推导和实际执行路径，保留参数、单张量、累计执行、resolved/inference节点及独立BN持久状态限制。每个definition默认路径和根路径各自验证，未使用定义不会被跳过。嵌套同源输出别名只在同一call内归并，不会合并两个真实调用的BN/Dropout效果。

有限四层宽图的实测账本：resolved72、inference228、compile228,000,000、根execution20,000,000，各定义execution4/8/12/16M。相同固定batch旧档仍按原累计口径拒绝。这证明重复编译与执行拆分解决了有限算例中的失败机制；没有生成或执行生产ResNet18，A4仍须用真实闭包和独立网络参考验收。

新策略/bound进入新模型和checkpoint身份，改变后拒绝原恢复身份。CPU微型BN+Dropout分类测试核对完整epoch恢复、模型/buffers、历史和最终指标；该新用例未另做优化器状态逐项相等断言，不能据此承诺CUDA位级续训或任意任务恢复。

## Python与芯片往返

所有`vision-pytorch-v4`导出代码都声明有限`MLFORGE_EXECUTION_POLICY`。旧资源档位可以为None或classification-v2，只有vision-v4携带bound及活跃根检查。删除visual metadata后策略仍在当前源码中，改变合法bound和构造参数会反映到导回图。

vision策略必须配套精确helper、根forward首条检查及全部根输入。删除、移动、改写检查，或者用普通chip类/import别名覆盖活跃检查函数、策略变量或数学helper，都明确静态拒绝导入。根forward的局部结果或别名赋值也不能遮蔽活跃检查函数，包括出现在检查之后的赋值；否则Python会在首条检查处产生UnboundLocalError。导入器不执行用户Python来决定结构。无活跃新声明时，旧公开参数、普通chip类及局部结果的合法命名保留。

新v4模型身份将显式`resource_profile:null`与缺省字段视为同一策略表示，带/不带visual metadata的未编辑往返保持model_digest和checkpoint_signature。只在v4路径规范化，历史v1–v3黄金身份保持。去掉visual metadata会失去文档ID，身份比较仍须恢复同一文档ID，不能把布局信息被删除误当成跨文档身份保证。

独立Python比较范围包括输出、梯度、一步SGD、BN buffers、冻结/共享和实际batch上界。**独立导出没有复制平台完整资源预算账本，不等于平台全部分配限制或峰值内存保证。**任意第三方Python无损导入也不在支持范围。

新算子或vision策略芯片用v2包；旧芯片继续v1。v2导入要求调用方明确提供完全匹配的策略，失败不改变工程或历史。本轮没有策略升级对话框；I1才提供明确应用策略的产品路径。包只含架构、接口、公开参数及依赖闭包，不含权重。

## 画布、事件与实际渲染

每个view使用独立常驻child_window和node_editor。纯导航隐藏/显示父容器；跨pane只移动容器，保持editor及未变节点的真实ID。仅隐藏node_editor曾导致实际图形串画布，已由真实S3渲染断言发现并修复，没有弱化原断言。

回调令牌为view_id、generation、context_epoch、document_epoch。过期图控件、旧调用上下文、已关闭/重开的view、模型替换前回调均丢弃。定义内命令通过显式view_scope落到其owner；Delete另要求当前可见焦点及无活动输入。整体导入重置epoch并关闭旧模态。

逻辑选择通过ViewState及主题高亮恢复，不伪造ImNodes原生选择位。显式“清除选择”会清除逻辑和原生选择；空native getter不能清掉已有逻辑选择。真实鼠标空白点击、正向原生选择位和精确pan/zoom没有验证，不承诺跨会话视口恢复。

应用自身framebuffer检查包括单root、单chip、左右、上下四种状态；同尺寸S3历史对照用于检查旧命名输出/探针区域。小pane中超出视口的节点仍可能裁切，本轮没有自动fit保证。所有检查没有使用电脑控制或OS/浏览器输入自动化。

独立审查额外发现并修复了三条实际操作序列：Undo删除当前打开定义后焦点回root；有效非焦点延迟事件完成后共享属性/组件按钮回到真正焦点；空闲位置轮询不重建输入框或擦除未提交文字。最后一项用真实widget ID、三轮轮询及tick验证，区分模型命令归属与界面呈现归属。

## 最终验证记录

| 最终验证 | 结果 | 证据 |
|---|---|---|
| CPU后端完整套件 | 251运行，250通过、1旧Windows symlink权限跳过；29.590s，退出0 | `backend-cpu-release-source-full.log/json/exitcode.txt` |
| CPU桌面完整套件 | 141通过，无跳过；61.875s，退出0 | `native-cpu-release-source-full.log/json/exitcode.txt` |
| CPU / cuda:0独立v4微型检查 | 两环境实际设备执行；输出/梯度/一步SGD/BN buffers/严格state重放/batch guard通过，退出0 | `device-cpu-final.json`、`device-cuda-final.json`及对应日志/退出码 |
| 定向修复 | Python20、CanvasView23、芯片策略8项最终通过；两项跨线测试已纳入141项 | `a2/`、`b2/`、`bridge/`、`final-review-fixes/`及完整native日志 |

旧symlink跳过因本机WinError1314无法创建夹具链接，不是该路径的成功验证。没有运行本轮完整CUDA套件或新的冻结EXE验收；此前138/140项桌面绿灯分别是较早候选，最终以141项及当前源码哈希为准。测试中的CSV/PNG短训练和生成张量只用于工程核对，不代表科研性能。

证据目录：`output/native-s4-a2-b2/`。后端定向红绿、两轮复审及bridge嵌套封装策略红绿均保留；跨线测试`test_policy_canvas_integration.py`检查无metadata的Multiply端口、有效芯片连线、实际Python导入回调和模型替换后旧事件失效。

可携带记录`docs/评审附件/2026-11-17-S4-A2-B2验收记录.json`包含实际套件摘要/退出码/日志哈希、设备报告、22份修改或新增Python源码哈希、独立审查结论及其全文、历史产物保护核对。110份既有Python源的基线保留，原ml_forge33份Python源码未变。S3两份EXE、CPU/CUDA及源码三个ZIP的整体哈希仍匹配历史身份。

两线范围审查、native桥接复审与整轮交叉审查均已接受其声明范围。整体审查发现的局部guard遮蔽和null策略身份漂移，经唯一组合修复波次、真实红绿及一次独立范围复审关闭；没有该修复范围遗留P1/P2。原初审diff及失败日志保留，最终源码另有冻结清单，不覆盖原证据。`git diff --check`退出0，无HEAD、remote、提交或上传动作。

同尺寸渲染[对照图](../../docs/images/native-s4-policy-canvas-comparison.png)含S3参考/旧场景回归、单root/单chip、左右/上下状态；原始四图与几何/原生ID记录在本轮output目录。可见裁切和未验手势边界保持上文限定。

## 下一交付

下一步并行实施A3功能芯片工厂与B3页签/分屏/侧栏控件；随后A4 ResNet18、A5/I1组件目录接线、B4视图持久化，最后I2重新打包CPU/CUDA S4 EXE并实际验收。训练关系图、交互式反向/更新单步、3D/分割、文本/大模型、多GPU/远程仍属于后续里程碑。
