# DearPyGui 原生工作台设计 QA

日期：2026-11-14。结果：**passed，源码版及冻结版原生渲染与所列回调均已通过**。

本轮目视未发现 P0/P1/P2 视觉阻断。主协调任务已实际完成重新打包后的冻结程序渲染验收，`render-frozen.json` 为 passed；8节点真实矩形无重叠、20步CPU训练、具体调用探针、非法Python拒绝与草稿保留、取消和worker退出均有记录。**当前三张原生PNG已由冻结程序自己的framebuffer重新生成，并已据此重新组合与更新哈希**。已有 `frozen-runtime.json` 仍只作为运行器诊断；冻结GUI证据由新的 `render-frozen.json` 独立提供。

## 证据来源与参考边界

新实现由 `studio/native/ui.py` 创建 DearPyGui 原生 viewport、node editor 和控件。当前截图来自打包后的 `studio/dist/desktop/MLForgeStudio/MLForgeStudio.exe`，由同一程序的 `dpg.output_frame_buffer` 产生，经 DPG 自身 PNG 编码后原样复制到中文输出路径。可核对 `studio/native/render_acceptance.py` 的 `_capture`，以及 `ui.py:828` 的 `create_viewport`。这些是冻结程序自身 framebuffer，**不是浏览器截图、网页嵌入，也不是重新绘制的 UI**；不包含操作系统窗口边框。

本轮只读取现有截图、源码和 JSON，使用 Pillow 等比缩放/裁切并增加图外 QA 标签，没有操作浏览器、Windows 桌面或修改业务代码。原版截图是布局参考；新实现不是像素克隆。对照检查的是左侧组件/层级、中央网格画布、右侧属性/实验与底部结果区，以及芯片进入/返回和具体调用的测量上下文。

| 来源 | 实际读取尺寸 | 用途 |
|---|---|---|
| `ml_forge/assets/screenshot.png` | 1918 × 1053 | 原版 DearPyGui 布局参考 |
| `output/native/native-root.png` | 1364 × 821 | 冻结版根图和训练完成状态 |
| `output/native/native-chip.png` | 1364 × 821 | 冻结版芯片内部、属性、具体调用探针 |
| `output/native/native-python-error.png` | 1364 × 821 | 冻结版 Python 静态导入失败的可见反馈 |
| `output/native/comparison-native.png` | 2848 × 1110 | 原版与新原生根图等比对照 |
| `output/native/comparison-native-focused.png` | 2560 × 1530 | 层级、内部、属性与探针局部 |

最新原图和组合图 SHA256、裁切坐标、组合尺寸、冻结运行记录、当前exe的hash与本轮读取的 JSON 内容保存在 `output/native/design-qa-manifest.json`。组合图不拉伸原图；不同纵横比以留白处理。历史源码渲染报告保留，旧源码PNG已被当前冻结PNG覆盖，不能将现图当作另一套保留的源码截图。

![原版与新原生工作台对照](/D:/学习/岳广辉课题组/科研工具/ml_forge-main/output/native/comparison-native.png)

## 逐状态目视检查

1. **根模型：passed。** 8个节点均在中央视口中，两个输入分别进入共享编码器的 `encode_a`、`encode_b` 调用，随后融合、分类并给出两个输出。标题、端口及连线可辨，未见节点互相覆盖。右侧显示“实验完成 · 20/20 · r1”；底部显示设计/结果 revision、真实训练损失和 S1 合成 CPU 声明。此图未选中节点，属性面板显示选择提示属于正常空状态。
2. **内部芯片和属性：passed。** 面包屑显示“编码芯片 · encode_b”，有返回上层入口。内部为输入 → Linear → ReLU → 输出；属性区明确 `dense` 实例、`project` 调用和输入4/输出8，并显示参数应用、冻结和共享调用入口。画面说明进入的是哪个调用的内部，未把两个调用误画成两套独立参数。
3. **具体调用探针：passed。** 内部页底部为 `call:encode_b/call:project`，结果 r1，shape `[16,8]`、float32，128个值，NaN/Inf 均为0；可见最小值约 -1.2717、最大值约 1.69728、均值约 0.12206、标准差约 0.545848。原截图的长数值预览受结果区视口裁切，本组合未补画不可见部分；完整测量保留于运行 JSON。
4. **Python 失败反馈：passed。** 原生弹窗保留代码编辑区，显示 `Unsupported top-level Python statement (line 53)`，并保留重新解析、关闭并保留草稿的操作。这里的错误是预期拒绝不支持源码，不是程序运行崩溃。回调断言另证实原图语义未变、关闭重开后失败草稿仍在。

![原生芯片、属性与探针对照](/D:/学习/岳广辉课题组/科研工具/ml_forge-main/output/native/comparison-native-focused.png)

![原生 Python 导入失败反馈](/D:/学习/岳广辉课题组/科研工具/ml_forge-main/output/native/native-python-error.png)

## 已有可核对的程序证据

本轮未重新启动程序或运行诊断，以下结论来自已有 JSON，并与诊断源码及截图相互核对。

| 证据 | 已记录内容 | 不能扩大为的结论 |
|---|---|---|
| `output/native/render-source.json` | `status=passed`；8节点；`root_node_rectangles_do_not_overlap=true`；具体探针路径；失败Python草稿保留；`workers_stopped=true` | 不覆盖任意用户布局、所有分辨率或冻结GUI |
| `output/native/render-frozen.json` | 冻结程序实际渲染 passed；8节点无重叠；具体调用探针；失败Python草稿保留；实际取消；`workers_stopped=true`；三张当前framebuffer | 不覆盖任意自定义图、所有窗口尺寸或所有桌面输入操作 |
| `output/native/source-runtime.json` | 源码运行器 passed；58参数；loss 0.9259226322 → 0.4936531782；同批次准确率0.9375；Python往返、具体调用探针和退出检查 | 同批次合成数据不证明测试集泛化、医学性能或全平台能力 |
| `output/native/frozen-runtime.json` | 已有冻结运行器诊断 passed，真实CPU结果、取消与worker停止已记录 | 该行只证明运行器；冻结GUI使用上一行独立渲染证据 |

`render-source.json` 对应的原始运行记录：

- `output/native/render-source-state/runs/184e6516aace4cff8983e40fdc222a49.json`：`succeeded`，20步，seed 42，batch 16，有实际 history、probes、parameters、outputs。
- `output/native/render-source-state/runs/f1ae8ecf82574a32a988a882639f528d.json`：`cancelled`，并非把取消请求直接当作完成。

最新冻结GUI验收对应的原始运行记录：

- `output/native/render-frozen-state/runs/4fa743fad2f44399b48224ee380dd9d8.json`：`succeeded`、`exit_code=0`；20步、seed 42、batch 16、58参数；loss 0.9259226322 → 0.4936531782，同批次准确率0.9375；记录含 history、probes、parameters 和 outputs。
- `output/native/render-frozen-state/runs/2ccd45d2a16a4a19a1d1631118e34ae0.json`：`cancelled`、`exit_code=0`；200步请求经实际取消后进入终态。
- `render-frozen.json` 指定测量路径为 `call:encode_b/call:project`，并确认失败Python草稿保留和全部worker已停止。

`render_acceptance.py` 在帧渲染后读取实际节点 position 与 rect size，逐对断言矩形不相交，断言8节点，再运行 CPU 实验、进入具体芯片调用、采集 framebuffer、拒绝非法 Python 并保留草稿、确认取消终态。它属于程序内回调验证，没有模拟桌面鼠标或浏览器。

## 历史修复与当前状态

以下是本轮开发中的已修复问题记录，不为再次重现历史崩溃而回退代码：

- 中文资源路径下 `dpg.load_image` 返回 None：改为 Pillow 读取实际文件 bytes 后解码并创建 DPG texture；现代码见 `studio/native/ui.py:140`。截图中的原资源标识可见。
- DearPyGui 2.3.1 Windows 中文 viewport title 在首帧触发 access violation：操作系统品牌标题改为 ASCII `ML Forge Studio`；工作台中文 UI 保留，见 `ui.py:825`。当前源码渲染完成不等于已验证所有DPG版本。
- 根图8节点曾重叠：结合真实 rect 测量调整显示映射和节点信息密度；`render-source.json` 与最新 `render-frozen.json` 的两两不重叠检查均通过，最新组合图目视也未见重叠。该检查限定当前示例和截图尺寸。

## 验收范围与剩余限制

当前截图保留原生 DPG 的紧凑字号和面板滚动；右侧完整配置、长张量预览需滚动查看。截图不能证明键盘顺序、屏幕阅读器支持、所有焦点状态、其他 DPI/窗口尺寸或长期运行稳定性；本轮不宣称完整可访问性认证。

冻结程序已完成本文件所列的实际启动、原生渲染、中文资源显示、程序内回调、真实CPU运行和worker退出验收；当前exe内容hash已记入manifest。**源码版与冻结版均为passed**，不再保留“冻结最终GUI待验证”的旧结论。该通过范围不等于任意操作系统环境、全部鼠标键盘路径或完整科研任务平台均已验收。
