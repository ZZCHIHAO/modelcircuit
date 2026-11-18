# 原生 Windows 桌面版验收记录

日期：2026-11-14。采用原版 ML Forge 的 DearPyGui 原生窗口、节点编辑器和桌面布局；不是浏览器包装。当前为 S1 垂直闭环，不是完整科研平台。

## 实际完成

- `studio/build-desktop.ps1` 构建成功，产物：`studio/dist/desktop/MLForgeStudio/MLForgeStudio.exe`。采用便携目录形式，必须保留旁边的 `_internal`，不是只有一个文件的自解压安装器。
- 原生文档/路径/生命周期：17项测试通过。后端核心、Python桥、进程与持久化：62项测试通过。新增用例实际验证关闭时持久化失败仍清理进程，以及 Windows 原子写入临时读锁重试。
- 源码运行器诊断与冻结EXE诊断均真实通过。默认CPU实验58参数，20次SGD更新，loss 0.9259226322174072→0.49365317821502686，同批次accuracy 0.9375；不得作为独立测试集或医学效果。
- 最新冻结原生渲染检查：`output/native/render-frozen.json`，exit 0，run `4fa743fad2f44399b48224ee380dd9d8` 为succeeded；cancel `2ccd45d2a16a4a19a1d1631118e34ae0` 为cancelled，均exit 0；最终 `workers_stopped=true`。
- 冻结检查实际创建DPG窗口并渲染framebuffer，调用应用自身回调，检验根图8节点真实矩形两两无重叠、真实训练、具体调用探针、非法Python不执行/草稿保留、取消和退出清理。它没有注入操作系统鼠标键盘，不将此记录冒充完整人工操作验收。
- 截图由冻结程序自身生成，1364×821：`output/native/native-root.png`、`native-chip.png`、`native-python-error.png`。可随源码浏览的主图和探针图复制至 `docs/images/`。组合比较和限制见 [design-qa.md](design-qa.md)。

## 修复过的真实问题

1. DPG文件名接口读取中文路径PNG返回None：改用Python读取实际图像字节并由Pillow解码为纹理。
2. DPG2.3.1中文viewport标题导致首帧原生access violation：单变量最小窗口复现后，使用ASCII品牌标题“ML Forge Studio”；中文界面与中文字体保留。
3. 默认芯片/输出节点重叠：依据真实rect测量压缩重复文字并调整显示映射及逆映射，未修改用户文档持久化坐标。
4. Windows读者短暂占用运行JSON导致替换失败：有限重试PermissionError，保持原子替换与最终错误报告；并非放宽文件权限。
5. 初始化失败不进入异常处理、manager.close失败绕过清理：已故障注入复现，纳入启动错误报告并保证清理兜底。

用户要求停止Use Computer调试后，所有后续检查均使用源码测试、应用自身回调/渲染和打包自检，未继续使用电脑控制。之前少量原生窗口观察仅用于定位首批问题。

## 运行边界

已验证本机Windows、中文/空格项目路径、非项目当前目录下的冻结进程。未验证全新Windows机器、全部显卡/DPI组合、完整键盘/辅助技术、触屏、大规模图、签名安装器和自动更新。系统中文字体来自Windows本机微软雅黑，不随包复制。

节点图、分支、共享实例、嵌套单输出芯片、根图多输出、七个算子、冻结、根图封装、受限Python往返和真实合成CPU实验可用。医学影像、3D真实训练、LLM/RAG、传统ML、任意Python和复杂训练策略仍是后续阶段。原生UI尚未覆盖网页版本的所有编辑入口，详见README。

## 源码与上传

原 `ml_forge/` 基线40个文件哈希均未变化。原45项基底核查列表中仍只有根`pyproject.toml`变更（限制包发现范围）；新桌面实现位于studio。根`.gitignore`另有排除本地构建/依赖/日志的更新。

为用户授权的GitHub上传已初始化本地Git，未伪造GitHub身份、未提交到远端。用户要求个人private、仅自己访问、无协作者。GitHub网络与登录未完成，用户明确要求本轮暂缓，后续新对话继续；详见 [../HANDOFF.md](../HANDOFF.md)。
