# S5 源码阶段验收

日期：2026-11-18。版本：`v0.5.0-alpha.2`。本记录针对现有 Windows 桌面源码与 Python 环境，不包含新版便携 EXE 或干净机器兼容性。

| 关口 | 证据 |
|---|---|
| G0/G1 张量契约与兼容 | schema v2 轴、类型、批次与命名连接检查；10 个算子，参数共享、冻结、梯度、持久掩码和非法输入测试；保留旧格式与旧包 |
| G2/G3 数值与训练 | CPU、CUDA 各 41 个独立 PyTorch 对照，覆盖 Attention、Encoder、ViT、显式状态映射、前向、梯度与 SGD。默认 ViT 74,562 参数。两种设备的实际 worker 都读入生成的 PNG，完成训练、checkpoint v3 恢复和取消 |
| G4 Python 转换 | ViT 比较平台、独立导出及带/不带元数据的重导入；校验前向、梯度、状态及一次更新。修改原构造参数后优先采用源码；破坏掩码 helper 或缺少轴注解时拒绝导入 |
| G5 观测和容量 | 只搬运请求的矩形切片，过预算时不搬运。检查布尔 buffer 字节量、参数及捕获预算；硬件报告是保守估计。实际 worker 的 `[2,4,65,65]` 注意力测量只采集 80 字节，并通过 checkpoint 重放核对 |
| G6 原生与分类接口 | CPU 与 CUDA 各完成 13 次已注册回调和 4 张真实帧。覆盖放置、升级、连线、运行、注意力热图、最小窗口及工作区恢复。分类适配器分离输入映射、loss 和统计；旧 v2 分类 checkpoint 回归，S5 CPU v3 续训与连续训练的模型、优化器、历史一致 |

验证环境为现有 `torch 2.6.0+cpu` 和 `torch 2.6.0+cu124`，CUDA 设备为本机 `cuda:0`。完整 CPU 环境回归：backend 356 项运行、355 通过、1 项既有 Windows 符号链接权限跳过；native 278 项全部通过。实际 worker 和界面诊断均确认子进程退出。CUDA 续训通过功能核对，未宣称逐位确定性。

在项目根目录运行下列命令；CUDA 使用 `studio/.venv-cuda/Scripts/python.exe` 并把专项命令中的设备改为 `cuda:0`。界面诊断可将输出放在独立的可写临时目录，避开 D 盘测试过程观察到的文件占用。

```powershell
$env:PYTHONPATH='studio/backend;studio'
$env:OMP_NUM_THREADS='1'
$python='.\studio\.venv\Scripts\python.exe'
& $python -m unittest discover -s studio/backend/tests -v
& $python -m unittest discover -s studio/native/tests -v
& $python studio/backend/tests/transformer_acceptance.py --device cpu --section all --output output/s5-release/cpu-compare.json
& $python -m native.s5_acceptance --device cpu --output output/s5-release/cpu-worker.json
& $python -m native.s5_render_acceptance --device cpu --output "$env:TEMP/modelcircuit-s5-verification/cpu-render.json"
```

本地原始报告在 `output/s5-release/`。测试仅用生成小样本验证执行，未评估科研性能、跨机器显卡兼容性或物理拖拽。没有新增第二训练任务；回归、多头及生成不纳入 S5 完成声明。
