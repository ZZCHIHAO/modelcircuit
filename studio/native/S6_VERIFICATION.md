# S6 验收记录 · v0.6.0-alpha.1

2026 年 11 月 18 日在本机 Windows x64 验收。两个便携包都以 `fe39ddaf42ecbd648e672847af5441f1ed6ae503` 构建，代码快照 SHA-256 为 `6cf3cfd11b37de93ca71277e6077ee4680efb31aaf0304b9d8b4ffc4033537c2`。随包 `release-build.json` 记录来源、设备运行时和依赖版本。

| 验证 | 结果 |
| --- | --- |
| CPU 环境后端完整测试 | 363 通过，1 项因 Windows 符号链接权限跳过 |
| CPU 环境原生桌面完整测试 | 279 通过 |
| Windows CPU EXE | `cpu` PyTorch 回归、SVR CPU 拟合，全部通过 |
| Windows CUDA EXE | `cuda:0` PyTorch 回归、SVR CPU 拟合，全部通过 |
| SVC 和 SVR 独立对照 | 使用同一冻结数据与参数重新拟合 scikit-learn，预测和指标一致 |

桌面验收通过 DearPyGui 自身回调与帧缓冲运行。包括分库筛选、SVR 放置、回归目标列的显式选择、分组数据检查、起始图、保存重开、Python 往返、实际工作进程、取消、从第 3 个 epoch 继续到第 5 个，以及参数和梯度探针。回归测试集的预测来自验证集最优 checkpoint，独立重新加载模型逐样本核对，并复算 MAE、RMSE 和 R²。SVM 流程验证了 CPU 拟合与结果页，也验证了修改导出代码中的 `C` 后回读画布。

| 用途 | PyTorch / scikit-learn | 数据 |
| --- | --- | --- |
| CPU 包 | `2.6.0+cpu` / `1.7.2` | 本地合成 CSV |
| CUDA 包 | `2.6.0+cu124` / `1.7.2` | 同一数据，PyTorch 使用 RTX 4060 Laptop `cuda:0` |

诊断命令示例。将输出目录与实际用户数据目录分开。

```powershell
& .\studio\dist\s6-cpu\ModelCircuitStudio\ModelCircuitStudio.exe --data-dir "$env:TEMP\modelcircuit-s6-check-data" --s6-render-check "$env:TEMP\modelcircuit-s6-check\report.json" --check-device cpu
```

本验收使用小型合成数据。干净 Windows、长期压力测试和签名仍待验收。SVM 在单个根级估计器模式下运行，保存数据清单、运行参数、指标和预测。PyTorch 续训使用其原有检查点机制。
