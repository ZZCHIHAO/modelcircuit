# S10 科研工程与独立复现

`v0.9.1-alpha.1` 新增「工程」页。导出的工程可以在另一台机器重新绑定同一份数据，用 Python 运行训练和推理，再回到桌面打开记录与继续编辑模型。支持 ImageFolder 二维分类，以及使用配对清单的二维 PNG/TIFF 或三维 NIfTI 分割。

## 导出和迁移

1. 在「实验」页设置数据和模型。首次使用可直接到「工程」页点「创建示例工程」，生成合成图片、训练设置和包含共享层的起始图。
2. 在「工程」页选择一个新目录，点「导出工程」。需要移交最佳权重时，勾选「附带最佳权重」。程序会查找与当前图和数据匹配的已完成训练。
3. 复制整个工程目录。将同一份数据放到新位置，保留样本的相对路径和内容。打开工程中的 `project.json`，选择 ImageFolder 目录、分割 CSV 或影像与掩码所在目录，然后点「核对数据并恢复画布」。分割工程自带相对路径清单，原 CSV 使用绝对路径时可以直接选择新的数据根目录。
4. 在设备中选择 `cpu` 或一个可用的 `cuda:N`，点「运行保存的工程」。该按钮运行保存的 `model.py` 和工程配置。需要运行画布上的新改动时，可使用「实验」页或把修改导出到新的工程目录。

| 文件 | 用途 |
| --- | --- |
| `project.json` | 工程版本、模型标识、实验配置和画布布局 |
| `model.py` | 可阅读的模型源码。修改受支持的层参数、调用和连线后，重开工程可恢复实际结构 |
| `dataset.json` | 数据锁，记录样本相对位置、内容摘要、预处理和原有划分 |
| `dataset-pairs.csv` | 分割工程的便携配对清单 |
| `run.py`、`runtime/`、依赖清单 | 在安装依赖的 Python 环境中检查、训练和推理 |
| `weights/` | 可选的最佳权重与推理所需的运行快照 |

工程数据锁用于核对接收者手头的数据。数据要在新位置重新绑定，训练按原有种子和划分启动新实验。改动 `model.py` 后，如需使用新结构做推理，应重新训练。分割工程清单可能含原样本和分组编号，分享前按研究授权检查。

## 在独立 Python 环境中运行

使用 Python 3.12，新建虚拟环境后按导出工程里的 `README.md` 安装 `requirements.txt` 和 PyTorch。下例为 CPU 运行。`DATA_PATH` 为 ImageFolder 目录、分割 CSV 或影像配对根目录。分割 CSV 需要单独的影像根目录时，给 `check`、`train` 同时加 `--source-root PATH`。

```powershell
python -I run.py check --data DATA_PATH
python -I run.py train --data DATA_PATH --device cpu --out runs/replay-01
python -I run.py predict --source IMAGE_OR_FOLDER --weights runs/replay-01 --device cpu --out runs/predict-01
```

分割推理可以加 `--mode sliding` 和 `--overlap 0.25`。每个输出目录对应一次新的运行。工程已附带对应权重时，推理可省略 `--weights`。独立推理导出分类分数或原始网格的分割掩码。

## 查看记录和对照

每次运行的 `project-run.json` 会连接到模型快照、指标、进度日志、权重和预测产物。打开底部「工程运行记录」，可选择本地历史或打开迁移过来的记录目录，核对产物摘要，查看 A/B 指标和训练、验证损失。测试样本、标签、预处理和评价协议相符时显示差值。可从记录打开所在目录或进入分割结果视图。

Windows 桌面版可双击 `studio/启动模路.cmd`。手动构建使用 `./studio/build-s4-desktop.ps1 -Stage S10`。便携包位于 `studio/dist/s10/ModelCircuitStudio/`，移动时请保留整个目录。工作区位于 `%LOCALAPPDATA%\ModelCircuitStudio-S10`。
