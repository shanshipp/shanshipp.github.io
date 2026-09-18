---
layout: archive
permalink: /experience/day01/environment/
title: "Environment Setup / 环境配置"
author_profile: true
---

## 1. 最终环境

| 项目 | 已验证状态 |
|---|---|
| 主机系统 | Windows 11 |
| Linux 环境 | WSL2 + Ubuntu 24.04 |
| 本地 GPU | NVIDIA GeForce RTX 5060 Ti 8GB |
| 环境管理 | Miniforge / Conda 26.7.2 |
| Python 环境 | `dl`，Python 3.12.14 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| PyTorch CUDA Runtime | 13.0 |
| CUDA 可用 | `torch.cuda.is_available() == True` |
| GPU 设备 | `cuda:0` |

验证包括：PyTorch 正确识别 RTX 5060 Ti、成功创建 CUDA Tensor，并完成 `5000 × 5000` GPU 矩阵乘法。

## 2. WSL2 与 GPU

### 驱动关系

```text
Windows NVIDIA Driver
        ↓ 提供 WSL GPU 接口
WSL2 / Ubuntu
        ↓
PyTorch CUDA Runtime
        ↓
RTX 5060 Ti
```

WSL2 使用 Windows 主机安装的 NVIDIA 驱动。不要在 WSL Ubuntu 内再次安装 `nvidia-driver-*`，否则可能造成驱动冲突。

### `nvidia-smi` 检查什么

```bash
nvidia-smi
```

重点检查：

- GPU 名称是否为 RTX 5060 Ti。
- Driver Version 是否正常显示。
- 显存容量、当前占用、温度和 GPU 利用率是否可读取。
- 命令能正常运行，说明 Windows 驱动到 WSL 的 GPU 映射已生效。

### 三个容易混淆的“CUDA”

| 名称 | 含义 | Day 1 状态 |
|---|---|---|
| NVIDIA Driver | 操作系统与 GPU 通信所需的驱动 | 已正常工作 |
| CUDA Toolkit / `nvcc` | 开发、编译自定义 CUDA 程序的工具链 | 未安装，当前不需要 |
| PyTorch CUDA Runtime | PyTorch wheel 携带的 CUDA 运行库 | 已安装并正常运行 |

`nvidia-smi` 顶部的 **CUDA Version** 表示当前驱动最高支持的 CUDA 能力，不代表 Ubuntu 已安装同版本 CUDA Toolkit。`nvcc --version` 返回 `command not found`，只说明没有 Toolkit 编译器。

本阶段目标是运行 PyTorch，而不是编译自定义 CUDA 代码。因此，只要 PyTorch 自带的 CUDA Runtime 与驱动兼容，即使没有 `nvcc`，GPU 训练仍可正常运行。

## 3. Linux 基础工具

### 安装与验证

```bash
sudo apt update
sudo apt install -y build-essential htop tree vim unzip zip
```

`build-essential` 会安装常用编译组件，包括 `gcc`、`g++` 和 `make`。

| 工具 | 用途 | Day 1 验证 |
|---|---|---|
| `git` | 版本控制 | 2.43.0 |
| `curl` | HTTP 请求、下载、网络排查 | 已可用 |
| `wget` | 下载文件 | 1.21.4 |
| `gcc` / `g++` | C / C++ 编译器 | 13.3.0 |
| `make` | 自动化构建 | 4.3 |
| `tmux` | 保持和恢复终端会话 | 3.4 |
| `htop` | 查看进程与资源占用 | 3.3.0 |
| `tree` | 树状显示目录 | 2.1.1 |
| `vim` | 终端文本编辑器 | 9.1 |
| `unzip` / `zip` | 解压与压缩 | 已可用 |

```bash
git --version
gcc --version
g++ --version
make --version
tmux -V
htop --version
tree --version
vim --version
```

## 4. Miniforge、Conda 与 Python

Miniforge 提供 Conda 环境管理。`base` 是 Miniforge 自身的基础环境；`dl` 是深度学习项目的独立环境。

独立环境的作用：

- 隔离不同项目的 Python 和依赖版本。
- 避免把实验依赖全部装入 `base`。
- 降低升级、卸载包时破坏其他项目的风险。
- 便于记录和复现实验环境。

```bash
conda create -n dl python=3.12 -y
conda activate dl
which python
python --version
pip --version
```

通过标准：终端前缀显示 `(dl)`；`which python` 指向 `miniforge3/envs/dl/bin/python`；Python 为 3.12.14。运行 `pip install` 前应先确认当前处于 `dl`，避免把包安装到 `base` 或系统 Python。

## 5. PyTorch 安装与 GPU 验证

### 最终成功的安装版本

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

最终安装结果为 `torch 2.14.0+cu130` 和 `torchvision 0.29.0+cu130`。这里的 `cu130` 表示该 PyTorch 构建携带 CUDA 13.0 Runtime，不要求系统安装 CUDA Toolkit 13.0。

### 基础验证

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('Compute capability:', torch.cuda.get_device_capability(0))"
```

实际结果：

- PyTorch：`2.14.0+cu130`
- CUDA available：`True`
- PyTorch CUDA：`13.0`
- GPU：`NVIDIA GeForce RTX 5060 Ti`
- Compute capability：`(12, 0)`

### GPU 计算验证

```bash
python -c "import torch; x=torch.randn(5000,5000,device='cuda'); y=x@x; print('shape:',y.shape); print('device:',y.device); print('mean:',y.mean().item())"
```

通过标准：计算无报错，结果 shape 为 `[5000, 5000]`，device 为 `cuda:0`。这比只查看版本号更可靠，因为它实际完成了 Tensor 创建和 GPU 矩阵运算。

## 6. VS Code + WSL

使用的扩展：

- **WSL**：让 Windows VS Code 连接并运行在 WSL 环境中。
- **Python**：解释器选择、运行和调试。
- **Pylance**：补全、类型分析与跳转。
- **Jupyter**：运行 Notebook。

在 WSL 项目目录中启动：

```bash
cd ~/research
code .
```

VS Code 左下角应显示 WSL 远程环境。若命令面板没有 `Python: Select Interpreter`，通常是 Python 扩展只安装在 Windows 侧；应在扩展页面选择 **Install in WSL**。

随后执行 `Python: Select Interpreter`，选择 `dl` 环境。新建终端后还应检查：

```bash
conda activate dl
which python
```

编辑器选择的解释器与终端环境是两个相关但不同的状态。即使编辑器已选 `dl`，终端仍可能显示 `(base)`；为避免 `pip install` 装错位置，科研终端也应切换到 `(dl)`。

最终通过 `learning/test.py` 验证：脚本使用 `dl` 解释器运行，能够导入 PyTorch，并正确识别 CUDA 与 RTX 5060 Ti。

## 7. 环境验收 Checklist

- [x] WSL2 + Ubuntu 24.04 可正常使用。
- [x] `nvidia-smi` 识别 RTX 5060 Ti。
- [x] Linux 基础工具安装并验证。
- [x] Miniforge 和独立 `dl` 环境可用。
- [x] Python 3.12.14、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130。
- [x] `torch.cuda.is_available()` 返回 `True`。
- [x] GPU 矩阵乘法成功，结果位于 `cuda:0`。
- [x] VS Code 以 WSL 模式打开项目并使用 `dl` 解释器。
