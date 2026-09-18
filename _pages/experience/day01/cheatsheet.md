---
layout: archive
permalink: /experience/day01/cheatsheet/
title: "Day 01 Cheatsheet / 命令速查"
author_profile: true
---

## Conda

```bash
conda activate dl
```
进入深度学习环境。

```bash
which python && python --version && pip --version
```
确认当前解释器和版本。

## PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
```
验证 PyTorch、CUDA 和 GPU。

```bash
python 01_tensor.py
```
运行 Tensor 练习。

## Linux

```bash
nvidia-smi
```
查看 GPU、驱动和显存。

```bash
pwd && whoami && hostname
```
确认目录、用户和主机。

## VS Code

```bash
code .
```
用 VS Code 打开当前 WSL 目录。

## tmux

```bash
tmux new -s test
```
创建名为 `test` 的会话。

```bash
# 在 tmux 内按 Ctrl+B，再按 D
```
分离当前会话。

```bash
tmux attach -t test
```
恢复会话。

## Slurm

```bash
sinfo -o "%P %a %l %D %G"
```
查看分区与 GPU 资源。

```bash
squeue -u $USER
```
查看自己的任务。

```bash
srun -p <gpu_partition> --gres=gpu:<gpu_type>:1 --time=00:10:00 --pty bash
```
交互式申请 1 张目标 GPU，最长 10 分钟。
