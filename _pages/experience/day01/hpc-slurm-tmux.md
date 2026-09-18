---
layout: archive
permalink: /experience/day01/hpc-slurm-tmux/
title: "HPC, Slurm & tmux"
author_profile: true
---

## 核心架构

```text
AI HPC Studio → login node → Slurm → GPU compute node
```

- 登录后首先进入 login node，用于提交和管理任务。
- 登录节点没有 `nvidia-smi`、看不到 GPU 属于正常现象；不要直接在登录节点运行 GPU 计算。
- 集群使用 Slurm 管理 GPU 资源；已通过交互任务进入 GPU compute node。
- 在计算节点验证 NVIDIA A800，显存 81920 MiB（约 80GB）。

## 已实际使用的 A800 申请命令

```bash
srun -p <gpu_partition> --gres=gpu:<gpu_type>:1 --time=00:10:00 --pty bash
```

含义：在指定 GPU 分区申请 1 张目标 GPU、最长 10 分钟，并启动交互式 Bash。

## 常用 Slurm 命令

- `sinfo`：查看分区、节点和 GPU 资源。
- `squeue -u $USER`：查看自己的排队或运行任务。
- `srun`：申请资源并运行交互式或即时任务。

## tmux 与 Slurm

| 工具 | 职责 |
|---|---|
| tmux | 保持终端会话，断开连接后可恢复窗口与程序界面 |
| Slurm | 调度 GPU、CPU、节点和运行时间等计算资源 |

tmux 不会分配 GPU；Slurm 也不负责保存终端界面。两者可组合使用，但职责不同。

本地 WSL 已实际完成 tmux 会话的创建、detach 和 attach。实验室登录节点检查结果为未安装 tmux，且没有可用的 tmux module；不要把本地 tmux 状态误认为服务器状态。
