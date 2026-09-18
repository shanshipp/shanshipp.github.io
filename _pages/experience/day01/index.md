---
layout: archive
permalink: /experience/day01/
title: "Day 01 — Foundations / 基础"
author_profile: true
---

## 三个核心目标

- [x] 跑通 WSL2 + RTX 5060 Ti 的 PyTorch GPU 环境。
- [x] 建立 AI / ML / DL 与监督学习的基本框架。
- [x] 掌握 Tensor、Shape、Device 的核心操作。

## 完成状态

本地 GPU、VS Code WSL、A800 与 Tensor 操作均验证成功。

## 环境摘要

| 项目 | 最终状态 |
|---|---|
| 本地 | WSL2 + Ubuntu 24.04；RTX 5060 Ti |
| 软件 | Python 3.12.14；PyTorch 2.14.0+cu130；torchvision 0.29.0+cu130 |
| CUDA | 可用，设备 `cuda:0` |
| 服务器 | Slurm 成功申请 A800 80GB |

## 笔记索引

- [环境配置](/experience/day01/environment/)
- [机器学习基础](/experience/day01/ml-fundamentals/)
- [PyTorch Tensor](/experience/day01/pytorch-tensor/)
- [HPC / Slurm / tmux](/experience/day01/hpc-slurm-tmux/)
- [命令速查](/experience/day01/cheatsheet/)

## 最重要的结论

1. WSL 使用 Windows NVIDIA 驱动，不另装驱动。
2. `nvidia-smi` 的 CUDA Version 不等于 CUDA Toolkit；没有 `nvcc` 仍可运行 PyTorch。
3. Train 学、Validation 选、Test 最终评。
4. Tensor 优先关注 `shape`、`dtype`、`device`。
5. `permute` 重排维度；`cat` 不增维，`stack` 增维。
6. tmux 保持会话；Slurm 管理计算资源。

[← Back to Experience / 返回经验页](/experience/)
