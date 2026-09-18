---
layout: archive
permalink: /experience/day01/pytorch-tensor/
title: "PyTorch Tensor"
author_profile: true
---

核心直觉：**Tensor + Shape + Device**。

## Tensor 创建

```python
torch.tensor([1, 2, 3])   # 从已有数据创建
torch.zeros(2, 3)         # 全 0
torch.ones(2, 3)          # 全 1
torch.randn(2, 3)         # 标准正态随机数
```

## 三个核心属性

- `shape`：各维度的长度，决定数据结构。
- `dtype`：元素的数据类型，如 `int64`、`float32`。
- `device`：Tensor 所在设备，如 `cpu`、`cuda:0`。

## CV Shape

```python
x = torch.randn(32, 3, 224, 224)
```

`[B, C, H, W] = [32, 3, 224, 224]`：32 张图；3 个 RGB 通道；高度 224；宽度 224。该 Tensor 是 4 维，共 `32×3×224×224 = 4,816,896` 个元素。

## 索引 / 切片

| 操作 | 含义 | 结果 shape |
|---|---|---|
| `x[0]` | 第 1 张图 | `[3,224,224]` |
| `x[0, 0]` | 第 1 张图的第 1 个通道 | `[224,224]` |
| `x[:, 0]` | 所有图的第 1 个通道 | `[32,224,224]` |
| `x[:10]` | 前 10 张图 | `[10,3,224,224]` |
| `x[5:15]` | 第 6～15 张图 | `[10,3,224,224]` |

规律：具体 index 通常使该维度消失；slice 通常保留该维度。

## Reduction

- `x.mean()` / `x.sum()`：对全部元素归约，结果是标量。
- `x.mean(dim=1)`：沿通道维求均值，shape 为 `[32,224,224]`。
- 指定 `dim` 后，该维度默认消失。
- `x.mean(dim=1, keepdim=True)`：保留通道维，shape 为 `[32,1,224,224]`。

## Shape 操作

| 操作 | 核心作用 |
|---|---|
| `reshape` | 改 shape，元素总数必须不变；`-1` 表示自动推算 |
| `view` | 类似 `reshape`，但对内存连续性要求更严格 |
| `permute` | 按给定顺序重排维度 |

```python
y = x.permute(0, 2, 3, 1)
```

`[B,C,H,W] → [B,H,W,C]`，即 `[32,3,224,224] → [32,224,224,3]`。`(0,2,3,1)` 表示新维度依次取原来的第 0、2、3、1 维，不是两两交换。

## 拼接

- `torch.cat([a, b], dim=...)`：沿已有维度拼接，不增加维度数量。
- `torch.stack([a, b], dim=...)`：新增一个维度。两个 `[2,2]` Tensor stack 后为 `[2,2,2]`。

实际验证：`cat(dim=0) → [4,2]`，`cat(dim=1) → [2,4]`；`stack(dim=0/1) → [2,2,2]`。

## Device

```python
gpu_tensor = tensor.to("cuda")  # 移到默认 GPU
cpu_tensor = gpu_tensor.to("cpu")
```

实际验证：`cpu → cuda:0 → cpu`。参与同一运算的 Tensor 通常必须位于同一设备。
