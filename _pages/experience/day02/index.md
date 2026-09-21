---
layout: archive
permalink: /experience/day02/
title: "Day 02 — Tensor Shape, Linear Maps & Gradients / 张量形状、线性变换与梯度"
author_profile: true
---

![Day 2 思维导图：Shape、Broadcasting、Linear 与 Gradient](/images/learning/day02-mindmap.svg)

Day 02 把 PyTorch Tensor 操作与线性层背后的数学连接起来，主线是：

```text
Shape -> Broadcasting -> Matrix Multiplication -> Y = XW + b
      -> Loss -> Gradient -> Parameter Update
```

## 1. 先读维度语义，再看数字

对图像 Batch `x.shape == [32,3,224,224]`，`[B,C,H,W]` 分别表示 Batch、Channel、Height 和 Width。

| 表达式 | 输出 Shape | 含义 |
|---|---|---|
| `x[0]` | `[3,224,224]` | 选一张图，Batch 维消失 |
| `x[:,0]` | `[32,224,224]` | 取每张图的第一个通道 |
| `x[:8]` | `[8,3,224,224]` | 保留前八张图 |
| `x[:,:,:32,:32]` | `[32,3,32,32]` | 从每张图裁取相同区域 |

具体 index 通常删除对应维度，slice 通常保留对应维度。

三类 Shape 操作回答不同问题：

- `view` / `reshape`：同一批元素应如何重新分组？
- `permute`：各个轴应按什么顺序排列？
- `unsqueeze` / `squeeze`：在哪里增加或删除长度为 `1` 的维度？

```python
x_flat = x.view(x.shape[0], -1)   # [32,3,224,224] -> [32,150528]
x_bhwc = x.permute(0, 2, 3, 1)   # [B,C,H,W] -> [B,H,W,C]
```

Flatten 不改变元素总数和数据类型；这里两个 Tensor 都包含 `4,816,896` 个 `float32` 元素。

## 2. Broadcasting：从最右侧对齐

两个 Shape 不同时，从右向左逐维比较。对应维度相同或其中一方为 `1` 时兼容；缺少的左侧维度可以视为 `1`。

```text
[32,768] + [768]  -> [32,768]
[32,768] + [32,1] -> [32,768]
[32,768] + [32]   -> 不兼容
[2,3,4] + [3,1]   -> [2,3,4]
```

第一种情况的模型语义是：32 个样本各有 768 个特征，同一个 768 维 bias 向量被加到每个样本上。

## 3. 矩阵乘法与线性层

二维 Tensor 的矩阵乘法规则是：

```text
[a,b] @ [b,c] -> [a,c]
```

中间两个维度必须相同。用深度学习语义表示：

```text
[Batch, Input Features] @ [Input Features, Output Features]
-> [Batch, Output Features]
```

```python
X = torch.randn(32, 784)
W = torch.randn(784, 128)
b = torch.randn(128)
Y = X @ W + b
```

`W` 把每个 784 维输入映射到 128 个输出特征；Broadcasting 再把 128 个 bias 加到 32 个样本中的每一个。这就是理解 `nn.Linear(784, 128)` 的核心。

易错点：`*` 是逐元素乘法，`@` 才是矩阵乘法。

## 4. Reduction、max 与 argmax

对 `scores.shape == [32,10]`：

```text
scores.mean()                    -> []
scores.mean(dim=0)               -> [10]
scores.mean(dim=1)               -> [32]
scores.mean(dim=1, keepdim=True) -> [32,1]
```

Reduction 默认删除被归约的维度。`keepdim=True` 会保留该维，但长度变为 `1`，便于后续 Broadcasting。

```python
max_values, max_indices = scores.max(dim=1)
predicted_classes = scores.argmax(dim=1)
```

三者的 Shape 都是 `[32]`，即每个样本得到一个结果。`max` 可同时返回最大值和下标；`argmax` 只返回下标。对分类分数而言，这些下标就是预测类别编号。

## 5. 从导数到梯度

导数描述单变量函数在某点附近的变化率。对 `y=x²`：

$$
\frac{dy}{dx}=2x.
$$

在 `x=5` 时，导数为 `10`：`x` 的微小变化会使 `y` 近似以十倍速度变化。

对多变量函数求某个变量的偏导时，暂时把其他变量视为常数：

$$
z=x^2+y^2,\qquad
\frac{\partial z}{\partial x}=2x,\quad
\frac{\partial z}{\partial y}=2y.
$$

Gradient 是 Loss 对各参数偏导数组成的向量。对

$$
L(w_1,w_2)=w_1^2+w_2^2,
$$

在 `(3,4)` 处，Gradient 为 `[6,8]`。学习率取 `0.1`：

$$
[3,4]-0.1[6,8]=[2.4,3.2],
$$

Loss 从 `25` 降到 `16`。这体现了基本更新规则：沿负梯度方向移动参数。

## 6. Chain Rule 与平方误差

Chain Rule 把复杂表达式拆成简单步骤再求导。对

$$
L=(wx+b-y)^2,
$$

参数梯度为

$$
\frac{\partial L}{\partial w}=2(wx+b-y)x,
\qquad
\frac{\partial L}{\partial b}=2(wx+b-y).
$$

代入 `x=2,w=3,b=1,y=10`：prediction `=7`，error `=-3`，loss `=9`，`∂L/∂w=-12`，`∂L/∂b=-6`。

```text
Prediction -> Error -> Loss -> Gradient -> Parameter Update
```

Chain Rule 是通向 Computational Graph 与 Backpropagation 的数学桥梁，也是下一阶段的学习重点。

## 核心检查清单

- 操作 Tensor 前先解释每个维度的语义。
- 从右侧检查 Broadcasting 是否兼容。
- Flatten 样本时保留 Batch 维。
- 记住 `[B,I] @ [I,O] -> [B,O]`。
- Reduction 默认删维，除非使用 `keepdim=True`。
- Gradient Descent 沿梯度反方向更新参数。

[← Back to Experience / 返回经验页](/experience/)
