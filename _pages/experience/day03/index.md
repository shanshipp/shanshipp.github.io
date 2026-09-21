---
layout: archive
permalink: /experience/day03/
title: "Day 03 — Autograd, Training Loop & MLP / 自动求导、训练闭环与多层感知机"
author_profile: true
---

Day 03 把前一天的梯度与链式法则，连接到 PyTorch 真正的模型训练过程：

```text
Computational Graph → Autograd → Gradient
→ Optimizer → Parameter Update → Loss ↓
```

## 1. 计算图与 Autograd

Computational Graph 记录 Tensor 与运算之间的依赖。Forward 沿图计算数值；调用 `backward()` 后，Autograd 从 Loss 出发反向遍历计算图，并用 Chain Rule 求出梯度。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = 3 * y
z.backward()
print(x.grad)
```

三个概念需要分开：

```text
requires_grad=True → 开启相关运算的梯度追踪
backward()         → 计算梯度
.grad              → 保存叶子 Tensor 累积得到的梯度
```

PyTorch 默认累积梯度，所以每轮训练前需要清空旧梯度。参数更新本身不需要进入计算图，可放在 `torch.no_grad()` 中；`detach()` 则把某个 Tensor 从当前计算图分离。

## 2. 从手动梯度到自动求导

用 `y = 3x + 2` 构造一个最小线性回归问题：

```python
x = torch.tensor([1., 2., 3.])
y = torch.tensor([5., 8., 11.])
w = torch.tensor(1.)
b = torch.tensor(0.)

prediction = w * x + b
loss = ((prediction - y) ** 2).mean()
error = prediction - y
dw = (2 * error * x).mean()
db = (2 * error).mean()
```

手动求导得到 `dw ≈ -26.6667`、`db = -12`。换成 Autograd 后：

```python
w = torch.tensor(1., requires_grad=True)
b = torch.tensor(0., requires_grad=True)
loss = ((w * x + b - y) ** 2).mean()
loss.backward()
```

`w.grad` 与 `b.grad` 得到相同结果。这说明 Autograd 没有改变数学，它只是沿计算图自动完成了手动推导的链式法则。

训练中还观察到学习率的典型影响：过大时 Loss 与参数迅速溢出并发散；过小时训练方向虽然正确，却几乎不前进；合适的学习率能让 Loss 稳定下降。

## 3. Optimizer 的职责

```python
optimizer = torch.optim.SGD([w, b], lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    prediction = w * x + b
    loss = ((prediction - y) ** 2).mean()
    loss.backward()
    optimizer.step()
```

```text
zero_grad() → 清空旧梯度
forward     → 用当前参数计算预测
loss        → 衡量预测误差
backward()  → 计算梯度并写入 .grad
step()      → 按优化规则与学习率更新参数
```

`backward()` 不直接修改 Parameter；求梯度与更新参数被明确拆开，因此可以使用 SGD、Adam 等不同优化算法。

## 4. `nn.Module`、Linear 与 ReLU

`nn.Module.__init__()` 定义模型有哪些层，`forward()` 定义数据如何依次通过这些层。对 `nn.Linear(in_features,out_features)`：

```text
weight.shape = [out_features,in_features]
bias.shape   = [out_features]
Y = X @ weight.T + bias
```

因此 `nn.Linear(784,128)` 会把 `[Batch,784]` 映射为 `[Batch,128]`，Batch 维保持不变。

若多个 Linear 之间没有激活函数，它们仍可合并为一个新的 Linear。ReLU 通过 `max(0,x)` 引入非线性，使网络能够表达更复杂的关系；它不改变 Tensor Shape，也没有可训练参数。

## 5. 第一个 MLP

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

```text
[32,784] → [32,256] → [32,128] → [32,10]
```

三个 Linear 一共有 6 个 Parameter Tensor：每层各一个 weight 与 bias。总可训练标量数为：

```text
784×256+256 + 256×128+128 + 128×10+10
= 235,146
```

神经网络学习的本质，就是利用梯度不断调整这些 Parameter，使 Loss 逐步减小。

## 6. 分类任务的完整闭环

回归预测连续数值，常用 MSELoss；分类预测离散类别，常用 CrossEntropyLoss。分类模型输出的是每个类别的原始分数 logits，而不是概率：

```text
logits.shape = [Batch,Classes]
labels.shape = [Batch]
loss.shape   = []
```

`logits.argmax(dim=1)` 得到每个样本分数最高的类别。`CrossEntropyLoss` 会利用整组 logits 与真实类别计算损失，通常不需要预先手动执行 Softmax。

最终训练闭环：

```python
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

这五行把完整逻辑串联起来：清梯度、前向计算、计算损失、反向求导、更新参数。

## 核心检查清单

- 计算图描述依赖，Autograd 沿图执行链式法则。
- `.grad` 默认累积，每轮训练前清空。
- `backward()` 求梯度，`step()` 更新参数。
- Linear 的参数 Shape 是 `[out,in]`，输出最后一维是 `out`。
- ReLU 提供非线性，但没有可训练参数。
- logits 是类别原始分数，标签是每个样本的类别编号。

[← Back to Experience / 返回经验页](/experience/)
