---
layout: archive
permalink: /experience/day04/
title: "Day 04 — Data, Training & Evaluation / 数据、训练与评估"
author_profile: true
---

![Day 4 思维导图：从样本到可复现训练](/images/learning/day04-mindmap.svg)

Day 04 把前一天的 MLP 接入真实图像数据，完成了从批量读取、训练、评估到保存和单图推理的流程。以后换成 CNN，数据与评估的骨架仍可沿用。

```text
Dataset → DataLoader → MLP → Loss → Backward → Optimizer
                                  ↓
                            Evaluation → Save → Inference
```

## 1. 样本、批次与 Shape

FashionMNIST 的训练集有 60,000 张灰度图。Dataset 返回一张图和对应的真实类别；DataLoader 把样本组成批次。图像 Tensor 的形状从 `[B,1,28,28]` 展平成 `[B,784]`，才能输入第一层 `nn.Linear(784,256)`；最后的 logits 为 `[B,10]`，每张图对应十个类别分数。

```python
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
x = torch.flatten(images, start_dim=1)
```

一个 epoch 遍历整个数据集。60,000 张图按每批 64 张，默认保留最后一批，共 `ceil(60000/64)=938` 次迭代；最后一批只有 32 张。因此应使用 `B` 或 `x.size(0)`，避免将 batch 大小写死。

## 2. 每批训练与指标

模型、输入和标签需要放在同一个设备。每批执行：

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`backward()` 计算并保存梯度，`step()` 才更新参数。分类预测由 `logits.argmax(dim=1)` 得到，Accuracy 则是预测正确数除以样本总数。整轮 Loss 应跨所有 batch 汇总；只打印最后一个 batch 的 Loss 不能代表整个 epoch。

一次三轮训练中，Train Loss 从 `0.5256` 降至 `0.3324`，Train Accuracy 从 `81.19%` 升至 `87.83%`。这些结果对应同一次训练过程，不是学习率对比实验。

## 3. 评估与泛化

评估循环只做前向计算：

```python
model.eval()
with torch.no_grad():
    for images, labels in eval_loader:
        predictions = model(images).argmax(dim=1)
```

`model.eval()` 切换模型的评估行为；`torch.no_grad()` 关闭梯度记录。二者职责不同。评估数据不执行 `backward()` 和 `optimizer.step()`。本次练习使用 FashionMNIST 官方测试集作评估，尚未另划验证集；正式实验应保留独立测试集用于最终报告。

第三轮评估 Accuracy 为 `86.85%`，与训练 Accuracy `87.83%` 接近。若训练指标持续提高而独立数据表现明显落后或下降，才应警惕过拟合。第一轮评估 Accuracy 高于训练 Accuracy 也可能发生：训练指标汇总了模型整轮逐步学习的表现，评估指标使用该轮训练结束后的模型。

## 4. 保存、推理与继续训练

`model.state_dict()` 保存模型状态。加载时要先构建相同网络结构，再载入状态。单张图像需加上 batch 维：`[1,28,28] → [1,1,28,28] → [1,10]`。当天完成了一张图的加载与正确预测。

若要从中断处继续训练，Checkpoint 还应记录已完成的 epoch 和优化器状态。对于 Adam，优化器状态包含历史统计量和步数；它不等于当前梯度。当天学习了保存与恢复方法，但没有留下实际续训输出。

学习率影响参数更新尺度。为公平比较不同学习率，可固定随机种子，并保持模型、数据、epoch 数和优化器类型一致。`1e-2`、`1e-3`、`1e-4` 的三组对比尚待实测，因此这里不判断哪组更好。

[← Back to Experience / 返回学习索引](/experience/)
