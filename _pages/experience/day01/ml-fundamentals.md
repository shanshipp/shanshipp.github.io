---
layout: archive
permalink: /experience/day01/ml-fundamentals/
title: "ML Fundamentals / 机器学习基础"
author_profile: true
---

## 概念地图

- **AI（人工智能）**：让机器表现出感知、推理或决策等智能行为的总领域。
- **ML（机器学习）**：AI 的子集，让模型从数据中学习规律。
- **DL（深度学习）**：ML 的子集，主要使用多层神经网络学习复杂表示。
- **CV（计算机视觉）**：处理图像和视频的方向。
- **NLP（自然语言处理）**：处理文本和语言的方向。
- **Multimodal（多模态）**：联合处理图像、文本、音频等多种数据形式。

关系：`AI → ML → DL`；CV、NLP、多模态是可使用深度学习解决的任务领域。

## 监督学习

```text
x → fθ(x) → ŷ → 与 y 比较得到 Loss → 计算 Gradient → 更新 θ
```

| 概念 | 一句话解释 |
|---|---|
| Data | 用于训练或评估的样本集合 |
| Feature | 模型接收的输入信息或输入表示 |
| Label | 样本对应的真实目标，如“猫” |
| Model | 将输入映射为预测的函数 `fθ` |
| Parameter | 模型在训练中自动学习的数值 `θ` |
| Hyperparameter | 人工设定的训练配置，如学习率 |
| Prediction | 模型输出的预测 `ŷ` |
| Loss | 衡量预测与真实目标差距的标量 |
| Training | 用数据反复计算 Loss 并更新参数的过程 |

## Train / Validation / Test

- **Train = 学**：计算梯度并更新模型参数。
- **Validation = 选**：选择模型和超参数，判断是否过拟合。
- **Test = 最终评**：训练和选择结束后，仅用于估计最终泛化能力。
- **Overfitting**：训练集表现很好，但新数据表现差，即“记住了训练集”。
- **Generalization**：模型在未见数据上仍能保持良好表现的能力。
- 不能反复看 Test 结果调模型，否则模型选择会间接适配 Test Set，最终成绩不再客观。

## Gradient Descent

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L
$$

- `θ_t`：第 `t` 步的模型参数。
- `L`：当前预测误差，即 Loss。
- `∇θL`：Loss 增长最快的方向，因此更新时取负号向下降方向走。
- `η`：学习率，控制每次更新的步长。
- `θ_(t+1)`：更新后的参数。

学习率过大可能跨过低点并振荡；训练就是重复计算梯度、沿降低 Loss 的方向更新参数。
