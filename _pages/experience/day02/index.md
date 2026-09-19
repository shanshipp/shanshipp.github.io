---
layout: archive
permalink: /experience/day02/
title: "Day 02 — Tensor Shape, Linear Maps & Gradients / 张量形状、线性变换与梯度"
author_profile: true
---

Day 02 connects PyTorch tensor operations with the mathematics behind a linear layer. The central thread is:

```text
Shape -> Broadcasting -> Matrix Multiplication -> Y = XW + b
      -> Loss -> Gradient -> Parameter Update
```

## 1. Read the meaning before the numbers

For an image batch `x.shape == [32,3,224,224]`, the dimensions mean `[B,C,H,W]`: batch, channel, height, and width.

| Expression | Output shape | Interpretation |
|---|---|---|
| `x[0]` | `[3,224,224]` | select one image; the batch dimension disappears |
| `x[:,0]` | `[32,224,224]` | select one channel from every image |
| `x[:8]` | `[8,3,224,224]` | keep the first eight images |
| `x[:,:,:32,:32]` | `[32,3,32,32]` | crop the same region from every image |

A concrete index usually removes a dimension; a slice usually preserves it.

Three shape operations answer different questions:

- `view` / `reshape`: how should the same elements be grouped?
- `permute`: in what order should the axes appear?
- `unsqueeze` / `squeeze`: where should a length-one dimension be added or removed?

```python
x_flat = x.view(x.shape[0], -1)   # [32,3,224,224] -> [32,150528]
x_bhwc = x.permute(0, 2, 3, 1)   # [B,C,H,W] -> [B,H,W,C]
```

Flattening preserves the number and type of elements. Here both tensors contain `4,816,896` `float32` values.

## 2. Broadcasting: align from the right

When two shapes differ, compare dimensions from right to left. A pair is compatible when the sizes are equal or one side is `1`; missing leading dimensions can be treated as `1`.

```text
[32,768] + [768]  -> [32,768]
[32,768] + [32,1] -> [32,768]
[32,768] + [32]   -> incompatible
[2,3,4] + [3,1]   -> [2,3,4]
```

The first case has a useful model interpretation: 32 samples each have 768 features, and the same 768-dimensional bias vector is added to every sample.

## 3. Matrix multiplication and a linear layer

For two-dimensional tensors:

```text
[a,b] @ [b,c] -> [a,c]
```

The inner dimensions must match. In deep-learning notation:

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

`W` maps each 784-dimensional input to 128 output features. Broadcasting then adds the 128 bias values to each of the 32 samples. This is the core idea behind `nn.Linear(784, 128)`.

Do not confuse the operators: `*` is element-wise multiplication, while `@` is matrix multiplication.

## 4. Reduction, max, and argmax

For `scores.shape == [32,10]`:

```text
scores.mean()                    -> []
scores.mean(dim=0)               -> [10]
scores.mean(dim=1)               -> [32]
scores.mean(dim=1, keepdim=True) -> [32,1]
```

A reduction removes the selected dimension by default. `keepdim=True` retains it with length `1`, which is often convenient for later broadcasting.

```python
max_values, max_indices = scores.max(dim=1)
predicted_classes = scores.argmax(dim=1)
```

All three results have shape `[32]`: one result per sample. `max` can return both values and indices; `argmax` returns only the indices. For class scores, those indices are the predicted class IDs.

## 5. From derivatives to gradients

A derivative measures local change. For `y=x²`,

$$
\frac{dy}{dx}=2x.
$$

At `x=5`, the derivative is `10`: a small change in `x` produces approximately ten times that change in `y`.

For a multivariable function, a partial derivative changes one variable while treating the others as constants:

$$
z=x^2+y^2,\qquad
\frac{\partial z}{\partial x}=2x,\quad
\frac{\partial z}{\partial y}=2y.
$$

The gradient collects the loss derivatives with respect to all parameters. For

$$
L(w_1,w_2)=w_1^2+w_2^2,
$$

the gradient at `(3,4)` is `[6,8]`. With learning rate `0.1`:

$$
[3,4]-0.1[6,8]=[2.4,3.2],
$$

and the loss falls from `25` to `16`. This illustrates the basic update rule: move parameters in the negative-gradient direction.

## 6. Chain rule and a squared-error loss

The chain rule differentiates a complex expression by splitting it into simple steps. For

$$
L=(wx+b-y)^2,
$$

the parameter gradients are

$$
\frac{\partial L}{\partial w}=2(wx+b-y)x,
\qquad
\frac{\partial L}{\partial b}=2(wx+b-y).
$$

Using `x=2,w=3,b=1,y=10`: prediction `=7`, error `=-3`, loss `=9`, `∂L/∂w=-12`, and `∂L/∂b=-6`.

```text
Prediction -> Error -> Loss -> Gradient -> Parameter Update
```

The chain rule is the mathematical bridge to computational graphs and backpropagation, which are the next topics to study.

## Quick checklist

- Explain every dimension before manipulating a tensor.
- Check broadcasting compatibility from the right.
- Preserve the batch dimension when flattening samples.
- Remember `[B,I] @ [I,O] -> [B,O]`.
- Reduction removes a dimension unless `keepdim=True`.
- Gradient descent updates parameters opposite the gradient.

[← Back to Experience / 返回经验页](/experience/)
