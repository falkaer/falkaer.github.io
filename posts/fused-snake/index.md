+++
title = "Fused snakes"
date = Date("2023-09-20", "yyyy-mm-dd")
description = "Writing efficient PyTorch kernels with analytic gradients and Triton."
+++

This (first!) blog post of mine is about how to improve the performance of custom operations in PyTorch by interfacing with the 

As part of my masters thesis I wanted to experiment with the _Snake_ activation function{{cite ziyin2020neural}} for waveform music generation. I was unable to find an efficient implementation of the function in PyTorch, so I ended up writing my own implementation using the Triton{{cite tillet2019triton}} GPU compiler. This post details how I arrived at my final implementation of the Snake function and demonstrates how to interface directly with both PyTorch's automatic differentiation package as well as Triton for significant memory and compute performance improvements.

The Snake function is defined element-wise on each input $x \in \mathbb{R}$ as
$$\mathrm{Snake}(x; \alpha) = x + \frac{1}{\alpha} \sin^2(\alpha x),$$
where $\alpha \in \mathbb{R}$ is a learnable per-channel parameter, analogous to the slope in parametric ReLUs. Visually inspecting the function for a few choices of $\alpha$ shows how the function imposes sine-like modulations onto the signal, with $\alpha$ controlling the frequency of the modulations.

{{include posts/fused-snake/snake_graph.svg}}

We can implement the function in plain PyTorch using only a few lines of code for an input with shape $(\dots, C, N)$ where $C$ is the number of channels and $N$ is sequence length:

```python
import torch
import torch.nn as nn

class Snake1D(nn.Module):
    def __init__(self, num_channels, init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(init * torch.ones(num_channels))

    def forward(self, x):
        alpha = self.alpha[..., None]
        return x + torch.sin(alpha * x) ** 2 / alpha
```

# Understanding autograd and memory usage
If we try to train a model using this implementation, we will immediately notice that the peak memory use is much higher than when using built-in activation functions like ReLU.

{{todo "Plot of memory use for activation functions"}}

PyTorch provides the [autograd package](https://pytorch.org/docs/stable/autograd.html) for automatic differentiation of PyTorch functions. This package is used to implement backpropagation, which PyTorch uses to optimize neural networks.

When the forward pass of a network is executed, a computational graph is traced by recording the order and operands of PyTorch functions as they are invoked. Then, during the backward pass, the computational graph is traced in reverse to compute gradients of the loss function with respect to parameters.

Conceptually, the backpropagation algorithm allows to efficiently compute the gradients of a complicated loss function requiring only the analytic derivation of gradients between each node in the computational graph. This simplifies the computation greatly, but requires storing the intermediate results at every node to evaluate the gradient during backpropagation --- a compute/memory trade-off.

Going back to the Snake function, we see that the straight-forward implementation above requires storing four intermediate results of the same shape as the input. In neural networks where activations are applied after nearly every layer, these intermediate values can quickly dominate the peak memory use of the network during training.

## Providing gradients

A solution for bringing down the memory use is to directly provide the analytic gradients of the Snake function so that PyTorch can represent it as a single node in the computational graph.

Let us first consider the case with a scalar input $x$ and a single channel with parameter $\alpha$. Let $y = x + \frac{1}{\alpha} \sin^2(\alpha x)$ be the output of the Snake function, and $\phi$ the scalar loss value to be minimized by the network. Then the gradients we need to provide for PyTorch are
$$\frac{\partial \phi}{\partial x} = \frac{\partial \phi}{\partial y} \frac{\partial y}{\partial x}, \qquad \frac{\partial \phi}{\partial \alpha} = \frac{\partial \phi}{\partial y} \frac{\partial y}{\partial \alpha},$$
where $\frac{\partial \phi}{\partial y}$ is the gradient of the loss with respect to the output value, which PyTorch provides to our autograd function as part of the backward pass.

The gradient of the function output with respect to input is straight-forward to evaluate,

$$\begin{align}
\frac{\partial y}{\partial x} &{}= \frac{\partial}{\partial x} \Big(x + \frac{1}{\alpha} \sin^2(\alpha x)\Big) \notag \\
&{}= 1 + 2 \sin(\alpha x) \cos(\alpha x) \notag \\
&{}= 1 + \sin(2 \alpha x),
\end{align}$$

and the gradient with respect to the learnable parameter $\alpha$,

$$\begin{align}
\frac{\partial y}{\partial \alpha} &{}= \frac{\partial}{\partial \alpha} \Big(x + \frac{1}{\alpha} \sin^2(\alpha x)\Big) \notag \\
&{}= \frac{1}{\alpha} x \sin(2 \alpha x) - \frac{1}{\alpha^2} \sin^2(\alpha x).
\end{align}$$

We have now derived everything we need for optimizing the Snake function with respect to a single scalar input $x$ and parameter $\alpha$, but in practice we apply the function across multiple timesteps, channels and batches. Seeing as every input $x$ only influences the output $y$ in the same position we can apply the scalar case $\frac{\partial \phi}{\partial x}$ as-is, but since the per-channel parameter $\alpha$ influences all outputs $\mathbf{y} = \{y_i\}_{i = 0}^N$ in the same channel, $\frac{\partial \phi}{\partial \mathbf{y}}$ is now a row-vector while $\frac{\partial \mathbf{y}}{\partial \alpha}$ is a column-vector such that $\frac{\partial \phi}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \alpha}$ is a dot-product, and we obtain

$$\begin{align}
\frac{\partial \phi}{\partial \alpha} &{}= \frac{\partial \phi}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \alpha} \notag \\
&{}= \sum_{i=1}^N \frac{\partial \phi}{\partial y_i} \frac{\partial y_i}{\partial \alpha} \notag \\
&{}= \frac{1}{\alpha} \sum_{i=1}^N \frac{\partial \phi}{\partial y_i} \Big(x_i \sin(2 \alpha x_i) - \frac{1}{\alpha} \sin^2(\alpha x_i)\Big),
\end{align}$$

which can be independently computed for each per-channel parameter. We can now implement the custom autograd function in PyTorch:

```python
import torch
import torch.nn as nn

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        alpha = alpha[..., None]
        return x + torch.sin(alpha * x) ** 2 / alpha
    
    # @staticmethod
    # def backward(ctx, out_grad):
    #     x, alpha = ctx.saved_tensors
    #     ax = alpha[..., None] * x
    #     sin2ax = torch.sin(2 * ax)
    #     sinax = torch.sin(ax)
    #     dydx = 1 + sin2ax
    #     dyda = x * sin2ax - sinax ** 2 / alpha[..., None]
    #     grad_x = dydx * out_grad
    #     grad_a = torch.sum(out_grad * dyda, dim=-1) / alpha
    #     return grad_x, grad_a

    @staticmethod
    def backward(ctx, out_grad):
        x, alpha = ctx.saved_tensors
        sin2ax = torch.sin(2 * alpha[..., None] * x)
        sinax = torch.sin(alpha[..., None] * x)
        dydx = 1 + sin2ax
        dyda = x * sin2ax - sinax ** 2 / alpha[..., None]
        grad_x = dydx * out_grad
        grad_a = torch.sum(out_grad * dyda, dim=-1) / alpha
        return grad_x, grad_a
    
class Snake1D(nn.Module):
    def __init__(self, num_channels, init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(init * torch.ones(num_channels))
    
    def forward(self, x):
        return SnakeFunction.apply(x, self.alpha)

x = torch.randn(5, 3, 10, requires_grad=True)
S = Snake1D(3)

y = S(x)
L = y.sum()
L.backward()

```

https://lucidmanager.org/productivity/hugo-bibliography/

# Fusing kernels

The computational side of things does not look much better --- awf

{include {svg posts/fused-snake/snake_timings.svg}}

Forward pass:

highlight python "hl_lines=30-31"
```python
import torch

import triton
import triton.language as tl

@triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
            triton.Config({}, num_warps=16),
        ],
        key=['N'],
)
@triton.jit
def _snake_fwd_triton(X, OUT, ALPHA,
                      X_stride1, X_stride2, X_stride3,
                      OUT_stride1, OUT_stride2, OUT_stride3,
                      A_stride, C, N,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    X = X + batch_idx * X_stride1 + channel_idx * X_stride2
    x = tl.load(X + offsets * X_stride3, mask=offsets < N)
    alpha = tl.load(ALPHA + channel_idx * A_stride)
    
    sinax = tl.sin(alpha * x)
    out = x + sinax * sinax / alpha
    
    OUT = OUT + batch_idx * OUT_stride1 + channel_idx * OUT_stride2
    tl.store(OUT + offsets * OUT_stride3, out, mask=offsets < N)

def snake_fwd(x, alpha, out=None):
    if out is None:
        out = torch.empty_like(x)
    B, C, N = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
    grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
    _snake_fwd_triton[grid](x, out, alpha,
                            x.stride(0), x.stride(1), x.stride(2),
                            out.stride(0), out.stride(1), out.stride(2),
                            alpha.stride(0), C, N, BLOCK_SIZE)
    return out
```

Backward pass:

highlight python "hl_lines=32-34 38 43-44"
```python
import torch

import triton
import triton.language as tl

@triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
            triton.Config({}, num_warps=16),
        ],
        reset_to_zero=['DYDA'],
        key=['N'],
)
@triton.jit
def _snake_bwd_triton(X, ALPHA, GRAD, DYDX, DYDA,
                      X_stride1, X_stride2, X_stride3,
                      GRAD_stride1, GRAD_stride2, GRAD_stride3,
                      DYDX_stride1, DYDX_stride2, DYDX_stride3,
                      DYDA_stride, ALPHA_stride, C, N,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    X = X + batch_idx * X_stride1 + channel_idx * X_stride2
    x = tl.load(X + offsets * X_stride3, mask=offsets < N, other=0)
    alpha = tl.load(ALPHA + channel_idx * ALPHA_stride)
    
    ax = alpha * x
    sin2ax = tl.sin(2 * ax)
    sinax = tl.sin(ax)
    
    GRAD = GRAD + batch_idx * GRAD_stride1 + channel_idx * GRAD_stride2
    grad = tl.load(GRAD + offsets * GRAD_stride3, mask=offsets < N, other=0)
    dydx = (sin2ax + 1) * grad
    
    DYDX = DYDX + batch_idx * DYDX_stride1 + channel_idx * DYDX_stride2
    tl.store(DYDX + offsets * DYDX_stride3, dydx, mask=offsets < N)
    
    dyda = tl.sum((x * sin2ax - sinax * sinax / alpha) * grad, axis=0) / alpha
    tl.atomic_add(DYDA + channel_idx * DYDA_stride, dyda)

def snake_bwd(x, alpha, grad):
    B, C, N = x.shape
    dydx = torch.empty_like(x, dtype=grad.dtype)
    dyda = torch.zeros_like(alpha, dtype=alpha.dtype)
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2 ** 14)
    grid = lambda meta: (B * C, triton.cdiv(N, meta['BLOCK_SIZE']))
    _snake_bwd_triton[grid](x, alpha, grad, dydx, dyda,
                            x.stride(0), x.stride(1), x.stride(2),
                            grad.stride(0), grad.stride(1), grad.stride(2),
                            dydx.stride(0), dydx.stride(1), dydx.stride(2),
                            dyda.stride(0), alpha.stride(0), C, N,
                            BLOCK_SIZE)
    return dydx, dyda
```

With autograd:

```python
import torch
import torch.nn as nn

class SnakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return snake_fwd(x, alpha)
    
    @staticmethod
    def backward(ctx, out_grad):
        x, alpha = ctx.saved_tensors
        return snake_bwd(x, alpha, out_grad)

class Snake1D(nn.Module):
    def __init__(self, num_channels, init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(init * torch.ones(num_channels))
    
    def forward(self, x):
        return SnakeFunction.apply(x, self.alpha)
```

{{ bibliography }}
