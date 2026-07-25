# Where cuDNN is structurally weak, and how to attack it

The goal is not parity. This is the measured map of cuDNN's architectural weaknesses and
which of them are worth attacking, based on profiling its actual kernels rather than
guessing.

## Do not fight the implicit GEMM

cuDNN's dense-convolution kernel issues **25,480 global load instructions where ours
issued 4,017,216** — 158x fewer for identical arithmetic — by staging tiles in shared
memory, blocking registers, and working in NHWC. It reaches 8.3 TFLOP/s where a naive
lowering reaches 1.9.

That is a decade of tuning against a well-understood problem. Reuse tiling closed most of
our gap (dense 3x3 went 0.22x to 0.64x) but the remainder is not where our advantage
lies. **Pick the battles below instead.**

## Flaw 1: it cannot fuse an epilogue, and the cost compounds

PyTorch cannot fuse through a cuDNN call. Every elementwise stage becomes a separate
kernel launch and a full round trip of the tensor through memory. Measured on
N16/C64->K64/28x28, CUDA graphs, true fp32:

| chain | cuDNN | marginal |
|---|---|---|
| conv only | 23.75 us | — |
| + bias | 26.59 us | **+2.84** |
| + bias + ReLU | 34.73 us | **+8.14** |
| + bias + scale + ReLU | 41.15 us | **+6.42** |

**The epilogue costs cuDNN 17.40 us against a 23.75 us convolution — 42% of its total.**

### But we pay for our epilogue too, and the head-to-head is much narrower

The obvious inference — that our epilogue is nearly free, so the gap widens fast with
depth — **is not what the head-to-head shows.** From the tracked bake-off, both lanes in
one session, cuDNN spreads 0.1-0.5%:

| chain | cuDNN | ours | ratio |
|---|---|---|---|
| conv + bias + ReLU | 31.5 | 30.2 | 1.04x |
| conv + bias + scale + ReLU | 36.9 | 34.8 | **1.06x** |
| marginal cost of the added stage | **+5.4 us** | **+4.6 us** | |

Our marginal cost is 4.6 us, not the ~free I expected. So on this shape the fusion
advantage is worth about **1.02x per epilogue stage, not 1.07x**. An isolated run of the
competitor alone had suggested 1.15x and 1.23x; that came from measuring the two lanes in
different sessions, which the tracked run supersedes.

The reason our epilogue is not free is visible in the limiter gate: this kernel is
**L2-bound at 66%**, not compute-bound, so adding operands and registers costs real time
rather than filling idle issue slots.

### Where fusion IS decisively worth it: memory-bound operators

Same run, same protocol:

| kernel | ratio | |
|---|---|---|
| depthwise_conv2d_3x3 (no epilogue) | 2.08x | |
| depthwise_conv2d_3x3_bias_relu | **3.01x** | **fusion is worth 1.45x here** |

That is the real fusion result: on a kernel already at 91% of the DRAM roofline, cuDNN's
extra epilogue passes each cost it another full traversal of the tensor, while ours cost
nothing because the data is already in registers.

**Action:** deepen epilogues on **memory-bound** operators, where each avoided pass is a
whole extra traversal for the competitor. On reuse-limited kernels such as dense 1x1 the
advantage is real but small, and should not be quoted as more than ~1.02x per stage.

## Flaw 2: it pays layout transforms we do not

Profiling cuDNN's "one convolution" found five kernels:

| kernel | time |
|---|---|
| `nchwToNhwcKernel` | 3.20 us |
| `sm86_xmma_fprop_implicit_gemm_...` | 31.58 us |
| `elementwise_kernel` (bias) | 6.27 us |
| **total** | **41.06 us** |

Its fast kernels want NHWC; PyTorch tensors are NCHW. So it converts in, computes, and
converts back. That is **3.20 us of pure overhead, ~8%**, on top of the epilogue cost.

We read whatever layout the specification declares, with no transform, because the index
map *is* the layout.

**Action:** this advantage is automatic and needs no work — but it disappears if a caller
hands us NHWC, so the catalog should keep NCHW shapes, which is what PyTorch models
actually produce.

## Flaw 3: it is a fixed library and cannot specialise on shape

We fully unroll the reduction, which folds every reduction index into a compile-time
constant and lets interval analysis prove bounds guards unnecessary — **476 guards elided**
on the depthwise kernel. cuDNN ships one binary that must handle every shape, so it cannot
take that specialisation.

**Action:** favour shapes where generic code pays most — unusual channel counts, small
spatial extents, odd strides — and specialise per shape at build time.

## Flaw 4: its tiling machinery buys nothing where there is no reuse

Depthwise convolution has almost no data reuse to exploit, so cuDNN's shared-memory
staging and register blocking cost generality without returning anything. Measured, it
reaches only 323 GB/s where we reach ~700.

| kernel | ours | cuDNN | ratio |
|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 74.0 | 222.9 | **3.01x** |
| depthwise_conv2d_3x3_bwd_data | 74.3 | 201.4 | **2.71x** |
| depthwise_conv2d_3x3 | 74.6 | 155.1 | **2.08x** |
| maxpool2d_2x2 | 166.4 | 225.9 | **1.36x** |

**Action:** depthwise, pooling and elementwise-dominated operators are where we already
win by 1.5-3.2x. Extend the catalog there first — grouped convolution, dilated depthwise,
separable stacks — because those are wins we keep rather than gaps we close.

## The targeting map

| operator class | our position | why | invest? |
|---|---|---|---|
| depthwise / grouped | **2.2-3.2x ahead** | no reuse for their tiling to exploit; we fuse | **yes, extend** |
| pooling / elementwise | **1.5x ahead** | at the DRAM roofline, they are not | **yes, extend** |
| deep epilogue on memory-bound ops | **fusion worth 1.45x** | each avoided pass is a whole traversal | **yes, highest value** |
| deep epilogue on reuse-limited ops | 1.06x, ~1.02x per stage | we pay for our epilogue too | modest |
| transposed conv | ~parity | exactness predicates cost us, generality costs them | opportunistic |
| dense conv, large C/K | **0.60x behind** | their implicit GEMM is genuinely excellent | **no, do not chase** |

The strategy that follows: **do not try to become cuDNN. Be the thing cuDNN cannot be** —
a compiler that specialises per shape and fuses arbitrarily deep chains, pointed at the
operator classes where a general library's machinery is dead weight.
