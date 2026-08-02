# What is actually missing: a whole-system audit

The question: what prevents adopting this kernel system on all open PRs and winning against
every competitor on every kernel? Answered from the code, not from impression.

## What the system expresses today

| capability | measured |
|---|---|
| ops the IR defines | **66** |
| ops the front end translates | **11** — `LoadInput`, `Mul`, `MatMul` ×3, `BatchMatMul`, `ReduceSum`, `ReduceMax`, `Conv2D`, `DepthwiseConv2D`, `ConvTranspose2D` |
| reduce kinds | **3** — `None`, `Sum`, `Max` |
| epilogue activations | **2** — `None`, `ReLU` |
| transcendental instructions emitted | **0** — no `ex2`, `lg2`, `rcp`, `sqrt`, `rsqrt` |
| element types | **1** — fp32 only; `PtxGraphEmitter` declines everything else |
| outputs per kernel | **1** — `CodegenKernelSpec.Output` is a single binding |
| tensor-core / `cp.async` instructions | **0** |
| engine code paths that dispatch to it | **0** |

So the system expresses exactly one family: **convolution and matmul with an optional bias
and ReLU, in fp32, producing one output.** That is a narrow slice of what the open PRs
contain, and it is the whole answer to the adoption question.

## PR by PR

| PR | kernel family | status | blocked by |
|---|---|---|---|
| 877 | convolution | **expressible** | — |
| 869 | row-sum reduction | **expressible** | — |
| 859 | dense GEMM / linear / LoRA | **partly** | LoRA is a two-matmul chain; needs op-chain fusion |
| 876 | global average pooling | blocked | no `Mean` reduce |
| 872 | per-sample MSE loss | blocked | no `Mean`; needs subtract-and-square before the reduce |
| 864 | pointwise GLU | blocked | no sigmoid/SiLU activation |
| 863 | residual LayerNorm + GELU | blocked | transcendentals, two-pass statistics, GELU |
| 868, 884 | softmax, log-softmax, log-sum-exp | blocked | `exp`/`log`, **two-pass reduction** (max then sum) |
| 874 | fused SGD-momentum | blocked | **multi-output** (updates parameter *and* momentum), in-place |
| 870 | embedding gather | blocked | data-dependent indexing (an index operand) |
| 871 | fp32→fp16 cast | blocked | dtype |
| 860 | mixed-precision decode | blocked | fp16 / W8A8 dtypes |
| 879 | spectral complex multiply | blocked | complex element type |
| 885 | scientific / hypercomplex / hyperbolic | blocked | non-real element types |
| 880 | RNG / stochastic | blocked | RNG state, no pure index→value form |
| 878 | recurrent state | blocked | sequential dependence across the reduction |
| 882 | dense solvers | blocked | iterative / sequential |
| 883 | vision family | unassessed | — |

**Two of nineteen are expressible today.** Not "the system needs wiring up" — it needs
capabilities it does not have.

## The blockers, ranked by PRs unlocked

| # | missing capability | unlocks |
|---|---|---|
| 1 | **Transcendentals + activation set** (`ex2`, `lg2`, `rcp`, `rsqrt`, tanh; sigmoid, SiLU, GELU) | 863, 864, 868, 884 |
| 2 | **Two-pass / fused statistics reduction** (max→sum, mean→variance) | 863, 868, 884 |
| 3 | **`Mean` reduce + scalar scale in the epilogue** | 876, 872 |
| 4 | **fp16 / bf16 element types** | 860, 871, and every tensor-core path later |
| 5 | **Multi-output specs** | 874, and every backward pass producing two gradients |
| 6 | **Data-dependent indexing** (gather/scatter) | 870 |
| 7 | **Non-real element types** (complex, quaternion) | 879, 885 |

Items 1–3 are the cheapest and unlock seven PRs between them. Item 2 is the one real
design change: `CodegenKernelSpec` has a single `Reduce`, and softmax needs a maximum and
then a sum over the same axis, with the first result feeding the second. That is a
two-kernel program — which is **exactly the shape `CodegenSplitReduction` already
produces**, so the machinery for sequencing two kernels through a temporary exists and is
verified.

## The second problem: nothing dispatches

`grep` for `PtxGraphEmitter` and `CodegenKernelCatalog` outside the codegen folder returns
nothing. No engine path selects a generated kernel. Even a fully expressive system would
change no user-visible behaviour, and every measured win — depthwise at 2.99×, maxpool at
1.41× — is currently unrealised outside the benchmark harness.

This is deliberate today (the promotion track is #886, held on evidence), but it means
"adopt on all PRs" has two independent meanings that should not be conflated: *expressible
by the system*, and *actually used by the engine*.

## The third problem: "win against all competitors" is not achievable by scheduling alone

The dense-3×3 investigation is the evidence. After per-dimension staging was implemented,
verified at `0.000E+000` and measured:

- L1 rose 64.08% → 77.45% (shared memory *is* L1TEX)
- L1 was flat across coarsening 2/4/8 — not the binding constraint
- `mio_throttle` 3.03% — the load pipe was never the bottleneck
- warps issue ~64% of the time, no unit above 64%

A **balanced** profile with a 1.5× gap is an algorithm gap. cuDNN runs implicit-GEMM or
Winograd; F(2,3) alone removes 2.25× of the multiplies. No index-map scheduling change
closes that.

The same logic bounds the rest: for GEMM-shaped kernels the competitor uses **tensor
cores**, and this emitter emits zero `wmma`/`mma.sync` instructions. On fp16 or tf32 GEMM
we cannot win without them — not by tiling, not by staging.

So the honest target is **not** "win everywhere". It is:

- **win where the competitor is structurally weak** — fusion chains it cannot fuse through
  (already 1.35× on the deep epilogue), depthwise (2.08–2.99×), pooling (1.41×), and any
  memory-bound chain
- **reach parity where it is strong and we lack the algorithm** — dense 3×3, dense GEMM at
  large channel counts
- **add the algorithm where the gap is worth it** — tensor cores and Winograd, which are
  new operators, not new schedules

## Ordered plan

1. **Transcendentals and the activation set.** Cheapest, unlocks four PRs, no design change.
2. **`Mean` reduce and a scalar epilogue scale.** Two PRs; small spec addition.
3. **Two-pass reduction for fused statistics**, reusing the split-K sequencing that already
   exists and verifies. Unlocks softmax and LayerNorm — the highest-traffic operators in a
   transformer.
4. **fp16/bf16 in the emitter**, which is also the prerequisite for tensor cores.
5. **Multi-output specs.**
6. **Tensor cores**, aimed only where an algorithmic gap was measured, never as a default.
7. **Dispatch**, once a family carries both evidence columns — and per family, not globally.

Each step should extend `--frontend-check` with the newly expressible form and hold it to
`0.000E+000` before anything is claimed, which is the gate that has caught every real defect
on this branch so far.

## What this audit changes about the earlier blueprint

`PATH_TO_WINS.md` treated the thirteen convolution kernels as the board. They are not the
board — they are one family out of nineteen, and the four losses inside it are mostly
algorithmic rather than schedulable. The larger and cheaper win is **expressiveness**:
seven PRs are blocked on three capabilities that together are far less work than closing a
1.5× algorithmic gap against cuDNN's best convolution path.
