# Direct-PTX optimizer, gradient-scaling, clipping, and sparse-update blueprint

Date: 2026-07-22

Tracking issue: #848

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#848 assembly line and implements one exact
contiguous Ampere FP32 candidate: a **fused SGD-with-momentum update step**:

```
g' = grad + weightDecay * param
v  = momentum * v + g'
param = param - learningRate * v
```

applied elementwise. It is **not promoted**. Each thread loads one FP32x4 vector
of param, grad, and velocity, applies the three fused-multiply-adds entirely in
registers, and commits the updated velocity and param — so the decayed gradient
and the velocity update are **never materialized** to global memory. The naive
path runs two or three separate elementwise kernels, each a full global
round-trip that also writes an intermediate. Correctness is verified on the local
RTX 3080 (SM86) against the reference formula within relative tolerance, with and
without weight decay and across learning-rate/momentum settings.

Production fails closed with `sgd-momentum-performance-gate-not-met`
(`IsPromotedShape => false`). This draft does not close #848. Its 20-cell
manifest assigns the remaining SGD, Nesterov, Adam/AdamW, RMSProp, Adagrad,
Adadelta, LAMB, Lion, gradient-clip, gradient-scale, AMP-unscale, zero,
sparse-update, EMA, and public-routing families.

## Formal contiguous ABI

| Tensor | Extent | Access | Alignment |
|---|---:|---|---|---:|
| param    | `[size]` FP32 vector | read-write | 16 B |
| gradient | `[size]` FP32 vector | read       | 16 B |
| velocity | `[size]` FP32 vector | read-write | 16 B |

The launch has three 64-bit pointer parameters. **Hyperparameters are module
identity, not launch arguments:** the learning rate, momentum, and weight decay
are baked as immediate operands, exactly like the residual-RMSNorm kernel's
epsilon, so the pointer-only ABI carries no scalar parameters. A distinct
`(size, lr, momentum, weightDecay)` tuple yields a distinct cached module; a
constant-hyperparameter training regime caches exactly one module and a
learning-rate schedule caches a bounded, LRU-evicted set. Admitted sizes:
`65536, 262144, 1048576, 4194304`; Ampere only; else exact-reason fallback. The
`weightDecay == 0` path emits one fewer FMA per element.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor performs the same three ops
(`grad.add(param, alpha=wd)` then `v.mul_(momentum).add_(g)` then
`param.add_(v, alpha=-lr)`), where the first op **materializes the decayed
gradient** — the intermediate the fused kernel elides. The C# release-gate
harness feeds the **strongest** competitor (minimum median across the AiDotNet
`sgd_momentum_update` kernel and PyTorch) into `DirectPtxReleaseGate`. A small
learning rate keeps the repeated timing loop numerically bounded so values do not
diverge and distort timing.

The genuine advantage here is **fusion**: the optimizer step is a canonical case
of an elementwise chain that would otherwise materialize per-op intermediates and
pay several global round-trips. Folding the chain into one register-resident pass
is the durable win pattern for the entire optimizer family.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact the
pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with three vector loads, twelve (or eight,
without weight decay) fused-multiply-adds, two vector stores, no `.shared`/
`.local`/`bar.sync`/stride/`.param .u32`, a closed unpromoted size domain,
non-finite-hyperparameter rejection, manifest completeness with exactly one
experimental cell and no promoted cell, and (on a validated Ampere device)
param-and-velocity parity within relative tolerance with zero local and
static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-sgd-momentum 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_sgd_momentum_competitors.py --runs 3
```

## Next optimizer increments

1. Add the bare SGD and Nesterov steps on the same vector shape.
2. Add Adam/AdamW with baked betas, epsilon, and a step-corrected effective
   learning rate (so the step count stays out of the hot loop).
3. Add gradient clipping (global-norm two-stage) and AMP loss-scale unscale with
   an inf/nan finite-check flag.
4. Add FP16/BF16 master-weight variants and independently measured Ada, Hopper,
   and Blackwell modules; never infer promotion from Ampere.
