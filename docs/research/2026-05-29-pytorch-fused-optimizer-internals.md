# PyTorch fused / compiled optimizer internals vs AiDotNet.Tensors

Research for issue tracking the compiled-training perf gap (companion to the
AMSGrad plan-dispatch work, PR #501 / #500). Goal: identify what PyTorch does in
its fused / `torch.compile` optimizer path that we don't, and where we can
*exceed* it. Grounded in PyTorch 2.x (`torch/optim/*`, ATen multi-tensor apply,
TorchInductor) as of 2026-05.

## 1. How PyTorch implements optimizer steps

PyTorch ships **three** implementations per optimizer, in increasing speed:

| Tier | Mechanism | Where |
|------|-----------|-------|
| `for-loop` | one Python loop iteration + one set of ATen ops **per parameter** | reference path |
| `foreach` (multi-tensor) | parameters grouped by (device, dtype); the whole group's update runs as **one** `_foreach_*` / `multi_tensor_apply` call | default on CUDA |
| `fused` | a single hand-written CUDA/CPU kernel does the entire update (and, with `capturable=True`, keeps step on-device) | `fused=True` |

Performance ordering: **fused > foreach > for-loop**.

### multi_tensor_apply (the foreach engine)
`multi_tensor_apply` passes a `float**` (array of tensor base pointers) plus
per-tensor sizes into ONE kernel. A device-side functor walks the tensor list,
handles alignment/vectorization, and applies the math. The win is **horizontal
fusion**: N parameter tensors → 1 kernel launch instead of N, amortizing launch
and loop-setup overhead. Bias-correction scalars (`1 - β^t`) are computed **once
per step** and broadcast, not per parameter.

### torch.compile (TorchInductor) optimizer compilation (PT ≥ 2.2)
- Inductor does **vertical fusion**: it can fuse the optimizer's elementwise ops
  (and potentially the tail of backward) into a small number of generated
  kernels, automating what the hand-written fused kernels do.
- Works with all `foreach` optimizers except `LBFGS` / `SparseAdam`.
- **Cost**: optimizers update params **in place** (non-functional). Inductor must
  functionalize, which historically caused multi-minute compile times for
  thousand-parameter models. `capturable=True` + CUDA-graph capture removes the
  per-step CPU launch overhead once compiled.

## 2. What AiDotNet.Tensors already does well

- **Compile-once / replay-many**: `CompiledTrainingPlan` builds the optimizer
  update closure once (`_optimizerUpdate`) and replays it; no per-step graph
  rebuild, no functionalization tax. This is structurally *ahead* of eager
  PyTorch and avoids Inductor's compile-time problem.
- **Inlined LR schedule**: `lrSchedule.GetLr(step)` is an inlined `Math.Cos`/`Pow`
  per step — no `LRScheduler.step()` Python dispatch (`CompiledTrainingPlan.cs`,
  "Issue #348").
- **Live-backed in-place writes**: params are pinned via
  `GetLiveBackingArrayAllowingPaddingOrNull` and updated through `fixed` pointers
  — true in-place, no functionalization copies (Issue #350).
- **AVX2 fused kernels** per optimizer (`FusedOptimizer.*UpdateSimd`) with FMA.
- **Epilogue fusion**: GEMM + bias + activation run as one pass
  (`ActivationEpilogue` / `CpuFusedOperations`), comparable to Inductor vertical
  fusion for the forward.

## 3. What PyTorch does that we DON'T — adopt these

### 3a. Horizontal (multi-tensor) fusion of the optimizer step  ★ highest value
Our `_optimizerUpdate` closure loops **per parameter**, calling one
`*UpdateSimd` per tensor:
```
for (int p = 0; p < paramCount; p++)
    fixed (float* pParam = ..., pM = m[p], ...) AMSGradUpdateSimd(pParam, ...);
```
For a 53-layer net that's 100+ separate kernel calls per step. PyTorch's foreach
collapses these into one. On **CPU** the cost isn't launch latency but per-call
loop setup + the FMA pipeline not staying full across tiny params (biases, norm
scales of length 64–512). A `multi_tensor`-style batched CPU kernel — one call
over a packed list of (param, grad, state) spans — would keep the SIMD pipeline
saturated and cut per-param overhead. **Biggest structural gap.**

### 3b. Hoist step-constant scalars out of the per-param kernel  ★ quick win
`AMSGradUpdateSimd` (and Adam/AdamW/Nadam/RAdam/…) recompute
`bc1 = 1 - MathF.Pow(beta1, step)` and `bc2 = 1 - MathF.Pow(beta2, step)` **inside
every per-parameter call**. Those are **step-constant** — PyTorch computes them
once per step. With 100+ params that's 200+ redundant `Math.Pow` per step. Fix:
compute `bc1`/`bc2` (and `lrAdj`) once in `_optimizerUpdate` and pass them into
the kernels, or add `*UpdateSimdPrecomputed` overloads. Low-risk, measurable.

### 3c. CUDA-graph capture of the whole training step (GPU)
Our GPU path dispatches one backend kernel per parameter (`AdamUpdate`, …).
PyTorch `capturable=True` keeps `step` on-device so the entire forward + backward
+ optimizer step is one CUDA graph replay — near-zero CPU launch overhead.
We already replay a compiled plan; capturing it as a graph would remove the
remaining per-param launch overhead on GPU.

### 3d. Fuse the optimizer update into the backward's gradient write
We run the optimizer as a **separate O(MN) pass** over params/grads/state after
backward. Inductor can fuse the optimizer tail into the kernel that *produces*
the gradient, so param/grad/state are touched while still hot in cache. For the
fused-linear backward this would mean updating the weight as its gradient is
computed — one fewer full pass over weights + grads + m + v(+vMax).

## 4. Where we can EXCEED PyTorch

- **No functionalization / no compile-time cliff.** Our plan is built once from a
  live graph and replayed; we never pay Inductor's minutes-long functionalizing
  compile for large param counts. For "compile then train N steps" we win on
  total wall-clock at small/medium N.
- **AOT shape specialization.** The plan knows every parameter length at
  configure time. We can emit/Select kernels specialized to the exact lengths
  (alignment, full-vector vs remainder split, unroll factor) — PyTorch's generic
  `multi_tensor` functor handles arbitrary runtime sizes and can't specialize as
  tightly.
- **Single-pass CPU epilogue + optimizer.** On CPU we control the whole pipeline;
  we can fuse bias+activation (forward) and (per 3d) the optimizer update into
  fewer memory passes than PyTorch's separate foreach kernels, which is bandwidth
  the CPU actually feels.
- **Zero-GC persistent state.** `TensorArena.CreatePersistent`/`Activate` keeps
  optimizer state + scratch across steps with no per-step allocation; PyTorch
  allocates some per-step temporaries in eager and relies on the caching
  allocator.

## 5. Prioritized action items (for follow-up Tensors issues)

1. **(3b) Hoist `bc1`/`bc2`/`lrAdj` out of the per-param kernels** — quick,
   low-risk, measurable on many-param models. Do first.
2. **(3a) Multi-tensor batched CPU optimizer kernel** — one call over a packed
   span list; the big structural win for many-param nets. Largest effort.
3. **(3d) Fuse optimizer update into fused-linear backward** — removes a full
   memory pass; medium effort, high bandwidth payoff.
4. **(3c) CUDA-graph capture of the compiled step** — GPU launch-overhead removal;
   depends on capturable step state.

## Sources
- [GPU MODE Lecture 6: Optimizing Optimizers in PyTorch](https://christianjmills.com/posts/cuda-mode-notes/lecture-006/)
- [torch.optim — PyTorch main documentation](https://docs.pytorch.org/docs/main/optim.html)
- [Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [RFC: APEX-style fused optimizers in PyTorch (#68041)](https://github.com/pytorch/pytorch/issues/68041)
