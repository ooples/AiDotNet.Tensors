# FP16 Activation Storage for the Fused Compiled-Training Plan

**Goal:** cut resident GPU activation memory ~50% by storing the dominant activations (matmul/conv/FF outputs) as **FP16** in the compiled training plan, while keeping numerics correct and the fused compile-once-replay path. This is the memory win that batch-size and the (PCIe-bound) host offload cannot give cheaply, and it stacks with the FP16-compute autocast already shipped.

**Why it is NOT a wrap-it-in-a-scope increment (measured):** the shipped `AutocastScope` is FP16-*compute* / FP32-*storage* — at d512/L6/B256 it cut resident memory only 21.8GB→19.4GB (~11%), because every lazy-graph node still allocates `RentUninitialized<float>` (4 bytes/elem). The 50% win requires the *activation buffers* to be FP16 (2 bytes/elem), i.e. a **mixed-precision autodiff graph**.

## Primitives that already exist (the build is wiring, not green-field)
- `IGpuHalfPrecisionBackend` on `CudaBackend`: `Hgemm` (FP16 in/out, CC≥5.0 — RTX 3080 is 8.6), `GemmFp16In32fOut` (FP16 in / FP32 accumulate), `ConvertToFp16` / `ConvertToFp32`.
- `AutocastScope` (`AIDOTNET_AUTOCAST=fp16`): the policy + `_fp16AllowedOps` allowlist (MatMul/Conv/Linear/elementwise/activations FP16; **norms, softmax, loss, reductions stay FP32** — correct AMP policy).
- `MixedPrecision.cs`: `GradScaler` (dynamic loss scaling), `MasterWeights` (FP32 master copy), `MixedPrecisionConfig`.
- `LazyTensorScope.RecordCrossType<TIn,TOut>` + `CrossTypeLazyNode` — the lazy graph **can** hold mixed-type nodes. **GAP: `CrossTypeLazyNode` is forward-only (no `BackwardFn`).** This is the keystone to fill.

## Architecture (PyTorch-AMP shape)
- Activation tensors for autocast-eligible ops are stored `Tensor<Half>`; norms/softmax/loss operate in FP32 with a convert at the boundary. Params + grads + optimizer moments stay FP32 (**master weights** — the optimizer ops aren't autocast-eligible, already true).
- Forward matmul/FF: `Hgemm` (FP16 in → FP16 out) so the stored activation is FP16 (½ bytes).
- Backward through an FP16 activation: read the FP16 save, up-convert to FP32 inside the backward kernel (gradient math in FP32), accumulate into FP32 grad buffers.
- **Loss scaling** (`GradScaler`): scale loss before backward, unscale grads before the optimizer, skip-step + back off scale on inf/nan — prevents FP16 gradient underflow.

## Phases (each gated by the gradient-check harness `LayerNormGradientCheckTests`-style finite-diff + a memory + cortex-end-to-end check; kill-on-NaN)
1. **Keystone — cross-type node with backward.** Add `BackwardFunction<TOut>?` to `CrossTypeLazyNode` and a `RecordCastWithBackward<TStore,TCompute>` on `LazyTensorScope`: stores `Tensor<Half>` output, but registers a backward that up-converts and accumulates FP32 grads. Unit-test forward+backward parity vs the all-FP32 node.
2. **FP16 matmul-store op.** A `MatMulFp16Store` path: under autocast, the engine records the matmul as a cast-with-backward node whose output buffer is `Half` (via `Hgemm`); the consuming op reads FP16 directly. Gradient-check vs FP32 within FP16 tolerance.
3. **Boundary converts.** Norm/softmax/loss inputs auto-convert FP16→FP32; their outputs convert back to FP16 for the next matmul. Reuse `AutocastScope.ShouldAutocast` for the policy.
4. **Loss scaling.** Wire `GradScaler` into `CompiledTrainingPlan` (scale `_lossGradSeed`, unscale parameter grads before the fused optimizer, dynamic backoff). Verify no-underflow on a deep run.
5. **Compile under autocast.** `CompiledTapeTrainingStep` opens the `AutocastScope` around `GetOrCompileTraining` so the lazy graph allocates `Half` activation buffers at trace time (the actual memory reduction), and `StepEager` keeps it open for replay (already done for compute).
6. **Validate + measure.** d512/L6/B256: resident ~½ (target ≤12GB), loss matches FP32 to AMP tolerance, 0 NaN over a full cortex run; the gradient-check CI test extended to mixed precision.

## Risks / non-goals
- FP16 underflow → handled by `GradScaler` (Phase 4); until then expect divergence on deep runs.
- Any host-specialized GEMM closures in a plan need FP16 variants (separate, smaller item, also tracked in #555).
- Regression to the verified FP32 fused path: every phase is behind `AIDOTNET_AUTOCAST=fp16`; default stays FP32-exact.

## Phase 1 finding — the real crux, and why we take the HARD (industry-standard) path
Investigating the keystone pinned the core difficulty precisely:

- AiDotNet's autodiff is **statically single-type**: `GradientTape<T>`, `TapeEntry<T>`, and a `Dictionary<Tensor<T>,Tensor<T>>` grads map — one element type per tape. The compiled plan's backward (built from the lazy graph) is likewise single-type. There is no mixed-dtype gradient flow today (`CrossTypeLazyNode` is forward-only, used for Complex↔real FFT, with no backward).
- **Industry-standard AMP (PyTorch/JAX) is mixed-dtype autograd:** FP16 activations are *first-class `Tensor<Half>` nodes in the graph*; the backward computes FP16 activation grads; gradients are cast at FP16↔FP32 boundaries; params/grads/optimizer stay FP32 (master weights); `GradScaler` prevents FP16 grad underflow. The memory win is a *consequence* of activations genuinely being FP16 in the graph — not a storage trick on logically-FP32 tensors.
- A buffer-compression shortcut (store FP16, keep the graph FP32) would hit the memory number but is **not** real mixed precision and not PyTorch parity. We reject it.

**Therefore Phase 1 is the genuine hard core: extend the autodiff to support mixed-dtype graphs** so a `Tensor<Half>` activation can carry a backward that bridges FP16↔FP32 gradient spaces. Concretely:
- A **cross-type differentiable cast** (`CastToFp16` / `CastToFp32`) recorded on the tape with a backward that converts the incoming gradient to the input dtype and accumulates into the input's grad space.
- The backward walk (`GradientTape.ComputeGradients` and the compiled-plan backward) extended to carry a **secondary `Half` grads map** alongside the `float` one, with the cast nodes as the bridge between them.
- This is the change that makes ALL of FP16/BF16 mixed precision work, so it is gated hard: a finite-difference gradient-check on a 2-op FP16↔FP32 chain MUST match FP32 before anything builds on it (kill-on-mismatch). It is careful autograd surgery — done wrong it corrupts every model's gradients, not just FP16 — so it is implemented and tested in isolation first, behind `AIDOTNET_AUTOCAST`, with the default FP32 path byte-identical.

**Phases (industry-standard):**
- P1: mixed-dtype autograd keystone — the cross-type cast-with-backward + the secondary Half grads space, with a finite-diff gradient-check gate. (In progress.)
- P2: FP16-eligible ops (matmul/conv/FF) emit `Tensor<Half>` activations in the graph under autocast (`Hgemm` output); norm/softmax/loss insert boundary casts. Gradient-check each.
- P3: `GradScaler` loss scaling wired into `CompiledTrainingPlan` (scale loss seed, unscale param grads, dynamic backoff on inf/nan).
- P4: validate d512/L6/B256 resident ≈ ½ (≤ ~12GB), loss-parity to AMP tolerance, 0 NaN end-to-end.

## Test gate (non-negotiable, every phase)
- `LayerNormGradientCheckTests`-style finite-diff vs analytic, extended to the mixed-precision node.
- Resident-memory measurement via `opt-parity LEAKPROBE` (no HE rebuild).
- End-to-end cortex: loss drops, param-L1 moves, **0 NaN**.
