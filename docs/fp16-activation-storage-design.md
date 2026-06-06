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
- **P1 — DONE (verified, eager-tape).** The mixed-dtype autograd keystone is implemented and gated:
  - `MixedPrecisionCast` — the cross-type FP32↔FP16 differentiable cast (forward + backward bridge; cast Jacobian = identity). Gate: `MixedPrecisionCastTests` 3/3 (forward round-trip, gradient-bridge round-trip, exact on FP16-representable values).
  - `GradientTape.ComputeGradients` gains an **additive** `seedOverride` (default null ⇒ byte-identical; supplied ⇒ seed interior tensors + force the slow tape walk). This is the only new mechanism the bridge needs from the engine.
  - `MixedPrecisionTape` — pairs an FP32 and FP16 inner tape joined at cast boundaries, driving the backward as a **Gauss-Seidel sweep** over the cast boundaries (exact reverse-mode in ≤(#boundaries+1) rounds, no double-counting). Gate: `MixedPrecisionTapeGradientCheckTests` — a 2-segment **interleaved** chain (the case a single pass cannot do; dL/dx is only reachable after the 2nd segment's FP16 grad bridges back and re-seeds the FP32 tape) matches the analytic gradient exactly.
  - Regression: default FP32 path proven byte-identical (31/31 autograd correctness green).
- **P3 — DONE (verified).** `GradScaler` wired into `MixedPrecisionTape.ComputeGradients(loss, scaler)`: scale the backward seed, up-cast every grad to FP32 (master space) then unscale (cannot re-underflow), `FoundInfNan` for dynamic backoff. Gate: `MixedPrecisionLossScalingTests` 2/2 — shows the underflow FAILURE (unscaled ⇒ 0) and the RECOVERY (scaled ⇒ correct 2e-8, no spurious overflow).
- **P2 — backward machinery DONE (verified, CPU); forward-emission + production wiring REMAINING (GPU).**
  - **DONE (a) — `CrossTypeLazyNode` backward.** The node was forward-only (FFT/Complex); it now optionally carries a `CrossTypeBackwardFunction<TIn,TOut>` + `Backward()`, with `LazyTensorScope.RecordCrossTypeWithBackward` to record one. Additive (forward-only usage unchanged). Gate: `CrossTypeLazyNodeBackwardTests` 2/2 (cast node bridges FP16↔FP32 exactly; a non-identity cross-type Jacobian matches finite-difference — proves it isn't hard-wired to identity). 735/735 compilation/lazy/FFT/Complex subset green.
  - **DONE (b) — mixed-dtype graph backward.** `MixedPrecisionGraphBackward.Backward(loss)`: because the lazy graph is a single **unified DAG** (cross-type nodes link the dtypes via `GetInputNodes`), ONE reverse-topological sweep is exact — no Gauss-Seidel needed (that was only an artifact of the two separate eager tapes). Each `LazyNode<T>` runs its `BackwardFunction<T>` against the matching-dtype grad map; cross-type nodes bridge between the FP32/FP16 maps. Gate: `MixedPrecisionGraphBackwardTests` — the interleaved chain as a lazy graph, dL/dx + both FP16 param grads match analytic exactly. Additive — does not touch the hot `CompiledTrainingPlan`.
  - **DONE (c) — forward Half-buffer emission + engine seam.** `MixedPrecisionEmit.MatMul` records the down-cast→FP16-matmul→up-cast triple (matmul activation = `Tensor<Half>`); `CpuEngine.TensorMatMul` under GraphMode auto-emits it when `MixedPrecisionEmit.ActivationStorageActive<T>()` (FP16 `AutocastScope` + `AIDOTNET_FP16_ACTIVATIONS`). DirectGpu delegates here under GraphMode, so the same seam covers CPU + GPU Half execution at realize. Gates: `MixedPrecisionEmitTests` (activation is `Tensor<Half>` + auto-emitted graph backprops to correct grads) and `MixedPrecisionTrainingE2ETests` (full training loop descends loss >2× with Half activations). Default off ⇒ 650/650 compilation/matmul/mixed-precision subset byte-identical.
- **P4 — buffer-win MEASURED (GPU).** `Fp16ActivationMemoryTests` on `DirectGpuTensorEngine`: an 8-layer matmul stack stores activations at **exactly 0.500×** the FP32 bytes (4.00 MB → 2.00 MB), every layer `Tensor<Half>`. Activation-storage halving proven on real GPU. (At-scale resident measurement at d512/L6/B256 lands with the fused-plan rewrite Phase D below.)

## Remaining: route the FUSED compiled plan to mixed-dtype (the multi-week heterogeneous rewrite)

The mixed-dtype mechanism AND the memory win are done via the lazy-graph path. What remains is giving the *fused* `CompiledTrainingPlan` (compile-once-replay performance) mixed-dtype support. It is single-type — `CompileTraining` (~line 2005) only emits `CompiledStep<float>` and **skips** `CrossTypeLazyNode`/`LazyNode<Half>`, and its fused groups / IL backward walker / CUDA-graph capture all assume one element type. This is a genuine subsystem rewrite, phased:

- **Phase A — DONE.** `MixedPrecisionCompiledPlan` — heterogeneous compile-once/replay forward (topo-sorted, lazy-sources detached, per-node `Execute` into stable buffers; float/Half/cross-type dispatched by type). Gate: replay matches a fresh trace, initially + after in-place param mutation.
- **Phase B — DONE.** `MixedPrecisionCompiledPlan.Backward()` over the captured order via the shared `MixedPrecisionGraphBackward.BackwardOverOrder` (single dispatch; works after lazy-source detachment). Gate: matches the non-compiled backward within FP16 GEMM tolerance.
- **Phase C — DONE.** `MixedPrecisionCompiledPlan.Step(params, lr, scaler)` — forward + loss-scaled backward + SGD on FP32 masters + `GradScaler` (skip-on-overflow + backoff). Gate: compiled training descends loss >2×, 0 overflow.
- **Phase D — DONE (correctness at scale).** Compiled plan trains the design-target stack (d512/L6/B256) on `DirectGpuTensorEngine`: loss finite + non-increasing, 0 NaN, VRAM 1229 MiB. The 0.500× activation-storage ratio is proven deterministically (`Fp16ActivationMemoryTests`). *Remaining refinement: CUDA-graph capture of the mixed-dtype step (perf), and a clean at-scale FP32-vs-FP16 in-process VRAM A/B.*
- **Phase E — REMAINING (cross-repo deployment).** Make the plan a public Tensors API and route AiDotNet `NeuralNetworkBase.TrainWithTape` / `CompiledTapeTrainingStep` to it under the flag, so HarmonicEngine/opt-parity exercise it through the high-level training path on GPU.

Phases A–D landed (CPU + GPU verified) behind `AIDOTNET_FP16_ACTIVATIONS`; default FP32 fused path byte-identical. **Phase E** (public API + AiDotNet wiring + GPU validation through the high-level path) is the remaining cross-repo step.

## Test gate (non-negotiable, every phase)
- `LayerNormGradientCheckTests`-style finite-diff vs analytic, extended to the mixed-precision node.
- Resident-memory measurement via `opt-parity LEAKPROBE` (no HE rebuild).
- End-to-end cortex: loss drops, param-L1 moves, **0 NaN**.
