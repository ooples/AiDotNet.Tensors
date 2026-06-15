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

## ⚠️ CRITICAL FINDING (GPU validation on the real cortex) — the net memory win is NOT yet realized

End-to-end GPU measurement through the fused-Adam path on the cortex (d512/L6/B64) shows the mixed path **uses MORE VRAM, not less**: FP32 = 4671 MiB, FP16-activations = 5712 MiB (after the consecutive-op FP16-reuse fix; 5745 before). Training is correct (engages, descends, 0 NaN) — but the memory goal fails.

**Root cause:** the earlier "0.500×" was measured on a *pure matmul stack*, which is unrepresentative. A real transformer separates matmuls with **FP32 ops** (LayerNorm, softmax, GELU, residual add). Each matmul's output is up-cast to FP32 and consumed by the next FP32 op, which **saves that FP32 tensor for its own backward** — so the dominant saved activation stays FP32. Emitting the matmul output as Half merely adds *transient* Half buffers (down-cast inputs + Half matmul output) on top of the still-resident FP32 activation → net footprint rises. The `ToFp16Input` reuse only helps for *directly consecutive* matmuls, which transformers rarely have.

**What the win actually requires (much larger, pervasive):** the FP32 ops between matmuls (norm/softmax/GELU/residual) must become **FP16-storage-aware** — read a Half input, up-cast internally only for their FP32 math, and **save the Half (not FP32) for backward** — so the activation chain stays Half end-to-end. That is an op-by-op change across the whole eligible surface, not a matmul-emission tweak. Until then, `AIDOTNET_FP16_ACTIVATIONS` delivers FP16 *compute* on matmuls (Tensor Cores) but **not** the resident-memory reduction.

**Status correction:** the mixed-dtype autograd engine, compiled plan (A–D), Adam, and routing (E) are built, tested, and train correctly — but the headline **resident-memory win is unachieved** and needs the FP16-storage-aware-FP32-ops work above. The `AIDOTNET_FP16_ACTIVATIONS` flag stays OFF by default; nothing regresses, but it should not be presented as a memory win yet.

## ⚠️ SECOND GPU FINDING — CPU-storage activation paging does NOT free GPU activation memory

Built `SaveFp16` (the FP16 activation-storage recipe) + a full **CPU activation-paging** scheduler in `MixedPrecisionCompiledPlan` (`AIDOTNET_FP16_PAGING`): page each FP32-op activation to Half after its last forward use (`TryDropStorageForStreaming`), upcast on demand in backward (`RestoreStorageFromBytes`), refcounted free. It is **CPU-correct and transparent** (gradients match within FP16; multi-step replay works — `MixedPrecisionPagingTests`).

But GPU measurement on the cortex shows it does **NOT** reduce GPU VRAM (5918 vs 4671 FP32, and ~10× slower). Root cause: on a CUDA backend the dominant activation memory lives in **DirectGpu's GPU-buffer activation cache** (`_activationCache`, evicted by VRAM cap), NOT in the `Tensor`'s CPU `_data` that `DropStorageForStreaming` frees. So CPU-storage paging frees host storage the cortex isn't bottlenecked on, while the GPU buffers stay resident — and `RestoreStorageFromBytes` adds host allocations + CPU↔GPU thrash each step.

**What the GPU memory win actually requires:** FP16 compression at the **GPU-buffer level**, integrated with the DirectGpu activation cache — store evicted/retained activation buffers as FP16 on-device (or Half host-offload) and upcast in the backward kernel. That is a DirectGpu-activation-cache change, distinct from both the autograd-graph dtype and the CPU saved-tensor paging. The CPU paging IS a real memory lever for **CPU** training; the GPU path needs this additional layer.

**Net honest status:** FP16 *compute* on matmuls works; FP16 activation-storage *format* + *CPU paging* work and are tested; the **GPU resident-memory win remains unachieved** and is now precisely localized to the DirectGpu activation-cache layer. All flags OFF by default; nothing regresses.

## Layer 5 design (DirectGpu activation-cache FP16) — primitives, seam, blocker

**Primitives (verified present):** `CudaBackend.AllocateByteBuffer(n)` = exactly `n` bytes (raw `cuMemAlloc`), so a true ½-size FP16 activation = `AllocateByteBuffer(elements*2)`. `ConvertToFp16(fp32,fp16,elements)` / `ConvertToFp32(fp16,fp32,elements)` convert between a float buffer and a half buffer (Tensor-Core path). `IGpuBuffer.SizeInBytes` makes the cache byte-accounting (`_currentActivationCacheBytes`) automatically correct for a half buffer.

**Secondary finding:** the shipped autocast `AutocastScope.MaybeConvertInput` allocates its fp16 buffer via `AllocateBuffer(size)` = `size*4` bytes (FLOAT-sized). So FP16-compute GEMM inputs are stored float-sized — the fp16 buffers save no memory. Fixing this to `AllocateByteBuffer(size*2)` is a contained ½-size win on the transient GEMM inputs (separate, smaller).

**Seam:** cache as FP16 in `CacheActivation` (compress `fp32 → AllocateByteBuffer(elements*2)` via `ConvertToFp16`); upcast on read at the two chokepoints `GetOrAllocateBuffer` / `UploadTensor` (cache hit + `entry.IsFp16` ⇒ allocate transient FP32, `ConvertToFp32`, return owned-transient). `ActivationCacheEntry` gains `IsFp16` + `ElementCount`.

**BLOCKER (why this is a careful redesign, not a patch):** the resident activation's FP32 buffer is owned JOINTLY by the cache AND the producing tensor (`tensor._gpuBuffer`, set on the fast path in `UploadTensor`). To free the FP32 and keep only the FP16, BOTH references must be redirected to the upcast-on-read path. `CacheActivation` only has the backing-array key, not the tensor, so it cannot clear `tensor._gpuBuffer` — freeing the FP32 there would dangle the tensor's pointer (use-after-free / CUDA-700, the #226/#552/#554 race class). The correct fix threads tensor-buffer invalidation through the cache→tensor boundary (e.g. cache owns the activation buffer exclusively and the tensor always re-resolves via `GetOrAllocateBuffer`, never caching `_gpuBuffer` for fp16-cached activations) — a deliberate buffer-lifecycle change validated step-by-step on GPU. This is the genuine remaining work; rushing it regresses the buffer-lifecycle safety the earlier PRs (#552/#554) established.

## Test gate (non-negotiable, every phase)
- `LayerNormGradientCheckTests`-style finite-diff vs analytic, extended to the mixed-precision node.
- Resident-memory measurement via `opt-parity LEAKPROBE` (no HE rebuild).
- End-to-end cortex: loss drops, param-L1 moves, **0 NaN**.

## ⚠️ SIXTH / BOTTOM FINDING — race-safe deferred-free defeats mid-step compression; real fix is a stream-ordered allocator

Layer 5 IMPLEMENTED (on-device `CompressActivationFp16`/`UpcastActivationFp32` swap the cache entry FP32↔FP16, schedule-driven, defensive read upcast; behind `AIDOTNET_FP16_GPU_CACHE`; CPU paths 51/51 green). GPU-measured on the cortex: **7155 MiB vs 4671 FP32 — higher again**, and ~10× slower.

Root cause (bottom of the stack): compression allocates the FP16 buffer but frees the FP32 via the race-safe deferred-free (`CudaBackend.FreeBufferDeferred`, the #226 fix), which holds the FP32 until a stream event — NOT reclaimed until step-end sync. So mid-step the FP16 AND the not-yet-reclaimed FP32 coexist → peak rises. A prompt synchronous free would reclaim it but reintroduces the #226 CUDA-700 race.

**The genuine remaining fix is at the CUDA memory-allocator layer: a stream-ordered allocator** (`cudaMallocAsync` / a stream-aware pool) so a deferred-freed FP32 buffer is reused within stream order by the next allocation WITHOUT a host sync — bounding peak while staying race-free. This is how PyTorch's caching allocator handles AMP activation churn. Every layer above (autograd, compiled plan, Adam, routing, FP16 format, CPU paging, on-device compress/upcast) is built + correct; the resident-memory win is gated on stream-ordered reclamation.

**Definitive status:** FP16 compute works; FP16 storage format + CPU paging + on-device compress/upcast are implemented + tested; the GPU resident-memory win is bottlenecked on the allocator reclamation model. All flags OFF by default; nothing regresses.

## ⚠️ SEVENTH / CONCLUSIVE FINDING — convert-based compression can't reduce peak at any scale; the win needs FP16-NATIVE kernels

Layer 6 (stream-ordered allocator, `cuMemAllocAsync`/`cuMemFreeAsync`, race-safe + pool-reuse, pool release-threshold maxed) IMPLEMENTED + committed (`AIDOTNET_CUDA_ASYNC_ALLOC`). **Note (updated):** this allocator is now **default ON** (opt out with `AIDOTNET_CUDA_ASYNC_ALLOC=0`) — it is required for correctness, not just memory, because the legacy synchronous allocator frees device memory before queued kernels execute (sticky CUDA-700 under buffer churn). On drivers without stream-ordered mem-pool support it falls back to the sync path with a logged warning (or throws if async was explicitly requested). The memory measurements below predate that default flip and were taken with the flag toggled explicitly. Measured the full stack (ACTIVATIONS+PAGING+GPU_CACHE+ASYNC_ALLOC) at BOTH scales on the cortex:
- d512/L6/**B64**: FP32 4671 → FP16-all **7087** MiB (higher)
- d512/L6/**B256**: FP32 8071 → FP16-all **11937** MiB (higher)

CONCLUSION (the absolute bottom): the win does not materialize at any tested scale because **convert-based compression has an irreducible transient peak** — `ConvertToFp16`/`ConvertToFp32` need the source AND destination buffers live simultaneously, so every compress/upcast momentarily holds ~1.5–2× the activation, and across the activation set this transient (plus the caching pool's high-water) exceeds the ½-retention saving. The stream-ordered allocator reuses freed memory but cannot lower the *max concurrent* set, which the convert transients define.

**The genuine GPU activation-memory win requires FP16-NATIVE op kernels:** each consuming op (LayerNorm/softmax/GELU/residual/elementwise + matmul-backward) reads the FP16 activation buffer DIRECTLY and up-casts in-register inside the kernel — never materializing a separate FP32 buffer. That eliminates the convert transient entirely and keeps the activation chain genuinely FP16 end-to-end (PyTorch's model: native fp16 kernels, not buffer compression). This is a pervasive per-kernel change across the GPU op surface — a separate major effort, distinct from everything in layers 1–6.

**FINAL HONEST STATE (7 layers, all GPU-validated):** FP16 compute works; the full mixed-dtype autograd / compiled plan / Adam / routing / FP16 storage format / CPU paging / on-device compress-upcast / stream-ordered allocator are all built + tested; and it is now PROVEN that buffer-level / allocator-level approaches cannot deliver the resident-memory win — it requires FP16-native kernels. All flags OFF by default; nothing regresses. The research farm ran throughout.
