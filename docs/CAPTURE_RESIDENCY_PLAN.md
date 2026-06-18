# PR #638 — Whole-step CUDA-graph capture for the cortex: completion plan

**Branch:** `feat/fp16-capture-residency-claude` → base `feat/fp16-in-capture` (PR #633)
**Owner anchor:** this doc is the single source of "what "done" means." We do NOT declare completion until the **Acceptance Test (§1)** passes. No "one more piece" framing — every remaining item is a checkbox below, and we work the list to the bottom.

---

## 0. Why this doc exists (the anti-pattern it fixes)

For several sessions the work was reported as "almost done — one last op" because we narrated the *next* capture-invalidator as if it were the *last*. The trace-driven loop (find op that breaks capture → fix → re-run → find next) makes a visibly-advancing frontier, but the frontier is a long chain across two phases. This doc enumerates the **entire** remaining chain so progress is measured against a fixed checklist, not against "the next blocker."

**Rule:** A phase is "done" only when its explicit done-criteria (checkboxes) are all checked *and* re-verified by a fresh `capture_health.sh` run. "The THREW op moved" is **progress**, not **done**.

---

## 1. Acceptance Test (the definition of "done" for #638)

#638 is COMPLETE when **all** of these hold in one clean run:

1. **Capture engages:** `scripts/capture_health.sh` on the cortex (d128/L1, N≥8000) prints **`VERDICT: CAPTURE FULLY ENGAGED`** — i.e. `eagerFallback=False`, `CUDA-700 none`, **0 THREW**, and `_stepGraphExec != Zero` (the graph instantiates and replays).
2. **Correctness gate:** held-out **CAND-RANK from the captured run == the eager run at the same seed+config** (within ±0.15pp). This compares *captured-vs-eager at identical config* — NOT against the historical 2.81% (which was a different N). A dedicated `HE_CAPTURE_PARITY` harness produces both numbers.
3. **Performance gate (the actual point):** captured **s/step < eager s/step by a meaningful margin** (target ≥1.5× at d128/L1; the real prize is the larger config). Measured from wall/steps, plus `GpuMemoryTracker` showing **no per-op host transfers** in the steady-state captured step (only the scalar loss downloads).
4. **Tests:** full `AiDotNet.Tensors` suite green + a new **resident-backward parity test** (GPU captured grads == CPU grads on a small net).
5. **PR hygiene:** #638 builds in CI, diff is reviewable, ready to consolidate into #633.

If any of 1–5 fails, #638 is **not** done. We do not stop on "frontier advanced."

---

## 1.5 ⚠️ CORRECTED DIAGNOSIS (2026-06-18) — the real blocker is a forward per-step GPU LEAK, not backward residency

**The plan's premise below (§2 "forward fully capture-resident") does NOT hold on the current PR-branch tip.** Measured fresh, the committed branch at d128/L1/N8000 does **not** engage capture and never reaches the backward:

- `capture_health.sh` → `eagerFallback=True`, **CUDA-700 FIRED**, `[CAPTURE-INVALIDATED-BY] forwardAction#0` (embedding) with `EmbFloat-resident FELLBACK … CUDA 901` — a *cascade*, not the root.
- The real error (from `train_loss`): **`cuCtxSynchronize (async-pool OOM reclamation) failed: CUDA-700`**. At d128/L1 (<1 GB expected on a 10 GB card) an OOM = a **per-step GPU memory leak**.
- **NOT a correctness regression:** the cortex `C=2.86%` (≈ ref 2.81%) — the model trains correctly on the **eager** fallback; only capture-engagement is blocked. (`capture_health`'s "CAND drift" warning misparses *hybrid* 1.79% vs the *C* number.)

**Leak localized via `GpuMemoryTracker` peak-dump (`AIDOTNET_GPU_MEMTRACK=1`, N=600 → 278 MB live, 2284 live allocs):** the dominant retention-by-COUNT is **`MultiHeadAttentionForward` (`DirectGpuTensorEngine.MissingKernels.cs:2481`, via `MultiHeadAttentionLayer.TryFusedAttentionInference`)** — **x540** `TensorMatMul`, **x536** `TensorMatMulTransposed`, **x536** `TensorMultiplyScalar`, **x536** `SoftmaxLastAxisGpu`. It builds a long chain of fresh intermediate tensors per step (Q/K/V projections + per-head `scores`/`attn`/`headOuts` + concat/permute/out-proj); each `TensorMatMul`/`Softmax` records backward so they're tape-live within the step, but they are **not released after the step** → accumulate ~0.46 MB/step at d128/L1 → cross 10 GB by ~N8000 → async-pool OOM → sticky CUDA-700 → capture aborts → eager. This is the SAME leak the sibling branch `gpu-stream-event-safety` WIP `3ff3aea` ("fused-mode ~500MB/step accumulation not yet cracked") was chasing.

**Revised order of work:** (1) **fix the MHA-forward / per-step intermediate leak** so capture can engage at all (the true §1 unblocker); THEN (2) the A1/A2 backward-residency work below (now testable). A2's in-place path is disabled (async-700 hazard, unrelated — see §3); my A0 instrumentation (download trace + backward capture probe) is committed and reusable.

---

## 2. What is already DONE (committed + verified, do not redo)

> ⚠️ **STALE — see §1.5.** This section reflects an EARLIER branch state. On the current tip the forward does **not** stay capture-resident (the MHA-forward leak OOMs and aborts capture in the forward). Re-verify before trusting any item here.

The **entire forward + loss is capture-resident** (commits `88ba2db`…`ef7e3b5`). Verified: `eagerFallback=False`, `CUDA-700 none`, test passes, THREW moved from forwardAction#0 all the way to the backward.

- [x] Embedding-table DtoH-download fix (`ResolveResidentEmbeddingTable` → reuse pinned `_cachedEmbTableBuffer`)
- [x] `BatchedGemmSequential` — capture-safe attention BMM (strided-batched is capture-unsafe)
- [x] Resident `TrySoftmaxResidentInto` (last-axis), `TryReLUResidentInto`, `TryLogResidentInto`, `TryNegateResidentInto`, `TryBroadcastMultiplyResidentInto`
- [x] `TryFusedLinearResidentInto` (FFN: GEMM-into + in-place BiasAdd + in-place GELU/ReLU) — flipped `eagerFallback`→False and cleared the intermittent CUDA-700
- [x] `TryAddInPlaceResident` (engages once operands are resident — currently declines in the backward)
- [x] Sub-op capture-status probes (`CAP-PROBE`/`EMB-PROBE`/`AddInPlace SKIP`, gated on debug flags)

**Proven boundary:** every remaining capture-invalidator is a **backward gradient op** whose operands are host-side (`bResident=False` across all grad shapes).

---

## 3. PHASE A — Backward gradient residency (the bulk)

**Goal:** the whole backward (the captured body's second half) runs GPU-pure — no `GetDataArray`/`DownloadBuffer` on any gradient tensor.

**What's true now (from code reading):**
- Param-grad buffers (`_preAllocatedGrads`, the `gradMap` leaves) ARE GPU-resident (zeroed via `MemsetBuffer` on GPU). ✅
- The backward runs the **generic** path (`preferGenericForGpu=True` on the DirectGpu engine) → `stepCopy.BackwardFn(gradOut, inputs, …)` dispatched eagerly (`CompiledTrainingPlan.cs:2715`).
- `BackwardFunctions.cs` has ~386 `engine.Tensor*` calls that create **fresh intermediate** grad tensors (e.g. `gradA = engine.TensorMultiply(gradOut, b)`), then `AccumulateGrad` (→ `TensorAddInPlace`) into the resident leaves. Those **intermediates are not resident** → the accumulation downloads → CUDA 900.

**Approach (same proven loop, applied to backward, in this order):**

- [x] **A0 — Instrument the backward once.** DONE (2026-06-18). (1) capture-path probe extended past the forward loop into the backward (`RunGpuStepBodyForCapture`) — logs `[CAPTURE-INVALIDATED-BY] backwardAction#i name=… op=…` via a new `s_currentBackwardOp` tag on the generic backward action. (2) frequency-ranked **DtoH download trace** (`GpuMemoryTracker.BeginDownloadTrace`/`OnDownload`/`DumpDownloadTrace`, env `AIDOTNET_GPU_DOWNLOAD_TRACE=1`, hooked in `CudaBackend.DownloadBuffer`, armed around the backward in `StepEager`), run EAGER so all copies are seen in one pass.

  **A0 result (eager d128/L1): 107 DtoH copies / 426 MB across 20 sites. Ranked work-list:**
  - **~70 copies (13 sites): `AccumulateGrad → TensorAddInPlace → TryRunBinaryInPlace`** (operand `GetDataArray` + unconditional result download) → **A2** (the dominant lever).
  - 26 copies / 104 MB: `GetDataArray → DeferredArrayMaterializer.TryMaterialize` (backward ops pulling forward activations to host) → A1.
  - 6: `LayerNormBackward`; 2: `SoftmaxBackward`; 1 ea: `ReluBackward`, `ReduceGradToShape→ReduceSum` (engine methods download internally) → A1.

  Trace at `HarmonicEngine/logs/dl_trace.txt`. NB: the eager trace is a valid *work-list* (same backward ops run) but `ResidentStepActive` is false in eager mode, so fixes are verified on the CAPTURE path (`capture_health.sh`, GRAPH_STEP=1).
- [ ] **A1 — Resident intermediates for the autodiff ops.** For each `engine.Tensor*` in the hot backward ops, ensure the DirectGpu eager op **keeps its result resident** during `ResidentStepActive` (bind `_gpuBuffer`, no download). Candidates from the eligibility/trace list: `TensorMultiply`, `TensorMatMul` (incl. transposed + the rank-4 batched backward), `Softmax` backward, `TensorBroadcastAdd`/reduce, `ReduceGradToShape`, `Negate`, `Log` backward, `FusedLinearWithActivationBackward`, `EmbeddingBackward`. Pattern per op: replace `AllocateOutputBuffer` + `DownloadBuffer` + `new Tensor(host)` with `GetOrCreateResidentBuffer` + `BindResidentBuffer`; gate on resident inputs (`TryGetGpuBuffer()`) to avoid the A2 owned-upload async-free trap.

  ⚠️ **A1 KEY CHALLENGE (why this is the multi-session phase): stable buffers for DYNAMIC intermediates.** The generic backward calls `engine.ReluBackward(...)` etc. which return a **freshly-allocated `Tensor` every step**. A naïve "bind resident" would `cuMemAlloc` a NEW device buffer each captured step → a graph-purity violation (the capture pass needs the SAME device pointers the pre-pass instantiated). So A1 isn't just "don't download" — each backward op needs a **stable per-op scratch buffer reused across steps** (like the forward's `_lnStatsScratch` keyed by the output backing array). Likely approaches: (a) key a scratch pool by (op-identity, gradOut stable array) so the same buffer returns each step; or (b) pre-allocate per-backward-step scratch in `CompiledTrainingPlan` (alongside `_preAllocatedGrads`) and have the generic backward write into it. (b) is the more invasive but more robust fix. The dynamic-allocation problem — not the download per se — is the real Phase-A work.
- [~] **A2 — Resident grad accumulation.** IMPLEMENTED + hardened (2026-06-18). GPU-resident fast path at the top of `TryRunBinaryInPlace`: when `ResidentStepActive` and BOTH operands are already resident on their own stable buffers, run the elementwise op (`Add`/`Multiply`/`Subtract`) in place on `a`'s device buffer and rebind — no `GetDataArray`, no result download. *Note: `TryAddInPlaceResident` did not previously exist (the §2 checkbox was aspirational).*

  ⚠️ **A2 DISABLED BY DEFAULT — in-place resident mutation is an async-CUDA-700 hazard (3 fix attempts failed).** The in-place fast path triggers an async CUDA-700 whenever it engages on the capture/pre-residency path; per-op sync (`AIDOTNET_RESIDENT_SYNC_DEBUG=1`) makes it vanish = textbook async hazard. Tried, all still 700: (1) require `b` resident (kills the owned-upload async-free); (2) `aResident≠bResident` + size guards; (3) gate to grad-accumulation only (`InGradAccumulation`, set by `AccumulateGrad`) so it never hijacks forward in-place ops on aliased activations. Clean A/B every time: `AIDOTNET_RESIDENT_INPLACE=0` → 0×700, `eagerFallback=False`; ON (any variant) → 700 + forward#0 cascade abort.

  **Root:** in-place mutation of a resident buffer races the buffer lifecycle on the eager pre-pass/warmup (the captured graph replays fine, but the pre-residency eager pass that sets it up does not). **The safe replacement is OUT-OF-PLACE accumulation:** `accumulated = TensorAddInto(existing, grad)` into a STABLE per-leaf destination buffer (the proven `TryRunBinaryInto` resident path, which never mutates an input), then store `accumulated`. That needs a stable per-grad-leaf accumulator buffer (same dynamic-buffer infra as A1). Code kept behind `AIDOTNET_RESIDENT_INPLACE=1` for that follow-up. **A2 is therefore blocked on the A1 stable-buffer infrastructure** — do A1 first.
- [ ] **A3 — Loss-seed + reductions.** The `lossGradSeed` fill and the `ReduceSum`→scalar-loss path: confirm GPU-resident; the **only** legal download is the final scalar loss (outside the captured region or via a device-scalar).
- [ ] **A4 — cuBLAS-in-capture workspace.** Any backward `cublas*` call must not allocate workspace / sync during capture (the strided→sequential fix handled the forward BMM; the backward GEMMs may need pre-allocated workspace, PyTorch-style). Watch for `INTERNAL_ERROR`/`Execution failed` reappearing.

**Phase A done-criteria:**
- [ ] `capture_health.sh` shows **0 THREW** (capture no longer aborts) at d128/L1.
- [ ] No `*-resident FELLBACK` / `AddInPlace SKIP` lines in the backward portion of the trace.
- [ ] Still `eagerFallback=False`, `CUDA-700 none`.

*Realistic size: this is the multi-session phase. ~10–20 backward ops + the accumulation path. Each is mechanical but there are many, and intermediates are created dynamically so A0's map is essential.*

---

## 4. PHASE B — Capture actually engages (instantiate + replay)

0 THREW means capture isn't *aborted*, but we must confirm it *engages and replays*.

- [ ] `_stepGraphExec != Zero` after the capture pass (graph instantiated). Add an explicit log.
- [ ] The graph **replays** across steps (`LaunchCapturedGraph`) instead of re-capturing every step (constant batch shape — drop/pad the trailing partial batch so the shape is stable; per memory, variable shape → recompile → never engages).
- [ ] No silent fallback to `StepEager()` mid-run (the `_graphStepDisabled` paths).

**Phase B done-criteria:** capture instantiates once, replays for the rest of the epoch, no per-step re-capture.

---

## 5. PHASE C — Correctness gate (non-negotiable)

- [ ] Build `HE_CAPTURE_PARITY`: run the SAME seed+config twice — once eager (`AIDOTNET_CUDA_GRAPH_STEP=0`), once captured (`=1`) — and diff held-out CAND-RANK + final train loss.
- [ ] **Captured CAND-RANK == eager CAND-RANK (±0.15pp).** If they diverge, the captured graph is reading stale/wrong buffers (e.g. a grad baked to the wrong pointer) — fix before any perf claim.
- [ ] New unit test in Tensors: resident captured backward grads == CPU backward grads on a tiny net (catches A1/A2 numeric bugs).

---

## 6. PHASE D — Performance gate (the actual reason for all of this)

- [ ] s/step (captured) vs s/step (eager), computed from wall/steps — captured must be **meaningfully faster** (≥1.5× at d128/L1; measure the bigger config too).
- [ ] `GpuMemoryTracker` (live-delta): **zero per-op host transfers** in the steady-state captured step.
- [ ] Power draw sanity (a compute-bound captured step pulls more watts than the launch-bound eager step).
- [ ] Record the numbers in this doc. If captured is NOT faster, capture is not worth shipping default-on — document why and stop (don't ship a slower default).

---

## 7. PHASE E — Tests + PR hygiene (close out #638)

- [ ] Full `AiDotNet.Tensors` suite green.
- [ ] Remove/keep-gated the debug probes (they're gated; decide keep vs strip).
- [ ] #638 CI green; squash the trace-driven micro-commits into a readable set if needed.
- [ ] Then (separate, post-#638): consolidate into #633, resolve #633's merge conflicts with main, add codecov coverage. **Out of scope for #638 itself** but noted so it isn't forgotten.

---

## 8. Risks / non-negotiables

- **Correctness over speed:** a captured graph that's fast but reads a stale grad buffer silently corrupts training (the exact "meaningful numbers" failure mode). Phase C gates every perf claim.
- **Don't claim "done" on a moved frontier.** Only §1's Acceptance Test = done.
- **One GPU run at a time;** kill stray `testhost.exe` before timing.
- **CPU farm:** keep ≥2 backlog experiments running throughout (this is GPU work; the farm is independent and must not idle).
- **The intermittent CUDA-700** may resurface when the backward goes resident (it's a resident-step illegal access). If it does, it gates progress — localize via `AIDOTNET_RESIDENT_SYNC_DEBUG` (per-op sync) before proceeding.

---

## 9. Execution order (after approval)

1. Commit this doc to the branch (the anchor).
2. **Clear context** (fresh session starts from this checklist).
3. Phase A0 (instrument) → A1/A2 (work the backward-op list to the bottom) → A3/A4.
4. Phase B → C → D → E.
5. Update the checkboxes in this file as each item lands; re-run `capture_health.sh` to verify, not to "see the frontier move."
