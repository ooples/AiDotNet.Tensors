# CUDA-Graph Training-Step — PR Scope

## Goal
Eliminate **per-kernel launch overhead** — the proven dominant bottleneck for small-model GPU
training in this engine. A small transformer fires hundreds of micro-kernels per step; each carries
CPU-side launch cost the GPU can't hide (measured: RTX 3080 cortex training sits at ~8–30% util with
the GPU memory-controller ~0%, and the fused fwd+bwd+update path is already engaged yet still
launch-bound). Capturing the *repeated* per-step GPU kernel sequence into a `cudaGraph` and replaying
it with a single `cuGraphLaunch` collapses N launches into 1.

## What's already in place (de-risked)
- **Graph bindings exist**: `cuStreamBeginCapture` / `cuStreamEndCapture` / `cuGraphInstantiateWithFlags`
  / `cuGraphLaunch` / `cuGraphExecDestroy` / `cuGraphDestroy` / `cuStreamIsCapturing` (CudaNativeBindings).
- **Single stream**: the backend issues every op + cuBLAS on one `_stream` → a capturable sequence.
- **Compute ops don't sync**: `Synchronize()` was removed from compute paths; only **downloads**
  (`cuMemcpyDtoH`) are synchronous. → Compute sequences are capturable; **host downloads are the only
  capture-blockers.**
- **Phase-0 probe (this PR)**: `CudaBackend.RunCudaGraphProbe()` (env `AIDOTNET_CUDA_GRAPH_PROBE=1`)
  captures a pure-GPU op sequence and measures graph-replay vs individual-launch latency — proving
  capture works on this stream and quantifying the launch-overhead removal that bounds the whole effort.

## Hard prerequisites (CUDA graph capture forbids host interaction in the captured region)
1. **No host reads/downloads inside the step** — the tape-walk's intermediate reads, the per-step loss
   readback, and the CPU optimizer reading gradients all break capture. Must be removed/relocated.
2. **Stable device buffers across steps** (same pointers) — no per-step `cuMemAlloc`/`cuMemFree`/realloc.
   `InvalidateAllWeightCaches` (dispose+realloc every weight each step) **violates this** and must become
   an **in-place GPU weight update**.
3. **Re-runnable with changed data** — overwrite the *same* input buffer's contents each step
   (`cuMemcpyHtoD` into the stable input buffer) then `cuGraphLaunch`; re-instantiate / `cuGraphExecUpdate`
   only on shape change.

## Phased plan
- **Phase 0 — feasibility + benefit probe** *(this PR)*: bindings confirmed; probe measures replay speedup.
  Decision gate: if replay removes a large fraction of launch cost, proceed; else stop here with evidence.
- **Phase 1 — host-read-free step**: apply the optimizer update **on the GPU** (fused Adam kernel; no
  gradient download); defer the loss-scalar readback to every-N-steps so the step has zero host reads.
  Start from the already-engaged fused-compiled fwd+bwd+update path.
- **Phase 2 — stable buffers**: replace `InvalidateAllWeightCaches` with in-place weight overwrite (no
  realloc); pin activation/gradient buffers (stable pointers) via the buffer pool across steps.
- **Phase 3 — capture/replay**: step 1 → `BeginCapture` → run host-read-free step → `EndCapture` →
  `Instantiate`. Steps 2+ → upload batch into the stable input buffer → `cuGraphLaunch`. Re-capture on
  shape change.
- **Phase 4 — validate**: loss-parity vs eager, util/throughput, multi-config; ship behind a flag, default
  on for CUDA once parity holds.

## Risk / size
**High risk, multi-phase (~several PRs).** Phases 1–2 (GPU-resident optimizer + stable buffers) are
themselves substantial and are prerequisites for any capture. This PR delivers the bindings audit, the
scope, and the **measured feasibility/benefit probe** — the evidence that justifies (or kills) phases 1–4.
