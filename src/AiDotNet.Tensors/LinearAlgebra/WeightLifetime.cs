// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Lifetime / placement hint a layer attaches to a weight tensor at construction.
/// The allocator and the GPU backend dispatch differently based on this hint;
/// activations stay <see cref="Default"/> (transient, fast pool).
///
/// <para>Issue #276 sub-features (1) BFloat16 storage, (2) streaming pool,
/// (3) quantization, and (4) GPU offload all read this flag — the engine
/// surfaces it via <see cref="Tensor{T}.Lifetime"/> so a model author writes
/// the hint once at weight construction and every backend (CPU /
/// CUDA / HIP / Metal / OpenCL / Vulkan / WebGPU) honours it on read.</para>
/// </summary>
public enum WeightLifetime
{
    /// <summary>Activation / transient buffer. Lives in the fast pool, GC-bound. Default.</summary>
    Default = 0,

    /// <summary>Weight that is too large to keep resident — the streaming
    /// pool may evict + rehydrate from a backing store between uses.</summary>
    Streaming = 1,

    /// <summary>Weight that should be allocated from pinned-host memory and
    /// DMA-mapped to the GPU on demand. No explicit cudaMemcpy per op.</summary>
    GpuOffload = 2,

    /// <summary>Weight that should be allocated from unified / managed memory
    /// (cudaMallocManaged / hipMallocManaged / Metal shared MTLBuffer / OpenCL
    /// SVM). Driver handles page migration; no explicit copies.</summary>
    GpuManaged = 3,

    /// <summary>
    /// Weight tagged for GPU-resident lifetime — same storage strategy as
    /// <see cref="GpuOffload"/> (pinned-host + DMA mapping) but with the
    /// semantic intent of "this lives on the GPU side of the train loop".
    /// Used by issue #336's optimizer-state-on-GPU work: Adam <c>m</c> /
    /// <c>v</c> buffers, BatchNorm running stats, and weights all tagged
    /// GpuPinned avoid the per-train-step <c>cuMemcpyHtoD</c> /
    /// <c>cuMemcpyDtoH</c> round-trip that dominates small-batch wall-time.
    /// <para>
    /// On CPU-only hosts, falls back to <see cref="Default"/> with a
    /// CPU-pinned arena tier (<c>TensorAllocator.RentPinnedOnGpu</c> handles
    /// the fallback transparently). The lifetime value is an intent hint;
    /// the actual allocator is dispatched by the active engine.
    /// </para>
    /// </summary>
    GpuPinned = 4,
}

/// <summary>
/// Precision the streaming pool stores weight bytes at on its backing store.
/// bf16 halves disk + resident I/O at ~0.17% RMS error; because the pool's
/// stored copy IS the canonical weight, bf16 during TRAINING effectively makes
/// the master weights bf16 (which mixed-precision training avoids — it keeps
/// fp32 masters), so the safe default is context-aware.
/// </summary>
public enum StreamingStoreDtype
{
    /// <summary>Context-aware default — compresses whenever the execution mode is known:
    /// in inference/eval (read-only weights → a one-time quantization, always safe) it is
    /// <b>RAM-aware</b> when <see cref="GpuOffloadOptions.ExpectedStreamingFootprintBytes"/>
    /// is set: it stores the HIGHEST-fidelity precision whose resident footprint still fits
    /// <see cref="GpuOffloadOptions.StreamingPoolMaxResidentBytes"/> — bf16 (2x) if the model
    /// fits at bf16, else int8 (4x), else int4 (8x) so an otherwise-too-large model runs on a
    /// constrained box; with no footprint hint it defaults to bf16. During TRAINING it stays
    /// LOSSLESS (byte-shuffle + Deflate, ~1.18x, BIT-EXACT) so the fp32/fp64 masters are
    /// preserved exactly (the quantized tiers can't write mutations back — write-back is
    /// native-only — so they are never chosen while weights are mutable). Unknown mode (no
    /// declared context) → full precision (don't guess).</summary>
    Auto = 0,

    /// <summary>Always store at the weight's native precision (fp32/fp64). No
    /// quantization, no I/O reduction — the pre-bf16 behaviour.</summary>
    FullPrecision = 1,

    /// <summary>Always store bf16 with round-to-nearest-even. 2x I/O. Safe for
    /// inference; for training it's bf16 masters (deterministic-rounding bias on
    /// small updates) — prefer <see cref="Bf16Stochastic"/> when training.</summary>
    Bf16 = 2,

    /// <summary>Always store bf16 with STOCHASTIC rounding. 2x I/O, and the
    /// rounding is unbiased so training stays correct in regimes that tolerate
    /// bf16 masters (large-batch pretraining). Opt-in for training.</summary>
    Bf16Stochastic = 3,

    /// <summary>Always store int8 with a per-row symmetric scale. 4x I/O at ~1.1% RMS
    /// error — more lossy than bf16, so <see cref="Auto"/> only steps down to it
    /// automatically in INFERENCE when bf16 would overflow the resident cap (see
    /// <see cref="GpuOffloadOptions.ExpectedStreamingFootprintBytes"/>); this explicit
    /// value forces it unconditionally for aggressive memory-bound inference.</summary>
    Int8 = 4,

    /// <summary>EXACT (lossless) storage: SIMD byte-plane shuffle + Deflate. ~1.18x I/O on
    /// dense fp weights (raw codecs yield ~0% — the shuffle exposes the structured
    /// sign/exponent byte-plane and Deflate's entropy stage shrinks it) at ZERO precision
    /// loss. This is the Auto default during TRAINING (bit-exact masters); bf16/int8 give far
    /// more (2x/4x) at a precision cost. ~1.1 GiB/s decode, overlapped by prefetch.</summary>
    Lossless = 5,

    /// <summary>Always store int4 with AWQ/GPTQ-style GROUP-symmetric quantization (one fp32
    /// scale per 128-weight group). 8x I/O — the most aggressive rung, intended to make the
    /// very largest (&gt;~20B) models RESIDENT inside a 16 GiB budget so they skip streaming.
    /// More lossy than int8, so <see cref="Auto"/> only steps down to it automatically in
    /// INFERENCE as the last rung when even int8 would overflow the resident cap; this explicit
    /// value forces it unconditionally for aggressive memory-bound inference.</summary>
    Int4 = 6,
}

/// <summary>
/// Per-engine options for the GPU offload / streaming subsystem. Plumbs
/// through every direct-GPU backend (CUDA / HIP / Metal / OpenCL / Vulkan /
/// WebGPU) so a single config struct controls the placement contract.
/// </summary>
public sealed class GpuOffloadOptions
{
    /// <summary>Preferred placement scheme for <see cref="WeightLifetime.GpuOffload"/>
    /// weights. <see cref="OffloadScheme.Auto"/> probes the device for
    /// pageable-memory-access support and picks the best.</summary>
    public OffloadScheme PreferredScheme { get; set; } = OffloadScheme.Auto;

    /// <summary>Maximum resident bytes for streaming-pool weights. When
    /// the pool exceeds this, oldest unused entries evict. Defaults to
    /// <see cref="DefaultResidentBudgetBytes"/> — min(16 GiB, ~70% of the
    /// memory available to the process) — so the pool's own resident byte[]s
    /// never push the box into OS-level swap (which would page on top of the
    /// pool's own paging — double I/O). Set explicitly to override.</summary>
    public long StreamingPoolMaxResidentBytes { get; set; } = DefaultResidentBudgetBytes();

    /// <summary>
    /// The caller's estimate of the model's TOTAL streaming-weight footprint, in the
    /// weights' native precision (fp32/fp64 bytes = ParameterCount × element size).
    /// Zero (the default) means "unknown". When known, it makes
    /// <see cref="StreamingStoreDtype.Auto"/> <b>RAM-aware in inference</b>: instead of
    /// always storing bf16, Auto picks the HIGHEST-fidelity store precision whose
    /// resident footprint still fits <see cref="StreamingPoolMaxResidentBytes"/> —
    /// bf16 (÷2) when it fits, else int8 (÷4), else int4 (÷8). So a model that fits
    /// resident at bf16 keeps bf16's fidelity, and only a model too large for bf16
    /// steps down to the lossier-but-smaller quantized stores needed to run at all on
    /// a constrained box. Training is unaffected — Auto stays lossless there (exact
    /// masters), because the quantized stores' mutations can't be written back
    /// (write-back is native-only); inference weights are read-only, so the quantized
    /// tiers are safe there. Ignored unless <see cref="StreamingStoreDtype"/> is
    /// <see cref="StreamingStoreDtype.Auto"/>.
    /// </summary>
    public long ExpectedStreamingFootprintBytes { get; set; }

    // Historical ceiling, kept as the cap for large-memory boxes so their
    // behaviour is unchanged.
    private const long ResidentBudgetCeiling = 16L * 1024 * 1024 * 1024;

    /// <summary>
    /// The default resident budget: min(16 GiB, ~70% of memory available to the
    /// process). The 70% headroom leaves room for activations, the GC heap, and
    /// the OS; on boxes / containers with ≥ ~23 GiB available this returns the
    /// historical 16 GiB (no behaviour change), and on smaller boxes it clamps
    /// down so the resident set stays within RAM instead of triggering OS swap.
    /// Never below 512 MiB (a tiny budget thrashes eviction). Falls back to the
    /// 16 GiB ceiling when available memory can't be determined (e.g. net471).
    /// </summary>
    public static long DefaultResidentBudgetBytes()
    {
        try
        {
#if NET5_0_OR_GREATER
            long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
#else
            long available = 0; // net471: no portable query — keep the ceiling.
#endif
            if (available <= 0) return ResidentBudgetCeiling;
            long ramAware = (long)(available * 0.7);
            long budget = Math.Min(ResidentBudgetCeiling, ramAware);
            budget = Math.Max(512L * 1024 * 1024, budget);
            // Cap by what's actually available — the 512 MiB floor used to
            // be unconditional, so on a container with e.g. 256 MiB free
            // we'd return 512 MiB and immediately push the pool over the
            // OS limit, triggering exactly the swap-thrash the budget is
            // meant to prevent.
            return Math.Min(budget, available);
        }
        catch
        {
            return ResidentBudgetCeiling;
        }
    }

    /// <summary>Backing store path for the streaming pool. Null ⇒ use the
    /// system temp dir. Memory-mapped file is the default backing format.</summary>
    public string? StreamingBackingStorePath { get; set; }

    /// <summary>
    /// LZ4-compress weight bytes before writing them to the backing store.
    /// Default false — and that is the right default: raw LZ4 does NOT shrink
    /// dense floating-point weights. Measured on near-Gaussian fp32/fp64 it
    /// lands at ~100% (the high-entropy mantissa is incompressible to a
    /// match-only codec; the eviction path's raw-fallback fires), so enabling it
    /// adds encode CPU for ~0 benefit on typical weights. It only helps weights
    /// with real byte-level structure (heavy sparsity / repeats).
    /// <para>
    /// What actually shrinks weight bytes (see StreamingWeightCompressionRatioTests):
    /// lossless bit-plane shuffle + an ENTROPY coder (zstd/Brotli) reaches only
    /// ~1.15–1.25x because the mantissa is near-random; the real lever is lossy
    /// quantization of the backing store — bf16 = 2x at ~0.17% RMS error,
    /// int8 per-tensor = 4x at ~1.1% — which is future work gated on
    /// lossy-during-training safety.
    /// </para>
    /// </summary>
    public bool EnableCompression { get; set; } = false;

    /// <summary>
    /// Precision the streaming pool stores weight bytes at. Default
    /// <see cref="StreamingStoreDtype.Auto"/> — <b>compresses by default</b>, always picking
    /// the max-safe option for the context: bf16 (2x I/O, 4x for fp64, at ~0.17% RMS error)
    /// in inference/eval where it's a one-time, always-safe quantization; and <b>lossless</b>
    /// (byte-shuffle + Deflate, ~1.18x, BIT-EXACT) during training — so fp32/fp64 masters are
    /// preserved exactly (no convergence risk) while still reclaiming disk + resident bytes.
    /// Set <see cref="StreamingStoreDtype.Bf16Stochastic"/> for 2x during training in regimes
    /// that tolerate bf16 masters, <see cref="StreamingStoreDtype.Int8"/> for 4x lossy
    /// inference, or <see cref="StreamingStoreDtype.FullPrecision"/> to disable compression.
    /// </summary>
    public StreamingStoreDtype StreamingStoreDtype { get; set; } = StreamingStoreDtype.Auto;

    /// <summary>
    /// Opt-in (default <see langword="false"/>): when materializing a paged-out weight
    /// whose backing-file slice holds its NATIVE element bytes (full precision, no LZ4),
    /// alias those bytes directly from a read-only memory-mapping instead of paging them
    /// into a fresh array — eliminating the per-access allocation + memory copy (measured
    /// 86–95% of the non-compute access cost for hot, page-cache-resident weights). The OS
    /// page cache manages residency.
    ///
    /// <para><b>INFERENCE-ONLY.</b> The mapping is READ-ONLY: it must back only weights
    /// that are never written while aliased (forward-read). The pool honors this by aliasing
    /// only when the streaming execution mode is inference (not training) and the store
    /// encoding is native (not bf16/int8/lossless — those require a decode, which copies).
    /// Do not enable for training deployments: a weight update would fault on the read-only
    /// pages. Leave off unless you are serving a larger-than-RAM model for inference.</para>
    /// </summary>
    public bool EnableZeroCopyMmapResidency { get; set; } = false;

    /// <summary>
    /// Number of layers to prefetch ahead of the current Forward / Backward
    /// step. The schedule-aware prefetcher in <c>NeuralNetworkBase.Predict</c>
    /// reads layer N+W's weights from disk in the background while layer N
    /// computes — for typical CPU compute and NVMe bandwidth, W=2 fully
    /// overlaps disk I/O with compute. Set to 0 to disable prefetch and
    /// pay full disk-read latency on each layer's hot path. Default 2.
    /// </summary>
    public int PrefetchWindow { get; set; } = 2;

    /// <summary>
    /// Transparent auto-eviction (issue #430 follow-up). When <c>true</c>, the
    /// streaming pool records every weight it pages out so the
    /// <c>WeightRegistry</c> can also drop the owning tensor's resident
    /// in-memory copy after transparent auto-rehydrate (see
    /// <c>TensorBase&lt;T&gt;.EnsureMaterialized</c>). This is what lets a model
    /// fit a bounded resident set with NO per-layer orchestration code — a
    /// forward pass that reads weight after weight keeps only ~budget worth of
    /// weights live instead of accumulating every weight it touched.
    /// <para>
    /// Default <c>false</c>: with explicit orchestration
    /// (<c>MaterializeMany</c> / <c>ReleaseToPool</c>, as in
    /// <c>NeuralNetworkBase</c>) the MODEL owns weight residency, and making
    /// <c>Materialize</c> drop other weights as a side effect would both change
    /// that contract and race a concurrent reader (one thread's materialize
    /// could drop a weight another thread just materialized and is reading).
    /// Enable this ONLY for single-threaded foundation-scale inference, where
    /// the cold weight being dropped is by definition not the one in use.
    /// </para>
    /// </summary>
    public bool TransparentAutoEviction { get; set; } = false;
}

/// <summary>Memory-placement scheme for <see cref="WeightLifetime.GpuOffload"/> weights.</summary>
public enum OffloadScheme
{
    /// <summary>Allocator picks a sensible default for the backend.
    /// Today: CUDA / HIP / OpenCL / Vulkan / WebGPU map this to
    /// <see cref="Pinned"/>; Metal maps to <see cref="Managed"/> (Apple
    /// Silicon's unified memory is the natural shared path). Future
    /// iterations may probe <c>cudaDeviceProp.pageableMemoryAccess</c> /
    /// equivalent to dynamically pick Managed when supported; for now
    /// the mapping is a fixed per-backend default.</summary>
    Auto = 0,

    /// <summary>Pinned host memory + zero-copy DMA. CUDA: cudaHostAlloc with
    /// Portable+Mapped. HIP: hipHostMalloc. Metal: shared MTLBuffer.
    /// OpenCL: clEnqueueMapBuffer with CL_MEM_ALLOC_HOST_PTR. Vulkan: HOST_VISIBLE
    /// + DEVICE_LOCAL. WebGPU: GPUBufferUsage.MAP_READ | MAP_WRITE.</summary>
    Pinned = 1,

    /// <summary>Unified / managed memory; driver migrates pages on access.
    /// CUDA: cudaMallocManaged. HIP: hipMallocManaged. Metal: storageModeShared.
    /// OpenCL 2.0+: SVM. Vulkan: HOST_COHERENT + DEVICE_LOCAL.</summary>
    Managed = 2,
}
