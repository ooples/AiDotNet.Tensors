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
            return Math.Max(512L * 1024 * 1024, budget);
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
