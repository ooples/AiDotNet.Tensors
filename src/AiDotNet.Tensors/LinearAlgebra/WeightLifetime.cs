// Copyright (c) AiDotNet. All rights reserved.

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
    /// the pool exceeds this, oldest unused entries evict. Default 16 GiB.</summary>
    public long StreamingPoolMaxResidentBytes { get; set; } = 16L * 1024 * 1024 * 1024;

    /// <summary>Backing store path for the streaming pool. Null ⇒ use the
    /// system temp dir. Memory-mapped file is the default backing format.</summary>
    public string? StreamingBackingStorePath { get; set; }

    /// <summary>
    /// LZ4-compress weight bytes before writing them to the backing store.
    /// On near-Gaussian fp32 weights this typically saves 30–40% of disk
    /// footprint at ~3 GB/s decompress (negligible vs. NVMe bandwidth).
    /// Default false to keep the basic streaming path identical to the
    /// pre-compression behaviour; enable for memory-bound (562B) models.
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
