// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET7_0_OR_GREATER
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WebGPU as an <see cref="IAwaitableGpuBackend"/>: every operation returns the task of its submission, so nothing
/// here blocks the calling thread. This is the WebGPU backend for a browser, where <see cref="WebGpuBackend"/> (the
/// synchronous <see cref="IDirectGpuBackend"/>) cannot be created.
/// </summary>
/// <remarks>
/// The kernels are <see cref="WebGpuBackend"/>'s own: this object holds a private instance of it and calls only its
/// task-returning members, never the synchronous ones, which wait on those tasks.
/// </remarks>
public sealed class WebGpuAwaitableBackend : IAwaitableGpuBackend
{
    private readonly WebGpuBackend _core = new(allowSynchronousUse: false);

    /// <summary>Starts the device. Returns false when WebGPU is not available here.</summary>
    public Task<bool> InitializeAsync() => _core.InitializeAsync();

    /// <inheritdoc/>
    public bool IsAvailable => _core.IsAvailable;

    /// <inheritdoc/>
    public string BackendName => _core.BackendName;

    /// <inheritdoc/>
    public TensorDevice DeviceType => _core.DeviceType;

    /// <inheritdoc/>
    public string DeviceName => _core.DeviceName;

    /// <inheritdoc/>
    public string DeviceVendor => _core.DeviceVendor;

    /// <inheritdoc/>
    public int ComputeUnits => _core.ComputeUnits;

    /// <inheritdoc/>
    public long GlobalMemoryBytes => _core.GlobalMemoryBytes;

    /// <inheritdoc/>
    public long LocalMemoryBytes => _core.LocalMemoryBytes;

    /// <inheritdoc/>
    public long MaxBufferAllocBytes => _core.MaxBufferAllocBytes;

    /// <inheritdoc/>
    public double TheoreticalGflops => _core.TheoreticalGflops;

    /// <inheritdoc/>
    public IGpuBuffer AllocateBuffer(float[] data) => _core.AllocateBuffer(data);

    /// <inheritdoc/>
    public IGpuBuffer AllocateBuffer(int size) => _core.AllocateBuffer(size);

    /// <inheritdoc/>
    public bool SupportsElementType(GpuElementType type) => _core.SupportsElementType(type);

    /// <inheritdoc/>
    public IGpuBuffer AllocateBuffer(int size, GpuElementType type) => _core.AllocateBuffer(size, type);

    /// <inheritdoc/>
    public Task<float[]> DownloadBufferAsync(IGpuBuffer buffer) => _core.DownloadBufferAsync(buffer);

    /// <inheritdoc/>
    public Task AddAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => _core.AddAsync(A, B, C, size);

    /// <inheritdoc/>
    public Task SubtractAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => _core.SubAsync(A, B, C, size);

    /// <inheritdoc/>
    public Task MultiplyAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => _core.MulAsync(A, B, C, size);

    /// <inheritdoc/>
    public Task DivideAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => _core.DivAsync(A, B, C, size);

    /// <inheritdoc/>
    public Task ExpAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.ExpAsync(A, B, size);

    /// <inheritdoc/>
    public Task LogAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.LogAsync(A, B, size);

    /// <inheritdoc/>
    public Task SqrtAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.SqrtAsync(A, B, size);

    /// <inheritdoc/>
    public Task AbsAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.AbsAsync(A, B, size);

    /// <inheritdoc/>
    public Task NegateAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.NegAsync(A, B, size);

    /// <inheritdoc/>
    public Task ReluAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.ReLUAsync(A, B, size);

    /// <inheritdoc/>
    public Task SigmoidAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.SigmoidAsync(A, B, size);

    /// <inheritdoc/>
    public Task TanhAsync(IGpuBuffer A, IGpuBuffer B, int size) => _core.TanhAsync(A, B, size);

    /// <inheritdoc/>
    public Task GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        => _core.GemmAsync(A, B, C, M, N, K, alpha, beta);

    /// <inheritdoc/>
    public void Dispose() => _core.Dispose();
}
#endif
