// Copyright (c) AiDotNet. All rights reserved.
// Metal FFT dispatcher — implements IFftBackend via parity212_fft.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IFftBackend
{
    private const string FftLibName = "Fft212";

    private MetalPipelineState GetFftPipeline(string kernelName)
    {
        if (_fftLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal FFT library was not compiled (shader compile failed at init). " +
                "Callers should catch and fall back to the CPU FftKernels path.");
        return GetPipeline(FftLibName, _fftLibrary, kernelName);
    }

    /// <inheritdoc />
    public void LaunchFft(IGpuBuffer buffer, int batchCount, int n, bool inverse)
    {
        ThrowIfDisposed();
        if (batchCount <= 0 || n <= 0) return;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException(
                $"Metal LaunchFft requires n to be a power of two (got n = {n}). " +
                "Non-pow-2 lengths must route through the CPU Bluestein path.",
                nameof(n));
        if (buffer is not MetalGpuBuffer mbuf)
            throw new ArgumentException("Buffer must be MetalGpuBuffer");

        var pipeline = GetFftPipeline("parity212_fft");
        // Thread-group size: 256 threads per workgroup is standard for Metal;
        // matches the workgroup_size used by the peer backends.
        uint tpg = Math.Min(1024u, Math.Max(32u, (uint)n));
        // Round to next power of two for cleaner dispatch.
        uint pw = 1;
        while (pw < tpg) pw <<= 1;
        tpg = Math.Min(256u, pw);

        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(mbuf, 0);
        encoder.SetBytes(batchCount, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(inverse ? 1 : 0, 3);
        encoder.DispatchThreadgroups(
            new MetalNativeBindings.MTLSize((uint)batchCount, 1, 1),
            new MetalNativeBindings.MTLSize(tpg, 1, 1));
    }
}
