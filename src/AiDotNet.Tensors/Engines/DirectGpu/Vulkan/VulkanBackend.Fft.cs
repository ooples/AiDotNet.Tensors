// Copyright (c) AiDotNet. All rights reserved.
// Vulkan FFT dispatcher — implements IFftBackend by compiling and launching
// VulkanFftKernels.Fft via the existing GlslUnaryOp compute pipeline. The
// kernel is in-place algorithmically; we use a (readonly input, in-place
// output) SSBO layout so it slots cleanly into GlslUnaryOp's two-binding
// pipeline without needing a single-buffer variant.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IFftBackend
{
    /// <inheritdoc />
    public void LaunchFft(IGpuBuffer buffer, int batchCount, int n, bool inverse)
    {
        if (batchCount <= 0 || n <= 0) return;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException(
                $"Vulkan LaunchFft requires n to be a power of two (got n = {n}). " +
                "Non-pow-2 lengths must route through the CPU Bluestein path.",
                nameof(n));

        uint[] pc = { (uint)batchCount, (uint)n, (uint)(inverse ? 1 : 0) };
        // One workgroup per batch slice; workgroup size 256 handles up to
        // n = 512 butterflies per slice with one butterfly per thread.
        GlslUnaryOp(
            VulkanFftKernels.Fft,
            buffer, buffer, // same buffer in both slots — algorithm is in-place
            dispatchSize: batchCount * 256,
            pushConstants: pc,
            pushConstantSize: sizeof(uint) * 3u);
    }
}
