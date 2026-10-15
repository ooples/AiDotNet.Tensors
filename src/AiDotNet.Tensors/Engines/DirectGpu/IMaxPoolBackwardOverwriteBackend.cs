namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a max-pool backward that writes EVERY cell of the input gradient (a non-overlapping pool,
/// one thread per window writing its whole block), so the caller neither zero-fills the gradient first nor has to
/// accumulate into a fresh buffer -- it can write straight into a gradient accumulator. Every backend computes the
/// backward natively as a zero-fill and a scatter.
/// </summary>
[BackendSpecificRoute("CUDA per-window max-pool backward that overwrites every input cell; every backend zero-fills and scatters natively",
    nameof(IDirectGpuBackend.MaxPool2DBackward), nameof(IDirectGpuBackend.Fill))]
internal interface IMaxPoolBackwardOverwriteBackend
{
    /// <summary>The input gradient of a max pool, every cell of <paramref name="gradInput"/> written: the window's
    /// winner (<paramref name="indices"/>, flat within the plane) gets the window's gradient, every other cell 0. False
    /// -- nothing launched -- unless the pool is non-overlapping (kernel equal to stride, the input an exact multiple of
    /// the kernel), the case where each input cell belongs to exactly one window.</summary>
    bool TryMaxPool2DBackwardOverwrite(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW);

    /// <summary>As <see cref="TryMaxPool2DBackwardOverwrite"/>, each window's winner found again from
    /// <paramref name="input"/> (its first strict maximum; a winnerless window, its first cell) instead of read from
    /// saved indices.</summary>
    bool TryMaxPool2DBackwardRecompute(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW);
}