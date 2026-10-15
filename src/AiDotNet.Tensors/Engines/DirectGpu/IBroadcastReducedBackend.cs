namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: the gradient of a sum or mean over one middle block of axes in one pass --
/// <c>output[o, r, i] = scale * g[o, i]</c> for an input viewed as [outer, reduced, inner] -- overwriting the output or
/// added to it, so a resident step can write it straight into a gradient accumulator. Every backend forms it natively
/// as a batched ones-GEMM and a scale into a fresh buffer.
/// </summary>
[BackendSpecificRoute("CUDA one-pass broadcast of a middle-axis reduction's gradient; every backend forms it natively as a batched ones-GEMM and a scale",
    nameof(IDirectGpuBackend.BatchedGemm), nameof(IDirectGpuBackend.Scale))]
internal interface IBroadcastReducedBackend
{
    /// <summary><c>output[(o * reduced + r) * inner + i] = scale * g[o * inner + i]</c>, overwriting
    /// <paramref name="output"/> or, when <paramref name="add"/>, added to it. False -- nothing launched -- when
    /// unavailable.</summary>
    bool TryBroadcastMiddle(IGpuBuffer g, IGpuBuffer output, int outer, int reduced, int inner, float scale, bool add);
}
