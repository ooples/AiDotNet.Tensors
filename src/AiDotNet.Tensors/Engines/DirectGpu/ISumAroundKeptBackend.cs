namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a sum over a leading and a trailing block of axes around one kept block -- a convolution's bias
/// gradient, dY [N, C, H, W] summed over N, H and W -- in two deterministic passes, without permuting the tensor so the
/// kept axis comes last. Every backend forms it natively as a permute and a row sum.
/// </summary>
[BackendSpecificRoute("CUDA and OpenCL two-pass sum around a kept axis block; every backend permutes and row-sums natively",
    nameof(IDirectGpuBackend.Permute), nameof(IDirectGpuBackend.SumAxis))]
internal interface ISumAroundKeptBackend
{
    /// <summary>Floats of scratch <see cref="TrySumAroundKept"/> needs for [leading, kept, trailing].</summary>
    int SumAroundKeptScratchFloats(int leading, int kept, int trailing);

    /// <summary><c>output[c] = scale · Σ_l Σ_t input[l, c, t]</c> for the input viewed as [leading, kept, trailing],
    /// summed in a fixed order. False -- nothing launched -- when unavailable or the scratch is too small.</summary>
    bool TrySumAroundKept(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output, int leading, int kept, int trailing,
        float scale);
}