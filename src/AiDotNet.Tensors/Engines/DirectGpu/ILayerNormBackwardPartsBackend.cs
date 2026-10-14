namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: LayerNorm's backward as separate pieces whose outputs the caller places -- the inverse std in
/// one launch, the input gradient, and the affine gradients written into (or added to) caller-supplied buffers. A
/// compiled training step then lands gamma's and beta's gradients straight in their accumulators: no zero-filled
/// temporaries and no copies, and an affine-gradient reduction that spreads over the whole device.
/// </summary>
[BackendSpecificRoute("CUDA LayerNorm backward split into placeable pieces (fused inverse std, device-wide affine-gradient reduction into the caller's buffers); every backend runs LayerNormBackward and the inverse std's AddScalar, Sqrt and Reciprocal natively",
    nameof(IDirectGpuBackend.LayerNormBackward), nameof(IDirectGpuBackend.AddScalar), nameof(IDirectGpuBackend.Sqrt),
    nameof(IDirectGpuBackend.Reciprocal))]
internal interface ILayerNormBackwardPartsBackend
{
    /// <summary><c>inverseStd[i] = 1 / sqrt(variance[i] + epsilon)</c>, computed as AddScalar, Sqrt and Reciprocal
    /// compute it. False -- nothing launched -- when unavailable.</summary>
    bool TryLayerNormInverseStd(IGpuBuffer variance, IGpuBuffer inverseStd, int rows, float epsilon);

    /// <summary>The input gradient of LayerNorm over [rows, cols] (every element of <paramref name="gradInput"/>
    /// overwritten), from the saved mean and inverse std. False -- nothing launched -- when unavailable.</summary>
    bool TryLayerNormInputGradient(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer mean,
        IGpuBuffer inverseStd, IGpuBuffer gradInput, int rows, int cols);

    /// <summary>Floats of scratch <see cref="TryLayerNormAffineGradients"/> needs for [rows, cols].</summary>
    int LayerNormAffineGradientScratchFloats(int rows, int cols);

    /// <summary>
    /// <c>gradGamma[c] = Σ_r dy[r,c] · (x[r,c] - mean[r]) · invStd[r]</c> and <c>gradBeta[c] = Σ_r dy[r,c]</c>, each
    /// overwriting its buffer or, when <paramref name="addGamma"/> / <paramref name="addBeta"/>, added to it. The
    /// summation order is fixed, so results are deterministic. False -- nothing launched -- when unavailable or the
    /// scratch is too small.
    /// </summary>
    bool TryLayerNormAffineGradients(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer mean, IGpuBuffer inverseStd,
        IGpuBuffer scratch, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int cols, bool addGamma, bool addBeta);
}
