namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: deterministic column sums of a row-major [rows, cols] matrix into (or added to) a caller's
/// buffer -- a linear layer's bias gradient. Spread over (column tile x row chunk) blocks with a fixed summation order;
/// a ones-GEMM computes the same sums as a GEMV that, on CUDA, ran ~10 us for 2048 x 64..256.
/// </summary>
[BackendSpecificRoute("CUDA two-pass column-sum kernel for bias gradients; every backend computes them natively as a ones-GEMM or a row-sum",
    nameof(IDirectGpuBackend.Gemm), nameof(IDirectGpuBackend.SumAxis))]
internal interface IColumnSumBackend
{
    /// <summary>Floats of scratch <see cref="TryColumnSums"/> needs for [rows, cols].</summary>
    int ColumnSumScratchFloats(int rows, int cols);

    /// <summary><c>output[c] = Σ_r input[r, c]</c>, overwriting <paramref name="output"/> or, when
    /// <paramref name="add"/>, added to it. False -- nothing launched -- when unavailable or the scratch is too
    /// small.</summary>
    bool TryColumnSums(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output, int rows, int cols, bool add);
}
