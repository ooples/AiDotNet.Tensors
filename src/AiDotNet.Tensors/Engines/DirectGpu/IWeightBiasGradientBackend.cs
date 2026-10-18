namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a linear layer's weight and bias gradients in one launch -- <c>dW[K,N] = A[M,K]ᵀ · dY[M,N]</c> and
/// <c>db[N] = Σ_m dY[m, :]</c>, the bias sums taken from the dY tiles the weight-gradient kernel already stages. A
/// compiled training step otherwise ran a column-sum kernel per linear layer after its weight GEMM.
/// </summary>
[BackendSpecificRoute("CUDA weight-gradient tile kernel that also sums its dY tiles into the bias gradient; every backend runs the transposed-A GEMM and the bias column sums separately",
    nameof(IDirectGpuBackend.Gemm), nameof(IDirectGpuBackend.SumAxis))]
internal interface IWeightBiasGradientBackend
{
    /// <summary>
    /// <c>dW = A[M,K]ᵀ · dY[M,N] + betaW · dW</c> and <c>db = Σ_m dY[m, :] + betaB · db</c> (beta 0: overwritten, 1: added),
    /// with fixed summation orders. False -- nothing launched -- for a shape the backend does not serve this way.
    /// </summary>
    bool TryWeightAndBiasGradient(IGpuBuffer a, IGpuBuffer gradOutput, IGpuBuffer gradWeight, IGpuBuffer gradBias,
        int k, int n, int m, float betaW, float betaB);
}
