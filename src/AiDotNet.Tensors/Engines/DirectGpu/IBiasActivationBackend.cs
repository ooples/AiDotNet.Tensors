namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a linear layer's bias add and activation as one pass over its output, where a bias add and an
/// activation kernel were two full passes (a Transformer FFN's first layer, every step).
/// </summary>
[BackendSpecificRoute("CUDA bias add fused with ReLU; every backend runs the bias add and the ReLU natively",
    nameof(IDirectGpuBackend.BiasAdd), nameof(IDirectGpuBackend.Relu))]
internal interface IBiasActivationBackend
{
    /// <summary><c>C = act(A + bias)</c> per row of [M, N], computed exactly as BiasAdd then the activation would.
    /// False -- nothing launched -- when <paramref name="activation"/> is not fused by this backend.</summary>
    bool TryBiasAddActivation(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N, FusedActivationType activation);
}
