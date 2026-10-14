namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: a linear layer's forward -- <c>C = act(A · W + bias)</c> -- as one kernel, the bias and activation
/// applied in the GEMM's epilogue instead of two more passes over the output. Equal bit for bit to the GEMM followed by
/// BiasAdd and the activation.
/// </summary>
[BackendSpecificRoute("CUDA narrow GEMM with a bias/ReLU epilogue; every backend runs the GEMM, the bias add and the activation natively",
    nameof(IDirectGpuBackend.Gemm), nameof(IDirectGpuBackend.BiasAdd), nameof(IDirectGpuBackend.Relu))]
internal interface IGemmBiasActivationBackend
{
    /// <summary><c>C[M,N] = act(A[M,K] · W[K,N] + bias[N])</c>. False -- nothing launched -- when this backend does not
    /// take the activation or the shape.</summary>
    bool TryGemmBiasActivation(IGpuBuffer A, IGpuBuffer W, IGpuBuffer bias, IGpuBuffer C, int M, int N, int K,
        FusedActivationType activation);
}
