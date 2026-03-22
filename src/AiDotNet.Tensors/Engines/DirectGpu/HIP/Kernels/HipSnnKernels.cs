namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for SNN, RBF, PRNG, and structured sparsity operations.
/// HIP is API-compatible with CUDA — kernel source is identical.
/// </summary>
public static class HipSnnKernels
{
    public static string GetSource()
    {
        return CUDA.Kernels.CudaSnnKernels.GetSource();
    }

    public static string[] GetKernelNames()
    {
        return CUDA.Kernels.CudaSnnKernels.GetKernelNames();
    }
}
