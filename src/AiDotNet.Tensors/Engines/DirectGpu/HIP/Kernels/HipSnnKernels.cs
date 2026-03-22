namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for SNN, RBF, PRNG, and structured sparsity operations.
/// HIP is API-compatible with CUDA — kernel source is identical.
/// </summary>
internal static class HipSnnKernels
{
    public static string GetSource()
    {
        return AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSnnKernels.GetSource();
    }

    public static string[] GetKernelNames()
    {
        return AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSnnKernels.GetKernelNames();
    }
}
