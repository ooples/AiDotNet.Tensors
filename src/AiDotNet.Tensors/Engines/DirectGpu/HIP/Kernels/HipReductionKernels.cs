namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP Reduction kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipReductionKernels
{
    public static string GetSource() => CUDA.Kernels.CudaReductionKernels.GetSource();
    public static string[] GetKernelNames() => CUDA.Kernels.CudaReductionKernels.GetKernelNames();
}
