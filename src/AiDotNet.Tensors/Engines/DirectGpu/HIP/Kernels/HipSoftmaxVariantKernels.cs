namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP SoftmaxVariant kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipSoftmaxVariantKernels
{
    public static string GetSource() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSoftmaxVariantKernels.GetSource();
    public static string[] GetKernelNames() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSoftmaxVariantKernels.GetKernelNames();
}
