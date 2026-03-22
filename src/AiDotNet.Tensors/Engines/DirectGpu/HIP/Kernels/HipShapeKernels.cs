namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP Shape kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipShapeKernels
{
    public static string GetSource() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaShapeKernels.GetSource();
    public static string[] GetKernelNames() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaShapeKernels.GetKernelNames();
}
