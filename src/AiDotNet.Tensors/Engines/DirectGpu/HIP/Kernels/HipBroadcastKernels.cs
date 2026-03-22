namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP Broadcast kernels — shared source with CUDA (HIP is CUDA-compatible).
/// </summary>
public static class HipBroadcastKernels
{
    public static string GetSource() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaBroadcastKernels.GetSource();
    public static string[] GetKernelNames() => AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaBroadcastKernels.GetKernelNames();
}
