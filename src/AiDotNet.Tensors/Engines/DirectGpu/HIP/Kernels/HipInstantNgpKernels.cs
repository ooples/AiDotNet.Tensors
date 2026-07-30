namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>HIPRTC accepts the same CUDA-like source used by the CUDA hash-grid kernels.</summary>
public static class HipInstantNgpKernels
{
    public static string[] GetKernelNames() =>
        CUDA.Kernels.CudaInstantNgpKernels.GetKernelNames();

    public static string GetSource() =>
        CUDA.Kernels.CudaInstantNgpKernels.GetSource();
}
