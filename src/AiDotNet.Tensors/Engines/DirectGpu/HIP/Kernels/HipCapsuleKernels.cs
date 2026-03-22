namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for capsule network operations.
/// HIP is API-compatible with CUDA — kernel source is identical.
/// </summary>
internal static class HipCapsuleKernels
{
    public static string GetSource()
    {
        // HIP kernel source is identical to CUDA — same __global__ syntax
        return CUDA.Kernels.CudaCapsuleKernels.GetSource();
    }

    public static string[] GetKernelNames()
    {
        return CUDA.Kernels.CudaCapsuleKernels.GetKernelNames();
    }
}
