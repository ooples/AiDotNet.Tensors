namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The multi-tensor reduction and scale kernels (multi_tensor_sum_squares, clip_scale_from_sum_squares,
// multi_tensor_scale_by_device_scalar) are implemented in CudaBackend.cs; this declares the capability.
public sealed partial class CudaBackend : IMultiTensorKernels
{
}
