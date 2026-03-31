namespace AiDotNet.Tensors;

/// <summary>
/// Represents the device where a tensor's data currently resides.
/// Matches PyTorch's tensor.device concept — tensors track their location
/// so operations can avoid unnecessary CPU-GPU transfers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This tells you where the tensor's data physically lives.
/// CPU means your regular computer memory. GPU means the graphics card's memory,
/// which is much faster for parallel math operations like neural networks.</para>
/// <para>When a GPU operation produces a result, the tensor is marked as GPU-resident.
/// If CPU code accesses the data (e.g., reading a value), it triggers a lazy download
/// and the device changes to CPU. This avoids unnecessary transfers between CPU and GPU.</para>
/// </remarks>
public enum TensorDevice
{
    /// <summary>
    /// Data resides in CPU memory. This is the default for all tensors.
    /// </summary>
    CPU = 0,

    /// <summary>
    /// Data resides in GPU memory via CUDA (NVIDIA GPUs).
    /// </summary>
    CUDA = 1,

    /// <summary>
    /// Data resides in GPU memory via OpenCL (cross-platform GPUs including AMD, Intel, NVIDIA).
    /// </summary>
    OpenCL = 2,

    /// <summary>
    /// Data resides in GPU memory via HIP (AMD GPUs with ROCm).
    /// </summary>
    HIP = 3,

    /// <summary>
    /// Data resides in GPU memory via Vulkan compute shaders (cross-platform).
    /// </summary>
    Vulkan = 4,

    /// <summary>
    /// Data resides in GPU memory via Metal Performance Shaders (Apple Silicon/macOS).
    /// </summary>
    Metal = 5,

    /// <summary>
    /// Data resides in GPU memory via WebGPU (browser and cross-platform GPU access).
    /// </summary>
    WebGPU = 6,

    /// <summary>
    /// Data resides in GPU memory via DirectML (Windows ML accelerator, supports DirectX 12 GPUs).
    /// </summary>
    DirectML = 7,

    /// <summary>
    /// Data resides on a neural processing unit (NPU/TPU/specialized AI accelerator).
    /// </summary>
    NPU = 8
}
