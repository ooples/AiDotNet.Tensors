using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Optional typed dispatch surface for element-wise kernels that consume and produce FP16 storage.
/// Keeping these methods separate from <see cref="IDirectGpuBackend"/>'s FP32 contract prevents a
/// converted half buffer from being passed to a kernel that still interprets it as <see cref="float"/>.
/// </summary>
public interface IGpuFp16ElementwiseBackend
{
    /// <summary>Gets whether the backend can execute the typed operations in this interface.</summary>
    bool SupportsFp16NativeOps { get; }

    /// <summary>Computes GELU with FP16 input/output storage.</summary>
    void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>Computes ReLU with FP16 input/output storage.</summary>
    void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size);

    /// <summary>Adds two FP16 buffers into FP16 output storage.</summary>
    void Fp16Add(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size);
}
