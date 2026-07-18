namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Capability interface for GPU backends that provide paged-attention kernels (vLLM-style paged KV
    /// cache decode/prefill). Implemented by the OpenCL, CUDA, HIP, and Metal backends; backends without
    /// paged-attention support (e.g. Vulkan, WebGPU) simply do not implement it.
    /// </summary>
    /// <remarks>
    /// Consumers obtain a backend from <see cref="DirectGpuTensorEngine.GetBackend"/> and test for this
    /// capability: <c>if (engine.GetBackend() is IPagedAttentionBackend paged) { ... }</c>. This exposes the
    /// existing per-backend kernels (previously public only on the concrete backend types) through a stable,
    /// backend-agnostic surface so higher layers (e.g. an inference/serving engine) can run paged attention
    /// on the resident <see cref="DevicePagedKVCache"/> without depending on a specific backend type.
    /// </remarks>
    public interface IPagedAttentionBackend
    {
        /// <summary>Single-query paged-attention decode: attends one query per head over the paged KV cache.</summary>
        IGpuBuffer PagedAttentionDecode(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
            int heads, int headDim, int blockSize, int seqLen, float scale);

        /// <summary>Multi-query paged-attention prefill: attends <paramref name="numQueries"/> queries (causal).</summary>
        IGpuBuffer PagedAttentionPrefill(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
            int heads, int headDim, int blockSize, int numQueries, int startPos, float scale);

        /// <summary>Grouped-query variant of <see cref="PagedAttentionDecode"/> (kvHeads &lt; heads).</summary>
        IGpuBuffer PagedAttentionDecodeGqa(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
            int heads, int kvHeads, int headDim, int blockSize, int seqLen, float scale);

        /// <summary>Grouped-query variant of <see cref="PagedAttentionPrefill"/> (kvHeads &lt; heads).</summary>
        IGpuBuffer PagedAttentionPrefillGqa(IGpuBuffer q, IGpuBuffer kcache, IGpuBuffer vcache, IGpuBuffer blockTable,
            int heads, int kvHeads, int headDim, int blockSize, int numQueries, int startPos, float scale);
    }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : DirectGpu.IPagedAttentionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA
{
    public sealed partial class CudaBackend : DirectGpu.IPagedAttentionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP
{
    public sealed partial class HipBackend : DirectGpu.IPagedAttentionBackend { }
}

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal
{
    public sealed partial class MetalBackend : DirectGpu.IPagedAttentionBackend { }
}
