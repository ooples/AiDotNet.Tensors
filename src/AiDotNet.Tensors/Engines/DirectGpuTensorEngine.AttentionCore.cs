using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    /// <remarks>
    /// On the GPU this composes the op from device-resident primitives -- per-head Reshape and TensorPermute, then
    /// ScaledDotProductAttention, then the merge permute -- each of which has its own GPU override, so the whole chain
    /// stays on the device and is differentiable and graph-capturable through them. The fused head-interleaved kernel
    /// of the base class is host float32 only; a dedicated device kernel can replace this composition later.
    /// </remarks>
    public override Tensor<T> MultiHeadAttentionCore<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, int numHeads, double? scale = null, bool causal = false)
        => base.MultiHeadAttentionCore(query, key, value, numHeads, scale, causal);
}
