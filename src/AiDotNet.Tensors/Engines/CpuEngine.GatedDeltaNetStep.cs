using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// One decode position of the gated delta rule (Yang et al. 2024, Gated Delta Networks), per state row r:
//   pred = S k,  delta = beta (v - pred),  S <- alpha S + delta k^T,  o = S q
// with S [Dv, Dk] the row's memory read BEFORE this position's decay. The state is updated in place, so a compiled
// decode plan keeps it in one persistent buffer (no state copy per token).
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> GatedDeltaNetStep<T>(Tensor<T> state, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> alpha, Tensor<T> beta)
    {
        var (rows, dv, dk) = ValidateGatedDeltaNetStep(state, query, key, value, alpha, beta);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var (cs, cq, ck, cv, ca, cb) = (state, query, key, value, alpha, beta);
            // One node computes the output and writes the state in place; the state's next producer is that write.
            return scope.RecordVariadicWritingInPlace(LazyNodeType.Custom, "GatedDeltaNetStep",
                new[] { state, query, key, value, alpha, beta }, new[] { rows, dv }, state,
                (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, eng.GatedDeltaNetStep(cs, cq, ck, cv, ca, cb), output));
        }
        return GatedDeltaNetStepHost(state, query, key, value, alpha, beta, rows, dv, dk);
    }

    /// <summary>
    /// The composed implementation of <see cref="GatedDeltaNetStep{T}"/> from this engine's ops (the reference, and the
    /// fallback of device engines without the fused kernel): the same products the delta rule's definition names.
    /// </summary>
    private protected Tensor<T> GatedDeltaNetStepHost<T>(Tensor<T> state, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> alpha, Tensor<T> beta, int rows, int dv, int dk)
    {
        var k = Reshape(key, new[] { rows, dk, 1 });
        var pred = BatchMatMul(state, k);                                                         // S k   [rows, dv, 1]
        var delta = TensorMultiply(TensorSubtract(Reshape(value, new[] { rows, dv, 1 }), pred), Reshape(beta, new[] { rows, 1, 1 }));
        var next = TensorAdd(TensorMultiply(state, Reshape(alpha, new[] { rows, 1, 1 })),
            BatchMatMul(delta, Reshape(key, new[] { rows, 1, dk })));                            // alpha S + delta k^T
        var o = Reshape(BatchMatMul(next, Reshape(query, new[] { rows, dk, 1 })), new[] { rows, dv });
        ((Compilation.ISignatureTensor)state).CopyLeafFrom(next);
        return o;
    }

    private protected static (int Rows, int Dv, int Dk) ValidateGatedDeltaNetStep<T>(Tensor<T> state, Tensor<T> query,
        Tensor<T> key, Tensor<T> value, Tensor<T> alpha, Tensor<T> beta)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (state.Rank != 3) throw new ArgumentException("state must be [rows, valueDim, keyDim].", nameof(state));
        if (!state.IsContiguous) throw new ArgumentException("state is updated in place and must be contiguous.", nameof(state));
        int rows = state.Shape[0], dv = state.Shape[1], dk = state.Shape[2];
        if (query.Length != rows * dk) throw new ArgumentException($"query must hold rows x keyDim = {rows * dk} values.", nameof(query));
        if (key.Length != rows * dk) throw new ArgumentException($"key must hold rows x keyDim = {rows * dk} values.", nameof(key));
        if (value.Length != rows * dv) throw new ArgumentException($"value must hold rows x valueDim = {rows * dv} values.", nameof(value));
        if (alpha.Length != rows) throw new ArgumentException($"alpha must hold one value per row ({rows}).", nameof(alpha));
        if (beta.Length != rows) throw new ArgumentException($"beta must hold one value per row ({rows}).", nameof(beta));
        return (rows, dv, dk);
    }
}
