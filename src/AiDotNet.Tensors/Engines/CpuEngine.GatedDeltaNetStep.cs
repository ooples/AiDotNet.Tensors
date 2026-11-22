using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// One decode position of the gated delta rule (Yang et al., Gated Delta Networks), per state row r, under the scan's
// GatedDeltaRule: with k' = c k, pred = d S k', delta = beta (v - pred), S <- alpha S + delta k'^T, o = S q, where
// Standard (the paper) has c = 1, d = alpha and ScaledKeys has c = 1/sqrt(Dk), d = 1 (S read before the decay). The
// state is updated in place, so a compiled decode plan keeps it in one persistent buffer (no state copy per token).
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> GatedDeltaNetStep<T>(Tensor<T> state, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> alpha, Tensor<T> beta, GatedDeltaRule rule)
    {
        var (rows, dv, dk) = ValidateGatedDeltaNetStep(state, query, key, value, alpha, beta, rule);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var (cs, cq, ck, cv, ca, cb, cr) = (state, query, key, value, alpha, beta, rule);
            // One node computes the output and writes the state in place; the state's next producer is that write.
            return scope.RecordVariadicWritingInPlace(LazyNodeType.Custom, "GatedDeltaNetStep",
                new[] { state, query, key, value, alpha, beta }, new[] { rows, dv }, state,
                (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, eng.GatedDeltaNetStep(cs, cq, ck, cv, ca, cb, cr), output));
        }
        return GatedDeltaNetStepHost(state, query, key, value, alpha, beta, rule, rows, dv, dk);
    }

    /// <summary>
    /// The composed implementation of <see cref="GatedDeltaNetStep{T}"/> from this engine's ops (the reference, and the
    /// fallback of device engines without the fused kernel): the same products the delta rule's definition names.
    /// </summary>
    private protected Tensor<T> GatedDeltaNetStepHost<T>(Tensor<T> state, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<T> alpha, Tensor<T> beta, GatedDeltaRule rule, int rows, int dv, int dk)
    {
        var (keyScale, decayFirst) = GatedDeltaRuleTerms(rule, dk);
        var scaledKey = keyScale.Equals(1.0) ? key : TensorMultiplyScalar(key, MathHelper.GetNumericOperations<T>().FromDouble(keyScale));
        var a = Reshape(alpha, new[] { rows, 1, 1 });
        var pred = BatchMatMul(state, Reshape(scaledKey, new[] { rows, dk, 1 }));                 // S k'  [rows, dv, 1]
        if (decayFirst) pred = TensorMultiply(pred, a);                                           // alpha S k'
        var delta = TensorMultiply(TensorSubtract(Reshape(value, new[] { rows, dv, 1 }), pred), Reshape(beta, new[] { rows, 1, 1 }));
        var next = TensorAdd(TensorMultiply(state, a),
            BatchMatMul(delta, Reshape(scaledKey, new[] { rows, 1, dk })));                      // alpha S + delta k'^T
        var o = Reshape(BatchMatMul(next, Reshape(query, new[] { rows, dk, 1 })), new[] { rows, dv });
        ((Compilation.ISignatureTensor)state).CopyLeafFrom(next);
        return o;
    }

    private protected static (int Rows, int Dv, int Dk) ValidateGatedDeltaNetStep<T>(Tensor<T> state, Tensor<T> query,
        Tensor<T> key, Tensor<T> value, Tensor<T> alpha, Tensor<T> beta, GatedDeltaRule rule)
    {
        if (rule is not (GatedDeltaRule.ScaledKeys or GatedDeltaRule.Standard))
            throw new ArgumentOutOfRangeException(nameof(rule), rule, "Unknown GatedDeltaRule.");
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (state.Rank != 3) throw new ArgumentException("state must be [rows, valueDim, keyDim].", nameof(state));
        if (!state.IsContiguous) throw new ArgumentException("state is updated in place and must be contiguous.", nameof(state));
        int rows = state.Shape[0], dv = state.Shape[1], dk = state.Shape[2];
        // q, k are [..., keyDim] and v [..., valueDim] with one shared leading shape that flattens to the rows, so a
        // transposed [keyDim, rows] query (same length) is rejected instead of being read as [rows, keyDim].
        var lead = RowLeadingShape(query, dk, rows, nameof(query), "keyDim");
        if (!LeadingShapeEquals(key, dk, lead)) throw new ArgumentException(
            $"key must be [{string.Join(", ", lead)}, {dk}] like query; got [{string.Join(", ", key.Shape.ToArray())}].", nameof(key));
        if (!LeadingShapeEquals(value, dv, lead)) throw new ArgumentException(
            $"value must be [{string.Join(", ", lead)}, {dv}] (query's leading shape, valueDim last); got [{string.Join(", ", value.Shape.ToArray())}].", nameof(value));
        ValidateRowGate(alpha, rows, lead, nameof(alpha));
        ValidateRowGate(beta, rows, lead, nameof(beta));
        return (rows, dv, dk);
    }

    // The leading shape of a [..., width] per-row tensor whose leading dimensions flatten to rows.
    private static int[] RowLeadingShape<T>(Tensor<T> x, int width, int rows, string name, string widthName)
    {
        if (x.Rank < 1 || x.Shape[x.Rank - 1] != width || x.Length != (long)rows * width)
            throw new ArgumentException(
                $"{name} must be [rows, {widthName}] = [{rows}, {width}] (or [..., {width}] with leading dimensions multiplying to {rows}); got [{string.Join(", ", x.Shape.ToArray())}].", name);
        var lead = new int[x.Rank - 1];
        for (int i = 0; i < lead.Length; i++) lead[i] = x.Shape[i];
        return lead;
    }

    private static bool LeadingShapeEquals<T>(Tensor<T> x, int width, int[] lead)
    {
        if (x.Rank != lead.Length + 1 || x.Shape[x.Rank - 1] != width) return false;
        for (int i = 0; i < lead.Length; i++) if (x.Shape[i] != lead[i]) return false;
        return true;
    }

    // A per-row gate is [rows], the query's leading shape, or either with a trailing singleton dimension.
    private static void ValidateRowGate<T>(Tensor<T> g, int rows, int[] lead, string name)
    {
        int rank = g.Rank > 1 && g.Shape[g.Rank - 1] == 1 ? g.Rank - 1 : g.Rank;
        bool flat = rank == 1 && g.Shape[0] == rows;
        bool matchesLead = rank == lead.Length;
        for (int i = 0; matchesLead && i < lead.Length; i++) matchesLead = g.Shape[i] == lead[i];
        if (!flat && !matchesLead)
            throw new ArgumentException(
                $"{name} must hold one value per row: [{rows}] or [{string.Join(", ", lead)}] (optionally with a trailing 1); got [{string.Join(", ", g.Shape.ToArray())}].", name);
    }
}
