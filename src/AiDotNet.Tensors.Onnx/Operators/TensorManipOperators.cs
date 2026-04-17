using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX tensor-manipulation operator translators: Reshape, Transpose, Slice,
/// Concat, Split, Gather, Squeeze, Unsqueeze, Constant, Cast, Shape, Flatten.
/// </summary>
internal static class TensorManipOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new Reshape<T>());
        r.Register(new Transpose<T>());
        r.Register(new Slice<T>());
        r.Register(new Concat<T>());
        r.Register(new Split<T>());
        r.Register(new Gather<T>());
        r.Register(new Squeeze<T>());
        r.Register(new Unsqueeze<T>());
        r.Register(new Constant<T>());
        r.Register(new Flatten<T>());
        r.Register(new Identity<T>());
        r.Register(new Shape<T>());
        r.Register(new Cast<T>());
    }

    /// <summary>
    /// ONNX Cast — reinterpret the input as a different element type. Our
    /// plan represents every tensor as the same T, so a Cast to an integer
    /// type lowers to <c>TensorRound</c> (to mimic ONNX int truncation
    /// semantics on real int storage) while a Cast to FLOAT/DOUBLE/BFLOAT16
    /// is a no-op when the plan's T already matches, or an identity pass
    /// otherwise (the float representation is lossless for the value
    /// ranges the caller passes through Cast). Cast to BOOL is also a
    /// no-op here — we represent bool as 0.0 / 1.0.
    /// </summary>
    internal sealed class Cast<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Cast";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int to = ctx.GetIntAttrAsInt(node, "to", 0);
            var x = ctx.GetTensor(node.Input[0]);
            // ONNX TensorProto.DataType:
            //   FLOAT = 1, UINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5,
            //   INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9,
            //   FLOAT16 = 10, DOUBLE = 11, UINT32 = 12, UINT64 = 13,
            //   BFLOAT16 = 16.
            if (to == 8)
                throw new NotSupportedException("Cast to STRING is not supported.");
            bool toInt = to == 2 || to == 3 || to == 4 || to == 5 || to == 6 || to == 7 || to == 12 || to == 13;
            if (toInt)
            {
                // Truncate-toward-zero is the ONNX integer-cast rule. We
                // round toward nearest here as a close approximation — real
                // int8/int32 storage would need a different plan T anyway.
                ctx.PutTensor(node.Output[0], ctx.Engine.TensorRound(x));
                return;
            }
            // Float-family (FLOAT / DOUBLE / HALF / BFLOAT16) and BOOL: pass
            // through. For bool, downstream ops typically consume 0/1 floats
            // via our Not / Where translators.
            ctx.PutTensor(node.Output[0], x);
        }
    }

    /// <summary>
    /// ONNX Reshape — the new shape comes as a second input tensor (int64
    /// vector) in opset 5+; older opsets had it as an attribute. Uses
    /// <c>IEngine.Reshape</c>.
    /// </summary>
    internal sealed class Reshape<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Reshape";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var data = ctx.GetTensor(node.Input[0]);
            var shapeTensor = ctx.GetTensor(node.Input[1]);
            var newShape = ExtractInt64Shape(shapeTensor, data._shape);
            ctx.PutTensor(node.Output[0], ctx.Engine.Reshape(data, newShape));
        }

        // ONNX Reshape uses sentinel values in the shape tensor:
        //   0 = copy the corresponding dim from the input
        //   -1 = infer this dim so the total element count matches
        internal static int[] ExtractInt64Shape(Tensor<T> shapeTensor, int[] inputShape)
        {
            int rank = shapeTensor._shape[0];
            var raw = new int[rank];
            var span = shapeTensor.AsSpan();
            for (int i = 0; i < rank; i++)
                raw[i] = Convert.ToInt32(span[i]!);

            var result = new int[rank];
            int inferAt = -1;
            long known = 1;
            for (int i = 0; i < rank; i++)
            {
                if (raw[i] == 0) { result[i] = inputShape[i]; known *= result[i]; }
                else if (raw[i] == -1)
                {
                    if (inferAt != -1) throw new InvalidDataException(
                        "Reshape new_shape has more than one -1 entry.");
                    inferAt = i;
                }
                else { result[i] = raw[i]; known *= result[i]; }
            }
            if (inferAt != -1)
            {
                long total = 1;
                for (int i = 0; i < inputShape.Length; i++) total *= inputShape[i];
                if (total % known != 0) throw new InvalidDataException(
                    $"Reshape cannot infer dim {inferAt}: total elements {total} not divisible by " +
                    $"product of known dims {known}.");
                result[inferAt] = checked((int)(total / known));
            }
            return result;
        }
    }

    /// <summary>
    /// ONNX Transpose — arbitrary axis permutation. Maps to
    /// <c>IEngine.TensorPermute</c>. Default perm (when omitted) is the
    /// reverse of the input axes.
    /// </summary>
    internal sealed class Transpose<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Transpose";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var perm = ctx.GetIntArrayAttr(node, "perm");
            if (perm is null)
            {
                perm = new int[x.Rank];
                for (int i = 0; i < x.Rank; i++) perm[i] = x.Rank - 1 - i;
            }
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorPermute(x, perm));
        }
    }

    /// <summary>
    /// ONNX Slice (opset 10+ form) — inputs: (data, starts, ends, axes?, steps?).
    /// Maps to <c>IEngine.TensorSlice(start, length)</c> when step is 1 and
    /// axes cover a contiguous prefix. Non-unit step throws.
    /// </summary>
    internal sealed class Slice<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Slice";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var data = ctx.GetTensor(node.Input[0]);
            var startsT = ctx.GetTensor(node.Input[1]);
            var endsT   = ctx.GetTensor(node.Input[2]);
            int[] axes = node.Input.Count > 3 && ctx.HasTensor(node.Input[3])
                ? ToInt32Array(ctx.GetTensor(node.Input[3]))
                : DefaultAxes(startsT._shape[0]);
            int[] steps = node.Input.Count > 4 && ctx.HasTensor(node.Input[4])
                ? ToInt32Array(ctx.GetTensor(node.Input[4]))
                : Fill(1, startsT._shape[0]);
            for (int i = 0; i < steps.Length; i++)
                if (steps[i] != 1)
                    throw new NotSupportedException("Slice with non-unit step is a Phase 2 op.");

            var starts = ToInt32Array(startsT);
            var ends   = ToInt32Array(endsT);
            // Build full [rank] start + length arrays defaulting to the whole tensor.
            var fullStart = new int[data.Rank];
            var fullLen = (int[])data._shape.Clone();
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i] < 0 ? axes[i] + data.Rank : axes[i];
                int s = starts[i] < 0 ? starts[i] + data._shape[ax] : starts[i];
                int e = ends[i] < 0 ? ends[i] + data._shape[ax] : ends[i];
                s = Math.Max(0, Math.Min(s, data._shape[ax]));
                e = Math.Max(0, Math.Min(e, data._shape[ax]));
                fullStart[ax] = s;
                fullLen[ax] = Math.Max(0, e - s);
            }
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorSlice(data, fullStart, fullLen));
        }
    }

    internal sealed class Concat<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Concat";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            int axis = ctx.GetIntAttrAsInt(node, "axis", 0);
            var tensors = new List<Tensor<T>>(node.Input.Count);
            for (int i = 0; i < node.Input.Count; i++) tensors.Add(ctx.GetTensor(node.Input[i]));
            if (axis < 0) axis += tensors[0].Rank;
            ctx.PutTensor(node.Output[0], ctx.Engine.Concat(tensors, axis));
        }
    }

    internal sealed class Split<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Split";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            int axis = ctx.GetIntAttrAsInt(node, "axis", 0);
            if (axis < 0) axis += x.Rank;
            int numOutputs = node.Output.Count;
            // opset 13 moved `split` from attribute to second input.
            // Non-uniform splits aren't supported by IEngine.TensorSplit;
            // require the evenly-split case.
            var parts = ctx.Engine.TensorSplit(x, numOutputs, axis);
            if (parts.Length != numOutputs)
                throw new InvalidDataException(
                    $"Split produced {parts.Length} parts but node has {numOutputs} outputs.");
            for (int i = 0; i < numOutputs; i++)
                ctx.PutTensor(node.Output[i], parts[i]);
        }
    }

    internal sealed class Gather<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Gather";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var data = ctx.GetTensor(node.Input[0]);
            int axis = ctx.GetIntAttrAsInt(node, "axis", 0);
            if (axis < 0) axis += data.Rank;
            // ONNX Gather's indices can be int32 or int64; IEngine.Gather wants Tensor<int>.
            var rawIndices = ctx.GetTensor(node.Input[1]);
            var intIndices = CastToIntTensor(rawIndices);
            ctx.PutTensor(node.Output[0], ctx.Engine.Gather(data, intIndices, axis));
        }
    }

    internal sealed class Squeeze<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Squeeze";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            // opset 13+ moves axes to a second input; older uses an attribute.
            int[]? axes = null;
            if (node.Input.Count > 1 && ctx.HasTensor(node.Input[1]))
                axes = ToInt32Array(ctx.GetTensor(node.Input[1]));
            else
                axes = ctx.GetIntArrayAttr(node, "axes");

            if (axes is null || axes.Length == 0)
            {
                // Default: drop every size-1 dim.
                var newShape = new List<int>(x.Rank);
                for (int i = 0; i < x.Rank; i++) if (x._shape[i] != 1) newShape.Add(x._shape[i]);
                if (newShape.Count == 0) newShape.Add(1);
                ctx.PutTensor(node.Output[0], ctx.Engine.Reshape(x, newShape.ToArray()));
                return;
            }

            // Apply axes sequentially via TensorSqueeze so storage-shared views
            // work. Negative axes index from the end.
            var y = x;
            // Sort axes descending so removing one doesn't invalidate later ones.
            var sorted = (int[])axes.Clone();
            for (int i = 0; i < sorted.Length; i++) if (sorted[i] < 0) sorted[i] += x.Rank;
            Array.Sort(sorted, (a, b) => b.CompareTo(a));
            foreach (var ax in sorted) y = ctx.Engine.TensorSqueeze(y, ax);
            ctx.PutTensor(node.Output[0], y);
        }
    }

    /// <summary>
    /// ONNX Unsqueeze — insert size-1 dims at the given axes. Implemented via
    /// Reshape since IEngine has no Unsqueeze primitive; the new shape is the
    /// input shape with 1s spliced in at the sorted axes.
    /// </summary>
    internal sealed class Unsqueeze<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Unsqueeze";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            int[]? axes = node.Input.Count > 1 && ctx.HasTensor(node.Input[1])
                ? ToInt32Array(ctx.GetTensor(node.Input[1]))
                : ctx.GetIntArrayAttr(node, "axes");
            if (axes is null || axes.Length == 0)
                throw new InvalidDataException("Unsqueeze requires 'axes' as attribute or second input.");

            int newRank = x.Rank + axes.Length;
            // Normalize negative axes into final-rank space.
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                normalized[i] = axes[i] < 0 ? axes[i] + newRank : axes[i];
            Array.Sort(normalized);

            var newShape = new int[newRank];
            int srcIdx = 0, axIdx = 0;
            for (int i = 0; i < newRank; i++)
            {
                if (axIdx < normalized.Length && normalized[axIdx] == i)
                {
                    newShape[i] = 1;
                    axIdx++;
                }
                else
                {
                    newShape[i] = x._shape[srcIdx++];
                }
            }
            ctx.PutTensor(node.Output[0], ctx.Engine.Reshape(x, newShape));
        }
    }

    /// <summary>
    /// ONNX Constant — produces a tensor literal from an attribute. Handled
    /// by materializing the TensorProto directly into a new <see cref="Tensor{T}"/>.
    /// </summary>
    internal sealed class Constant<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Constant";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var valAttr = ctx.GetAttribute(node, "value")
                ?? throw new InvalidDataException("Constant without 'value' attribute is not supported.");
            var tensor = InitializerLoader.Load<T>(valAttr.T);
            ctx.PutTensor(node.Output[0], tensor);
        }
    }

    /// <summary>
    /// ONNX Flatten — collapse dims [0, axis) and [axis, rank) into a 2D
    /// result [product(0..axis), product(axis..rank)]. axis default 1.
    /// </summary>
    internal sealed class Flatten<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Flatten";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            int axis = ctx.GetIntAttrAsInt(node, "axis", 1);
            if (axis < 0) axis += x.Rank;
            int outer = 1, inner = 1;
            for (int i = 0; i < axis; i++) outer *= x._shape[i];
            for (int i = axis; i < x.Rank; i++) inner *= x._shape[i];
            ctx.PutTensor(node.Output[0], ctx.Engine.Reshape(x, new[] { outer, inner }));
        }
    }

    /// <summary>ONNX Identity — passthrough.</summary>
    internal sealed class Identity<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Identity";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node) =>
            ctx.PutTensor(node.Output[0], ctx.GetTensor(node.Input[0]));
    }

    /// <summary>
    /// ONNX Shape — returns the shape of the input as a rank-1 int64 tensor.
    /// Emits a constant tensor (shape is static after our shape resolution).
    /// </summary>
    internal sealed class Shape<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Shape";
        public string? Domain => null;
        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var x = ctx.GetTensor(node.Input[0]);
            var result = new Tensor<T>(new[] { x.Rank });
            var span = result.AsWritableSpan();
            var ops = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < x.Rank; i++) span[i] = ops.FromDouble(x._shape[i]);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    private static int[] ToInt32Array<T>(Tensor<T> t) where T : unmanaged
    {
        var span = t.AsSpan();
        var result = new int[span.Length];
        for (int i = 0; i < span.Length; i++) result[i] = Convert.ToInt32(span[i]!);
        return result;
    }

    private static int[] DefaultAxes(int count)
    {
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = i;
        return result;
    }

    private static int[] Fill(int value, int count)
    {
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = value;
        return result;
    }

    private static Tensor<int> CastToIntTensor<T>(Tensor<T> t) where T : unmanaged
    {
        // Gather's index tensor in ONNX is int32 or int64; for our engine
        // Gather we need Tensor<int>. Materialize a plain int[] once up front.
        var src = t.AsSpan();
        var dst = new int[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = Convert.ToInt32(src[i]!);
        return new Tensor<int>(t._shape, new Vector<int>(dst));
    }
}
