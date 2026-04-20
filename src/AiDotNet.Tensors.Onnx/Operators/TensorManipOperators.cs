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
                // ONNX integer cast = truncate-toward-zero
                // (1.9 → 1, -1.9 → -1). Implement as sign(x) * floor(abs(x))
                // since the engine has no native trunc-to-int kernel but
                // does have Floor and Abs. (Using TensorRound here would
                // round 1.5 → 2 and diverge from ONNX.)
                var absX = ctx.Engine.TensorAbs(x);
                var floored = ctx.Engine.TensorFloor(absX);
                var sign = ctx.Engine.TensorSign(x);
                ctx.PutTensor(node.Output[0], ctx.Engine.TensorMultiply(sign, floored));
                return;
            }
            if (to == 9)
            {
                // BOOL: normalize to 0/1. Forwarding the raw tensor preserved
                // magnitudes (-3, 2, etc.), which broke downstream consumers
                // that expected boolean {0, 1} — a subsequent Cast(…, INT) or
                // arithmetic Where/Not would silently use the wrong values.
                // Emit (x != 0) → 1 else 0 via sign-of-abs:
                //   |x| > 0 ⇒ sign(|x|) == 1 ; |x| == 0 ⇒ sign == 0.
                var absX = ctx.Engine.TensorAbs(x);
                var normalized = ctx.Engine.TensorSign(absX);
                ctx.PutTensor(node.Output[0], normalized);
                return;
            }
            // Float-family (FLOAT / DOUBLE / HALF / BFLOAT16): pass through
            // since our plan represents every tensor as the same T.
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
            // ONNX 14+: allowzero=1 means a 0 in the shape tensor stays a
            // literal 0 dimension instead of the default "copy input dim"
            // sentinel. Default (0) preserves the legacy behaviour.
            bool allowZero = ctx.GetIntAttrAsInt(node, "allowzero", 0) != 0;
            var newShape = ExtractInt64Shape(shapeTensor, data._shape, allowZero);
            ctx.PutTensor(node.Output[0], ctx.Engine.Reshape(data, newShape));
        }

        // ONNX Reshape uses sentinel values in the shape tensor:
        //   0 = copy the corresponding dim from the input (opset < 14, or
        //       opset 14+ with allowzero=0). When allowzero=1 a 0 stays
        //       literal zero (valid when the graph genuinely wants an empty
        //       dimension — producers like HuggingFace attention-mask paths
        //       occasionally emit this).
        //   -1 = infer this dim so the total element count matches.
        internal static int[] ExtractInt64Shape(Tensor<T> shapeTensor, int[] inputShape, bool allowZero = false)
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
                if (raw[i] == 0)
                {
                    // allowzero=0 (default): copy the corresponding input dim
                    // (requires a corresponding axis exists in input). With
                    // allowzero=1 the 0 is a literal dimension.
                    result[i] = allowZero ? 0 : (i < inputShape.Length ? inputShape[i] : 0);
                    known *= result[i];
                }
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
                // Guard against divide-by-zero when the product of known
                // dims contains a 0 (e.g. allowzero=1 with -1 in the same
                // shape). In that case the -1 cannot be meaningfully
                // inferred — the ONNX spec says the element count must
                // match, which it does only vacuously when either side is
                // zero. Surface a crisp error instead of a runtime crash.
                if (known == 0)
                    throw new InvalidDataException(
                        $"Reshape cannot infer dim {inferAt}: the product of known dims is zero " +
                        "(typically allowzero=1 combined with a -1 entry or a zero-length input).");
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

            var starts = ToInt32Array(startsT);
            var ends   = ToInt32Array(endsT);
            // ONNX Slice requires starts, ends, axes (and steps, if present)
            // to all have the same length — each entry describes one sliced
            // axis. A length mismatch would IndexOutOfRange a few lines down;
            // fail fast with a clear diagnostic so the offending node is
            // identifiable.
            if (starts.Length != ends.Length || starts.Length != axes.Length || steps.Length != axes.Length)
                throw new InvalidDataException(
                    $"Slice expected starts/ends/axes/steps to all have the same length; got " +
                    $"{starts.Length}/{ends.Length}/{axes.Length}/{steps.Length}.");
            // Build full [rank] start + length + step arrays defaulting to
            // the whole tensor with unit step. Non-unit (including negative)
            // steps are honored below via a strided scatter.
            var fullStart = new int[data.Rank];
            var fullLen = (int[])data._shape.Clone();
            var fullStep = new int[data.Rank];
            for (int i = 0; i < data.Rank; i++) fullStep[i] = 1;
            bool anyNonUnitStep = false;
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i] < 0 ? axes[i] + data.Rank : axes[i];
                int step = steps[i];
                if (step == 0)
                    throw new InvalidDataException("Slice step must be non-zero.");
                int dim = data._shape[ax];
                int s = starts[i] < 0 ? starts[i] + dim : starts[i];
                int e = ends[i] < 0 ? ends[i] + dim : ends[i];
                if (step > 0)
                {
                    s = Math.Max(0, Math.Min(s, dim));
                    e = Math.Max(0, Math.Min(e, dim));
                }
                else
                {
                    // For negative step, valid range is [-1, dim-1] (ONNX
                    // spec). Clamp defensively.
                    s = Math.Max(-1, Math.Min(s, dim - 1));
                    e = Math.Max(-1, Math.Min(e, dim - 1));
                }
                fullStart[ax] = s;
                fullLen[ax] = step > 0
                    ? (e > s ? (e - s + step - 1) / step : 0)
                    : (s > e ? (s - e + (-step) - 1) / (-step) : 0);
                fullStep[ax] = step;
                if (step != 1) anyNonUnitStep = true;
            }
            if (!anyNonUnitStep)
            {
                ctx.PutTensor(node.Output[0], ctx.Engine.TensorSlice(data, fullStart, fullLen));
                return;
            }
            // Strided Slice — engine has no native non-unit-step op. Compose
            // via a per-element strided scatter recorded as a LazyNode so
            // the read of `data` happens at Execute time (matches the
            // deferred-read idiom used by Gather/OneHot).
            var outShape = (int[])fullLen.Clone();
            int outSize = 1;
            for (int i = 0; i < outShape.Length; i++) outSize *= outShape[i];
            var capturedData = data;
            var capturedStart = fullStart;
            var capturedLen = fullLen;
            var capturedStep = fullStep;
            var scope = Engines.Compilation.GraphMode.Current;
            if (scope is null)
            {
                // Eager path.
                var eagerResult = new LinearAlgebra.Tensor<T>(outShape);
                StridedCopy(capturedData, eagerResult, capturedStart, capturedLen, capturedStep);
                ctx.PutTensor(node.Output[0], eagerResult);
                return;
            }
            var lazy = scope.RecordUnary(Engines.Compilation.LazyNodeType.Custom,
                "Slice", data, outShape,
                (eng, output) => StridedCopy(capturedData, output, capturedStart, capturedLen, capturedStep));
            ctx.PutTensor(node.Output[0], lazy);
        }

        /// <summary>
        /// Copies elements of <paramref name="src"/> at starts/step strides
        /// into <paramref name="dst"/>. Generic over rank; uses a flat
        /// output-index to multi-index walk so any rank + negative step
        /// combination is correct.
        /// </summary>
        private static void StridedCopy(LinearAlgebra.Tensor<T> src, LinearAlgebra.Tensor<T> dst,
            int[] start, int[] len, int[] step)
        {
            var srcSpan = src.AsSpan();
            var dstSpan = dst.AsWritableSpan();
            int rank = start.Length;
            var srcStrides = new int[rank];
            int s = 1;
            for (int i = rank - 1; i >= 0; i--) { srcStrides[i] = s; s *= src._shape[i]; }
            int dstLen = dstSpan.Length;
            var coord = new int[rank];
            for (int outIdx = 0; outIdx < dstLen; outIdx++)
            {
                int rem = outIdx;
                for (int i = rank - 1; i >= 0; i--)
                {
                    coord[i] = rem % len[i];
                    rem /= len[i];
                }
                int srcIdx = 0;
                for (int i = 0; i < rank; i++)
                    srcIdx += (start[i] + coord[i] * step[i]) * srcStrides[i];
                dstSpan[outIdx] = srcSpan[srcIdx];
            }
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
            // tf2onnx and some HF exports emit Concat across operands whose
            // ranks differ (ONNX Runtime auto-aligns; the ONNX spec itself
            // requires matching rank). Prepend size-1 axes to each operand
            // until every tensor reaches the maximum rank in the list, then
            // forward to the engine's same-rank Concat.
            int maxRank = 0;
            for (int i = 0; i < tensors.Count; i++)
                if (tensors[i].Rank > maxRank) maxRank = tensors[i].Rank;
            for (int i = 0; i < tensors.Count; i++)
            {
                if (tensors[i].Rank < maxRank)
                {
                    int pad = maxRank - tensors[i].Rank;
                    var padShape = new int[maxRank];
                    for (int k = 0; k < pad; k++) padShape[k] = 1;
                    for (int k = pad; k < maxRank; k++) padShape[k] = tensors[i]._shape[k - pad];
                    tensors[i] = ctx.Engine.Reshape(tensors[i], padShape);
                }
            }
            if (axis < 0) axis += maxRank;
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
            // Opset 13 moved `split` sizes from attribute to second input.
            // When present, verify each piece is the same size — our engine
            // only supports uniform splits today. The older attribute-based
            // form (opset < 13) is also supported: inspect the attribute,
            // enforce the same uniformity contract.
            int[]? splitSizes = null;
            if (node.Input.Count > 1 && ctx.HasTensor(node.Input[1]))
                splitSizes = ToInt32Array(ctx.GetTensor(node.Input[1]));
            else
                splitSizes = ctx.GetIntArrayAttr(node, "split");
            if (splitSizes is not null && splitSizes.Length != numOutputs)
                throw new InvalidDataException(
                    $"Split sizes length {splitSizes.Length} != number of outputs {numOutputs}.");
            // Check uniformity — if every piece is the same size, route
            // through the fast engine.TensorSplit path. Non-uniform falls
            // through to per-piece engine.TensorSlice below.
            bool uniform = splitSizes is null;
            if (splitSizes is not null)
            {
                uniform = true;
                for (int i = 1; i < splitSizes.Length; i++)
                    if (splitSizes[i] != splitSizes[0]) { uniform = false; break; }
            }
            if (uniform)
            {
                var parts = ctx.Engine.TensorSplit(x, numOutputs, axis);
                if (parts.Length != numOutputs)
                    throw new InvalidDataException(
                        $"Split produced {parts.Length} parts but node has {numOutputs} outputs.");
                for (int i = 0; i < numOutputs; i++)
                    ctx.PutTensor(node.Output[i], parts[i]);
                return;
            }
            // Non-uniform split: emit one TensorSlice per piece.
            if (splitSizes!.Sum() != x._shape[axis])
                throw new InvalidDataException(
                    $"Split sizes [{string.Join(",", splitSizes!)}] sum {splitSizes!.Sum()} " +
                    $"!= axis {axis} size {x._shape[axis]}.");
            int offset = 0;
            for (int i = 0; i < numOutputs; i++)
            {
                var fullStart = new int[x.Rank];
                var fullLen = (int[])x._shape.Clone();
                fullStart[axis] = offset;
                fullLen[axis] = splitSizes![i];
                ctx.PutTensor(node.Output[i], ctx.Engine.TensorSlice(x, fullStart, fullLen));
                offset += splitSizes![i];
            }
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
            // ONNX Gather's indices can be int32 or int64; engine.Gather wants
            // Tensor<int>. Our plan's T is float (BERT token IDs are stored
            // as floats holding integer values), so the cast to int32 has to
            // happen AT EXECUTE TIME — taking it at trace time captures the
            // uninitialized placeholder (all zeros) instead of the caller's
            // actual token IDs.
            var indicesTensor = ctx.GetTensor(node.Input[1]);

            // Output shape: data shape with axis dim replaced by indices
            // shape (indices can be multi-dim, e.g. [B, S] gathering rows of
            // [V, H] produces [B, S, H]).
            int newRank = data.Rank - 1 + indicesTensor.Rank;
            var outShape = new int[newRank];
            int o = 0;
            for (int i = 0; i < axis; i++) outShape[o++] = data._shape[i];
            for (int i = 0; i < indicesTensor.Rank; i++) outShape[o++] = indicesTensor._shape[i];
            for (int i = axis + 1; i < data.Rank; i++) outShape[o++] = data._shape[i];

            var scope = Engines.Compilation.GraphMode.Current;
            if (scope is null)
            {
                // Eager path — caller is inside a constant-folding block, so
                // indices are real. Do the int cast here and call engine.Gather.
                var intIndicesEager = CastToIntTensor(indicesTensor);
                ctx.PutTensor(node.Output[0], ctx.Engine.Gather(data, intIndicesEager, axis));
                return;
            }

            // Lazy path — record a BINARY LazyNode that takes both data and
            // indices as tracked inputs. Passing indicesTensor via RecordBinary
            // (rather than capturing it in the closure) tells the memory
            // planner that indices are live through the Gather step — so it
            // won't reuse the indices buffer for an earlier-finishing op.
            var capturedData = data;
            var capturedIndices = indicesTensor;
            int capturedAxis = axis;
            var lazy = scope.RecordBinary(Engines.Compilation.LazyNodeType.Custom,
                "Gather", capturedData, capturedIndices, outShape,
                (eng, output) =>
                {
                    // Read the indices as int32 at Execute time — capturedIndices
                    // now has the actual values since Gather runs after the
                    // upstream op that produced them.
                    var idxSrc = capturedIndices.AsSpan();
                    var idxArr = new int[idxSrc.Length];
                    for (int i = 0; i < idxSrc.Length; i++)
                    {
                        double v = Convert.ToDouble(idxSrc[i]!);
                        idxArr[i] = v >= int.MaxValue ? int.MaxValue
                                  : v <= int.MinValue ? int.MinValue
                                  : (int)Math.Round(v);
                    }
                    if (GatherDebug.Enabled)
                    {
                        int stepNo = System.Threading.Interlocked.Increment(ref GatherDebug.StepCounter);
                        GatherDebug.Log($"step#{stepNo} Gather ilen={idxArr.Length} idx[0..3]={(idxArr.Length > 0 ? idxArr[0] : -1)},{(idxArr.Length > 1 ? idxArr[1] : -1)},{(idxArr.Length > 2 ? idxArr[2] : -1)}");
                    }
                    var intIndices = new Tensor<int>(capturedIndices._shape, new Vector<int>(idxArr));
                    var r = eng.Gather(capturedData, intIndices, capturedAxis);
                    r.AsSpan().CopyTo(output.AsWritableSpan());
                });
            ctx.PutTensor(node.Output[0], lazy);
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
            // Normalize negative axes into final-rank space, validate each
            // falls inside [0, newRank) — ONNX requires axes to be unique
            // and within [-output_rank, output_rank-1]. Duplicates or
            // out-of-range values silently corrupt the newShape scan below
            // (srcIdx tracking breaks), so fail fast with a clear error.
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i] < 0 ? axes[i] + newRank : axes[i];
                if (ax < 0 || ax >= newRank)
                    throw new InvalidDataException(
                        $"Unsqueeze axis {axes[i]} out of range for output rank {newRank}.");
                normalized[i] = ax;
            }
            Array.Sort(normalized);
            for (int i = 1; i < normalized.Length; i++)
                if (normalized[i] == normalized[i - 1])
                    throw new InvalidDataException(
                        $"Unsqueeze axes must be unique; got duplicate {normalized[i]}.");

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
        for (int i = 0; i < span.Length; i++)
        {
            // ONNX Slice uses INT_MAX / INT_MIN as "to end" sentinels on
            // int64 "starts" / "ends" inputs. Loaded as float those look
            // like ±9.2e18; clamp to int range so the downstream slice
            // logic treats them as "clip to dim length" naturally.
            double d = Convert.ToDouble(span[i]!);
            if (d >= int.MaxValue) result[i] = int.MaxValue;
            else if (d <= int.MinValue) result[i] = int.MinValue;
            else result[i] = (int)Math.Round(d);
        }
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
