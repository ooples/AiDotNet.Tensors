using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx.Operators;

/// <summary>
/// ONNX arithmetic operator translators: MatMul, Gemm, Add, Mul.
/// These are the fundamental tensor arithmetic ops that every model uses.
/// </summary>
internal static class ArithOperators
{
    internal static void Register<T>(OnnxOpTranslatorRegistry<T> r) where T : unmanaged
    {
        r.Register(new MatMul<T>());
        r.Register(new Gemm<T>());
        r.Register(new Add<T>());
        r.Register(new Mul<T>());
        r.Register(new Sub<T>());
        r.Register(new Div<T>());
    }

    /// <summary>
    /// ONNX MatMul — numpy-style matmul with broadcasting on leading dims.
    /// Maps directly to <c>IEngine.TensorMatMul</c> for the 2D case and
    /// <c>IEngine.TensorBatchMatMul</c> for 3D+ (where the last two dims are
    /// the matrix dims and everything before broadcasts).
    /// </summary>
    internal sealed class MatMul<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "MatMul";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);

            // ONNX MatMul spec: rank-1 operands get a singleton dim
            // prepended (a) / appended (b) for the matmul, then removed from
            // the output. E.g. Pow-1 cases: a=[K] · b=[K,N] → [N]; a=[M,K] ·
            // b=[K] → [M]; a=[K] · b=[K] → [] scalar.
            bool aWasVector = a.Rank == 1;
            bool bWasVector = b.Rank == 1;
            if (aWasVector) a = ctx.Engine.Reshape(a, new[] { 1, a._shape[0] });
            if (bWasVector) b = ctx.Engine.Reshape(b, new[] { b._shape[0], 1 });

            Tensor<T> result;
            if (a.Rank == 2 && b.Rank == 2)
            {
                result = ctx.Engine.TensorMatMul(a, b);
            }
            else
            {
                // Full-rank batched matmul with NumPy broadcasting over the
                // leading dims. Pad the smaller-rank operand with leading
                // ones, then broadcast each batch axis to the max size,
                // collapse the batch dims, run a 3D batched matmul, and
                // reshape back to the broadcast batch shape + [M, N].
                int mA = a._shape[a.Rank - 2];
                int kA = a._shape[a.Rank - 1];
                int kB = b._shape[b.Rank - 2];
                int nB = b._shape[b.Rank - 1];
                if (kA != kB)
                    throw new InvalidDataException($"MatMul inner dim mismatch: {kA} vs {kB}.");
                int maxBatchRank = Math.Max(a.Rank, b.Rank) - 2;
                // Right-aligned broadcast shape over the leading dims.
                var batchShape = new int[maxBatchRank];
                for (int i = 0; i < maxBatchRank; i++)
                {
                    int aDim = (i < maxBatchRank - (a.Rank - 2)) ? 1 : a._shape[i - (maxBatchRank - (a.Rank - 2))];
                    int bDim = (i < maxBatchRank - (b.Rank - 2)) ? 1 : b._shape[i - (maxBatchRank - (b.Rank - 2))];
                    if (aDim != bDim && aDim != 1 && bDim != 1)
                        throw new InvalidDataException(
                            $"MatMul batch shapes aren't broadcast-compatible: " +
                            $"a.shape=[{string.Join(",", a._shape)}] b.shape=[{string.Join(",", b._shape)}].");
                    batchShape[i] = Math.Max(aDim, bDim);
                }
                int totalBatch = 1;
                for (int i = 0; i < maxBatchRank; i++) totalBatch *= batchShape[i];

                // Broadcast each operand to the full batch shape (if already
                // matching shape, no-op).
                var aFull = new int[maxBatchRank + 2];
                for (int i = 0; i < maxBatchRank; i++) aFull[i] = batchShape[i];
                aFull[maxBatchRank] = mA; aFull[maxBatchRank + 1] = kA;
                var bFull = new int[maxBatchRank + 2];
                for (int i = 0; i < maxBatchRank; i++) bFull[i] = batchShape[i];
                bFull[maxBatchRank] = kB; bFull[maxBatchRank + 1] = nB;

                var aBroad = ShapesEqualND(a._shape, aFull)
                    ? a
                    : ctx.Engine.TensorBroadcastAdd(a, new Tensor<T>(aFull));
                var bBroad = ShapesEqualND(b._shape, bFull)
                    ? b
                    : ctx.Engine.TensorBroadcastAdd(b, new Tensor<T>(bFull));

                var a3 = ctx.Engine.Reshape(aBroad, new[] { totalBatch, mA, kA });
                var b3 = ctx.Engine.Reshape(bBroad, new[] { totalBatch, kB, nB });
                var r3 = ctx.Engine.TensorBatchMatMul(a3, b3);

                var outShape = new int[maxBatchRank + 2];
                for (int i = 0; i < maxBatchRank; i++) outShape[i] = batchShape[i];
                outShape[maxBatchRank] = mA;
                outShape[maxBatchRank + 1] = nB;
                result = ctx.Engine.Reshape(r3, outShape);
            }

            // Undo the rank-1 temp reshapes: drop the prepended / appended
            // singleton in the result shape.
            if (aWasVector || bWasVector)
            {
                var srcShape = result._shape;
                // Post-matmul shape is [..., M, N]. aWasVector means we
                // prepended 1 → M=1 to drop; bWasVector means we appended 1
                // → N=1 to drop.
                int newRank = srcShape.Length - (aWasVector ? 1 : 0) - (bWasVector ? 1 : 0);
                if (newRank < 0) newRank = 0;
                var finalShape = new int[newRank];
                int write = 0;
                for (int r = 0; r < srcShape.Length; r++)
                {
                    bool dropM = aWasVector && r == srcShape.Length - 2;
                    bool dropN = bWasVector && r == srcShape.Length - 1;
                    if (dropM || dropN) continue;
                    finalShape[write++] = srcShape[r];
                }
                result = ctx.Engine.Reshape(result, finalShape);
            }
            ctx.PutTensor(node.Output[0], result);
        }

        private static bool ShapesEqualND(int[] a, int[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
            return true;
        }
    }

    /// <summary>
    /// ONNX Gemm — generalized matrix multiply:
    ///   Y = alpha * (A' * B') + beta * C
    /// where A' = transA ? A^T : A and B' = transB ? B^T : B. The alpha,
    /// beta, transA, transB attributes are optional; defaults are
    /// alpha=beta=1, transA=transB=false. When C is absent the bias term
    /// is skipped.
    /// </summary>
    internal sealed class Gemm<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Gemm";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            bool transA = ctx.GetIntAttrAsInt(node, "transA", 0) != 0;
            bool transB = ctx.GetIntAttrAsInt(node, "transB", 0) != 0;
            float alpha = ctx.GetFloatAttr(node, "alpha", 1f);
            float beta = ctx.GetFloatAttr(node, "beta", 1f);

            if (transA) a = ctx.Engine.TensorTranspose(a);
            if (transB) b = ctx.Engine.TensorTranspose(b);
            var ab = ctx.Engine.TensorMatMul(a, b);

            var numOps = MathHelper.GetNumericOperations<T>();
            if (alpha != 1f)
            {
                var alphaScalar = numOps.FromDouble(alpha);
                ab = ctx.Engine.TensorMultiply(ab, ScalarTensor<T>(alphaScalar, ab._shape));
            }

            bool hasC = node.Input.Count > 2 && ctx.HasTensor(node.Input[2]);
            if (!hasC) { ctx.PutTensor(node.Output[0], ab); return; }

            var c = ctx.GetTensor(node.Input[2]);
            if (beta != 1f)
            {
                var betaScalar = numOps.FromDouble(beta);
                c = ctx.Engine.TensorMultiply(c, ScalarTensor<T>(betaScalar, c._shape));
            }
            ctx.PutTensor(node.Output[0], ctx.Engine.TensorBroadcastAdd(ab, c));
        }

        // Gemm frequently needs to scale by alpha/beta. We materialize a
        // shape-compatible tensor filled with the scalar, which the broadcast
        // multiply can consume without a special-case op.
        private static Tensor<T> ScalarTensor<TElem>(TElem value, int[] shape) where TElem : unmanaged
        {
            var t = new Tensor<T>(shape);
            var span = t.AsWritableSpan();
            var v = (T)(object)value!;
            for (int i = 0; i < span.Length; i++) span[i] = v;
            return t;
        }
    }

    /// <summary>
    /// ONNX Add — elementwise add with numpy-style broadcasting. Maps to
    /// <c>TensorAdd</c> when shapes match and <c>TensorBroadcastAdd</c>
    /// otherwise.
    /// </summary>
    internal sealed class Add<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Add";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            var result = ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorAdd(a, b)
                : ctx.Engine.TensorBroadcastAdd(a, b);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>
    /// ONNX Mul — elementwise multiply with numpy-style broadcasting.
    /// </summary>
    internal sealed class Mul<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Mul";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            var result = ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorMultiply(a, b)
                : ctx.Engine.TensorBroadcastMultiply(a, b);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>ONNX Sub — elementwise subtract.</summary>
    internal sealed class Sub<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Sub";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            var result = ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorSubtract(a, b)
                : ctx.Engine.TensorBroadcastSubtract(a, b);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    /// <summary>ONNX Div — elementwise divide with NumPy broadcasting.</summary>
    internal sealed class Div<T> : IOnnxOpTranslator<T> where T : unmanaged
    {
        public string OpType => "Div";
        public string? Domain => null;

        public void Translate(OnnxTranslationContext<T> ctx, NodeProto node)
        {
            var a = ctx.GetTensor(node.Input[0]);
            var b = ctx.GetTensor(node.Input[1]);
            var result = ShapesEqual(a._shape, b._shape)
                ? ctx.Engine.TensorDivide(a, b)
                : ctx.Engine.TensorBroadcastDivide(a, b);
            ctx.PutTensor(node.Output[0], result);
        }
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
