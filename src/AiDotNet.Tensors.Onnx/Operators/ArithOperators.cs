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
            Tensor<T> result;
            if (a.Rank <= 2 && b.Rank <= 2)
            {
                result = ctx.Engine.TensorMatMul(a, b);
            }
            else if (a.Rank == 3 && (b.Rank == 2 || b.Rank == 3))
            {
                result = ctx.Engine.TensorBatchMatMul(a, b);
            }
            else if (a.Rank >= 3 && b.Rank >= 3 && a.Rank == b.Rank)
            {
                // Collapse all leading batch dims into a single dim so the
                // 3D-only engine kernel applies. Reshape is a storage-sharing
                // view when the source is contiguous — but attention Q/K/V
                // come out of TensorPermute (non-contiguous strided views),
                // so Reshape across a dim-merge boundary WOULD reinterpret
                // the strided layout as row-major and produce wrong values.
                // The Permute GraphMode closure now materializes via
                // Contiguous, so at Execute time these inputs are row-major
                // — but their _shape still reflects the post-permute layout
                // which is what we want for the Reshape here.
                int batchA = 1;
                for (int i = 0; i < a.Rank - 2; i++) batchA *= a._shape[i];
                int batchB = 1;
                for (int i = 0; i < b.Rank - 2; i++) batchB *= b._shape[i];
                if (batchA != batchB)
                    throw new InvalidDataException(
                        $"MatMul batch dims don't match: a.shape = [{string.Join(",", a._shape)}], b.shape = [{string.Join(",", b._shape)}]. " +
                        "NumPy-style broadcasting across leading dims isn't yet supported.");
                int mA = a._shape[a.Rank - 2];
                int kA = a._shape[a.Rank - 1];
                int kB = b._shape[b.Rank - 2];
                int nB = b._shape[b.Rank - 1];
                if (kA != kB)
                    throw new InvalidDataException($"MatMul inner dim mismatch: {kA} vs {kB}.");
                var a3 = ctx.Engine.Reshape(a, new[] { batchA, mA, kA });
                var b3 = ctx.Engine.Reshape(b, new[] { batchB, kB, nB });
                var r3 = ctx.Engine.TensorBatchMatMul(a3, b3);
                // Reshape back to the original leading-batch form [a.shape[:-2], M, N].
                var outShape = new int[a.Rank];
                for (int i = 0; i < a.Rank - 2; i++) outShape[i] = a._shape[i];
                outShape[a.Rank - 2] = mA;
                outShape[a.Rank - 1] = nB;
                result = ctx.Engine.Reshape(r3, outShape);
            }
            else
            {
                throw new NotSupportedException(
                    $"MatMul with ranks a={a.Rank}, b={b.Rank} is not yet supported.");
            }
            ctx.PutTensor(node.Output[0], result);
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
