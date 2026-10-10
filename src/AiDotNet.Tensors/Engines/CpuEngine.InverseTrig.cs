using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Inverse trigonometry on the tensor surface, recorded on the tape.
/// </summary>
/// <remarks>
/// <para>
/// The forward maths was already here - <c>IEngine</c> exposes <c>Asin</c>, <c>Acos</c> and
/// <c>Atan</c> over spans and over <c>Vector&lt;T&gt;</c>, and <c>NativeAtan2</c> computes a
/// four-quadrant arctangent over tensors. None of it is on the tape. The <c>Vector&lt;T&gt;</c>
/// overloads sit outside autodiff entirely, and <c>NativeAtan2</c> is listed in
/// <c>OpRegistry.NonDifferentiableOps</c>.
/// </para>
/// <para>
/// That combination has a bad failure mode. An objective over angles - the phase term in a
/// prediction vocoder, for instance - compiles, runs, and trains, while the gradient through the
/// angle is silently dropped. Nothing throws; the model just never learns that path. Issue #905
/// reports exactly this, worked around by composing arctangent out of a dozen primitive ops.
/// </para>
/// <para>
/// These four record on the tape like <c>TensorSin</c> and <c>TensorCos</c> do, so the round trip
/// through an angle closes and a gradient survives it.
/// </para>
/// </remarks>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorAsin<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAsin", Math.Asin,
            static (eng, t) => eng.TensorAsin(t), BackwardFunctions<T>.AsinBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAcos<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAcos", Math.Acos,
            static (eng, t) => eng.TensorAcos(t), BackwardFunctions<T>.AcosBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAtan<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAtan", Math.Atan,
            static (eng, t) => eng.TensorAtan(t), BackwardFunctions<T>.AtanBackward);

    public virtual Tensor<T> TensorAtanh<T>(Tensor<T> tensor) =>
        InverseTrigUnary(tensor, "TensorAtanh", AtanhScalar,
            static (eng, t) => eng.TensorAtanh(t), BackwardFunctions<T>.AtanhBackward);

    /// <summary>atanh with torch.atanh's edges: +/-infinity at +/-1, NaN beyond.</summary>
    private static double AtanhScalar(double x)
    {
#if NET5_0_OR_GREATER
        return Math.Atanh(x);
#else
        // 0.5 * log1p(2x / (1 - x)), with log1p from Kahan's correction. The textbook
        // 0.5 * log((1 + x) / (1 - x)) loses every digit of a tiny x to the rounding of 1 + x.
        if (x == 1.0) return double.PositiveInfinity;
        double u = 2.0 * x / (1.0 - x);
        double w = 1.0 + u;
        return w == 1.0 ? x : 0.5 * Math.Log(w) * u / (w - 1.0);
#endif
    }
    /// <inheritdoc/>
    public virtual Tensor<T> TensorAtan2<T>(Tensor<T> y, Tensor<T> x)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (x is null) throw new ArgumentNullException(nameof(x));

        // Same shape, not merely the same length. Equal-length tensors of different shapes would be
        // paired by flat index, which is almost never what the caller meant - NativeAtan2 rejects
        // that too and this stays consistent with it.
        if (y.Rank != x.Rank)
        {
            throw new ArgumentException($"y rank ({y.Rank}) must equal x rank ({x.Rank}).", nameof(x));
        }

        for (int d = 0; d < y.Rank; d++)
        {
            if (y._shape[d] != x._shape[d])
            {
                throw new ArgumentException(
                    $"y and x must have the same shape; they differ at axis {d}: y={y._shape[d]}, x={x._shape[d]}.",
                    nameof(x));
            }
        }

        if (GraphMode.IsActive)
        {
            var ac = AutoTracer.TryGetCompiledPlan<T>("TensorAtan2", y._shape);
            if (ac is not null) return ac.Execute();
        }

        {
            var scope = GraphMode.Current; scope?.BindEngineIfUnset(this);
            if (scope != null)
            {
                var capturedY = y;
                var capturedX = x;
                return scope.RecordBinary(LazyNodeType.Custom, "TensorAtan2", y, x, y._shape,
                    (eng, output) =>
                    {
                        var r = eng.TensorAtan2(capturedY, capturedX);
                        DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                    },
                    BackwardFunctions<T>.Atan2Backward);
            }
        }

        var yOrig = y;
        var xOrig = x;
        if (!y.IsContiguous) y = y.Contiguous();
        if (!x.IsContiguous) x = x.Contiguous();

        var result = AutoTensorCache.RentOrAllocate<T>(y._shape);
        int length = y.Length;

        if (typeof(T) == typeof(double))
        {
            using var dyLease = y.LeaseArray();
            var dy = (double[])(object)dyLease.Array;
            using var dxLease = x.LeaseArray();
            var dx = (double[])(object)dxLease.Array;
            using var dstLease = result.LeaseArray();
            var dst = (double[])(object)dstLease.Array;
            for (int i = 0; i < length; i++) dst[i] = Math.Atan2(dy[i], dx[i]);
        }
        else if (typeof(T) == typeof(float))
        {
            using var fyLease = y.LeaseArray();
            var fy = (float[])(object)fyLease.Array;
            using var fxLease = x.LeaseArray();
            var fx = (float[])(object)fxLease.Array;
            using var dstLease = result.LeaseArray();
            var dst = (float[])(object)dstLease.Array;
            for (int i = 0; i < length; i++) dst[i] = MathF.Atan2(fy[i], fx[i]);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            using var ySpanLease = y.Lease();
            var ySpan = ySpanLease.Span;
            using var xSpanLease = x.Lease();
            var xSpan = xSpanLease.Span;
            using var dstLease = result.LeaseWritable();
            var dst = dstLease.Span;
            for (int i = 0; i < length; i++)
            {
                dst[i] = numOps.FromDouble(Math.Atan2(numOps.ToDouble(ySpan[i]), numOps.ToDouble(xSpan[i])));
            }
        }

        DifferentiableOps.RecordBinary("TensorAtan2", result, yOrig, xOrig, BackwardFunctions<T>.Atan2Backward);
        { var cy = y; var cx = x; AutoTracer.RecordOp("TensorAtan2", result, eng => eng.TensorAtan2(cy, cx)); }

        return result;
    }

    /// <summary>
    /// Shared body for the three single-argument inverse trigonometric functions.
    /// </summary>
    /// <remarks>
    /// They differ only in the scalar function applied and the derivative recorded, so the tape
    /// handling, the contiguity fix-up and the per-type dispatch live here once.
    /// </remarks>
    private Tensor<T> InverseTrigUnary<T>(
        Tensor<T> tensor,
        string opName,
        Func<double, double> scalar,
        Func<IEngine, Tensor<T>, Tensor<T>> reapply,
        BackwardFunction<T> backward)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        if (GraphMode.IsActive)
        {
            var ac = AutoTracer.TryGetCompiledPlan<T>(opName, tensor._shape);
            if (ac is not null) return ac.Execute();
        }

        {
            var scope = GraphMode.Current; scope?.BindEngineIfUnset(this);
            if (scope != null)
            {
                var captured = tensor;
                var capturedApply = reapply;
                return scope.RecordUnary(LazyNodeType.Custom, opName, tensor, tensor._shape,
                    (eng, output) =>
                    {
                        var r = capturedApply(eng, captured);
                        DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                    },
                    backward);
            }
        }

        var tensorOrig = tensor;  // #257: keep the user-facing reference before Contiguous() drops GradFn.
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(double))
        {
            using var srcLease = tensor.LeaseArray();
            var src = (double[])(object)srcLease.Array;
            using var dstLease = result.LeaseArray();
            var dst = (double[])(object)dstLease.Array;
            for (int i = 0; i < length; i++) dst[i] = scalar(src[i]);
        }
        else if (typeof(T) == typeof(float))
        {
            using var srcLease = tensor.LeaseArray();
            var src = (float[])(object)srcLease.Array;
            using var dstLease = result.LeaseArray();
            var dst = (float[])(object)dstLease.Array;
            for (int i = 0; i < length; i++) dst[i] = (float)scalar(src[i]);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            using var srcLease = tensor.Lease();
            var src = srcLease.Span;
            using var dstLease = result.LeaseWritable();
            var dst = dstLease.Span;
            for (int i = 0; i < length; i++)
            {
                dst[i] = numOps.FromDouble(scalar(numOps.ToDouble(src[i])));
            }
        }

        DifferentiableOps.RecordUnary(opName, result, tensorOrig, backward);
        { var c = tensor; var a = reapply; AutoTracer.RecordOp(opName, result, eng => a(eng, c)); }

        return result;
    }
}
