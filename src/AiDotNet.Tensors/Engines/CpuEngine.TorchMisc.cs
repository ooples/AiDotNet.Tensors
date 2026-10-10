using AiDotNet.Tensors.Engines.Compilation;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Activations, fused add-multiply ops, index-mapping ops and predicates (see the matching <see cref="IEngine"/> members).</summary>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorCelu<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (alpha == 0) throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be non-zero.");
        return SpecialUnary("TensorCelu", tensor, x => x > 0 ? x : alpha * (Math.Exp(x / alpha) - 1),
            (x, _) => x > 0 ? 1 : Math.Exp(x / alpha), e => e.TensorCelu(tensor, alpha));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHardtanh<T>(Tensor<T> tensor, double minValue = -1.0, double maxValue = 1.0)
        => SpecialUnary("TensorHardtanh", tensor, x => Math.Min(Math.Max(x, minValue), maxValue),
            (x, _) => x > minValue && x < maxValue ? 1 : 0, e => e.TensorHardtanh(tensor, minValue, maxValue));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogSigmoid<T>(Tensor<T> tensor)
        => SpecialUnary("TensorLogSigmoid", tensor, x => Math.Min(x, 0) - Math.Log(1 + Math.Exp(-Math.Abs(x))),
            (x, _) => 1 / (1 + Math.Exp(x)), e => e.TensorLogSigmoid(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftsign<T>(Tensor<T> tensor)
        => SpecialUnary("TensorSoftsign", tensor, x => x / (1 + Math.Abs(x)),
            (x, _) => 1 / ((1 + Math.Abs(x)) * (1 + Math.Abs(x))), e => e.TensorSoftsign(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftmin<T>(Tensor<T> tensor, int axis = -1) => Softmax(TensorNegate(tensor), axis);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRrelu<T>(Tensor<T> tensor, double lower = 1.0 / 8, double upper = 1.0 / 3, bool training = false, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (lower > upper) throw new ArgumentException("lower must not exceed upper.", nameof(lower));
        var ops = MathHelper.GetNumericOperations<T>();
        var rng = training ? RandomSource(seed) : null;
        double evalSlope = (lower + upper) / 2;
        var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
        var values = source.AsSpan().ToArray();
        // PyTorch draws a slope for every x <= 0 (and leaky_relu's derivative at 0 is the slope), so the slope
        // array is also the exact derivative; the backward reads it rather than recovering y / x after rounding.
        var slopes = new double[values.Length];
        for (int i = 0; i < slopes.Length; i++)
            slopes[i] = ops.ToDouble(values[i]) > 0 ? 1 : rng is null ? evalSlope : lower + (upper - lower) * rng.NextDouble();
        var result = FromDoubles<T>((int[])source._shape.Clone(), i => slopes[i] * ops.ToDouble(values[i]));
        DifferentiableOps.RecordUnary("TensorRrelu", result, tensor, RreluBackward<T>, new object[] { slopes });
        return result;
    }

    private static void RreluBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var slopes = (double[])savedState[0];
        var g = (gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous()).AsSpan().ToArray();
        DifferentiableOps.AccumulateGrad(grads, inputs[0],
            FromDoubles<T>((int[])inputs[0]._shape.Clone(), i => ops.ToDouble(g[i]) * slopes[i]), engine);
    }

    private static T Scalar<T>(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRsub<T>(Tensor<T> input, Tensor<T> other, double alpha = 1)
        => TensorSubtract(other, alpha == 1 ? input : TensorMultiplyScalar(input, Scalar<T>(alpha)));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddcmul<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1)
        => TensorAdd(input, TensorMultiplyScalar(TensorMultiply(tensor1, tensor2), Scalar<T>(value)));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddcdiv<T>(Tensor<T> input, Tensor<T> tensor1, Tensor<T> tensor2, double value = 1)
        => TensorAdd(input, TensorMultiplyScalar(TensorDivide(tensor1, tensor2), Scalar<T>(value)));

    // β·input + α·product; as in PyTorch, β = 0 ignores input entirely (NaN or ∞ there does not propagate).
    private Tensor<T> ScaledSum<T>(Tensor<T> input, Tensor<T> product, double beta, double alpha)
    {
        var scaled = alpha == 1 ? product : TensorMultiplyScalar(product, Scalar<T>(alpha));
        if (beta == 0) return scaled;
        return TensorAdd(beta == 1 ? input : TensorMultiplyScalar(input, Scalar<T>(beta)), scaled);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddmv<T>(Tensor<T> input, Tensor<T> mat, Tensor<T> vec, double beta = 1, double alpha = 1)
    {
        if (mat == null) throw new ArgumentNullException(nameof(mat));
        if (vec == null) throw new ArgumentNullException(nameof(vec));
        var product = Reshape(TensorMatMul(mat, Reshape(vec, new[] { vec.Length, 1 })), new[] { mat._shape[0] });
        return ScaledSum(input, product, beta, alpha);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddr<T>(Tensor<T> input, Tensor<T> vec1, Tensor<T> vec2, double beta = 1, double alpha = 1)
    {
        if (vec1 == null) throw new ArgumentNullException(nameof(vec1));
        if (vec2 == null) throw new ArgumentNullException(nameof(vec2));
        var product = TensorMatMul(Reshape(vec1, new[] { vec1.Length, 1 }), Reshape(vec2, new[] { 1, vec2.Length }));
        return ScaledSum(input, product, beta, alpha);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBaddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1)
        => ScaledSum(input, BatchMatMul(batch1, batch2), beta, alpha);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddbmm<T>(Tensor<T> input, Tensor<T> batch1, Tensor<T> batch2, double beta = 1, double alpha = 1)
        => ScaledSum(input, ReduceSum(BatchMatMul(batch1, batch2), new[] { 0 }, keepDims: false), beta, alpha);

    // output[o] = input[source[o]] (0 where source[o] < 0); the backward scatter-adds the gradient back.
    private static Tensor<T> IndexMap<T>(string opName, Tensor<T> input, int[] outputShape, int[] source)
    {
        var values = (input.IsContiguous ? input : input.Contiguous()).AsSpan().ToArray();
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(outputShape);
        using var dstLease = result.LeaseWritable();
        var dst = dstLease.Span;
        for (int o = 0; o < source.Length; o++) dst[o] = source[o] < 0 ? ops.Zero : values[source[o]];
        DifferentiableOps.RecordUnary(opName, result, input, IndexMapBackward<T>, new object[] { source });
        return result;
    }

    private static void IndexMapBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var source = (int[])savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var g = (gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous()).AsSpan();
        var gx = new Tensor<T>((int[])inputs[0]._shape.Clone());
        using var dstLease = gx.LeaseWritable();
        var dst = dstLease.Span;
        for (int o = 0; o < source.Length; o++)
            if (source[o] >= 0) dst[source[o]] = ops.Add(dst[source[o]], g[o]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gx, engine);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMsort<T>(Tensor<T> tensor)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank == 0) return tensor;
        var ops = MathHelper.GetNumericOperations<T>();
        var values = (tensor.IsContiguous ? tensor : tensor.Contiguous()).AsSpan().ToArray();
        int extent = tensor._shape[0], inner = tensor.Length / Math.Max(1, extent);
        var source = new int[tensor.Length];
        var column = new int[extent];
        // NaN sorts last, as in PyTorch; OrderBy is stable, so ties keep their order.
        var nanLast = Comparer<double>.Create((a, b) => double.IsNaN(a) ? (double.IsNaN(b) ? 0 : 1) : double.IsNaN(b) ? -1 : a.CompareTo(b));
        for (int c = 0; c < inner; c++)
        {
            for (int k = 0; k < extent; k++) column[k] = k * inner + c;
            var sorted = column.OrderBy(i => ops.ToDouble(values[i]), nanLast).ToArray();
            for (int k = 0; k < extent; k++) source[k * inner + c] = sorted[k];
        }
        return IndexMap("TensorMsort", tensor, (int[])tensor._shape.Clone(), source);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDiagflat<T>(Tensor<T> tensor, int offset = 0)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int n = tensor.Length, size = n + Math.Abs(offset);
        var source = Enumerable.Repeat(-1, size * size).ToArray();
        for (int i = 0; i < n; i++)
        {
            int row = offset >= 0 ? i : i - offset, col = offset >= 0 ? i + offset : i;
            source[row * size + col] = i;
        }
        return IndexMap("TensorDiagflat", tensor, new[] { size, size }, source);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDiagonalScatter<T>(Tensor<T> input, Tensor<T> src, int offset = 0, int dim1 = 0, int dim2 = 1)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (src == null) throw new ArgumentNullException(nameof(src));
        int d1 = NormalizeDim(dim1, input.Rank), d2 = NormalizeDim(dim2, input.Rank);
        if (d1 == d2) throw new ArgumentException("dim1 and dim2 must differ.");
        var shape = input._shape;
        var strides = new int[shape.Length];
        for (int k = shape.Length - 1, s = 1; k >= 0; s *= shape[k], k--) strides[k] = s;
        int diagonal = Math.Max(0, Math.Min(shape[d1] - Math.Max(0, -offset), shape[d2] - Math.Max(0, offset)));
        // torch.diagonal's layout: the other axes in order, then the diagonal.
        var others = Enumerable.Range(0, shape.Length).Where(k => k != d1 && k != d2).ToArray();
        int otherCount = others.Aggregate(1, (a, k) => a * shape[k]);
        if (src.Length != otherCount * diagonal)
            throw new ArgumentException($"src has {src.Length} elements; the diagonal has {otherCount * diagonal}.", nameof(src));
        var fromSrc = Enumerable.Repeat(-1, input.Length).ToArray();
        for (int o = 0; o < otherCount; o++)
        {
            int baseIndex = 0, rest = o;
            for (int j = others.Length - 1; j >= 0; j--) { baseIndex += rest % shape[others[j]] * strides[others[j]]; rest /= shape[others[j]]; }
            for (int t = 0; t < diagonal; t++)
            {
                int i1 = t + Math.Max(0, -offset), i2 = t + Math.Max(0, offset);
                fromSrc[baseIndex + i1 * strides[d1] + i2 * strides[d2]] = o * diagonal + t;
            }
        }
        var inputValues = (input.IsContiguous ? input : input.Contiguous()).AsSpan().ToArray();
        var srcValues = (src.IsContiguous ? src : src.Contiguous()).AsSpan().ToArray();
        var result = new Tensor<T>((int[])shape.Clone());
        using var dstLease = result.LeaseWritable();
        var dst = dstLease.Span;
        for (int i = 0; i < dst.Length; i++) dst[i] = fromSrc[i] < 0 ? inputValues[i] : srcValues[fromSrc[i]];
        DifferentiableOps.RecordBinary("TensorDiagonalScatter", result, input, src, DiagonalScatterBackward<T>, new object[] { fromSrc });
        return result;
    }

    private static void DiagonalScatterBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var fromSrc = (int[])savedState[0];
        var g = (gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous()).AsSpan();
        var gInput = new Tensor<T>((int[])inputs[0]._shape.Clone());
        var gSrc = new Tensor<T>((int[])inputs[1]._shape.Clone());
        using var giLease = gInput.LeaseWritable();
        var gi = giLease.Span;
        using var gsLease = gSrc.LeaseWritable();
        var gs = gsLease.Span;
        for (int i = 0; i < fromSrc.Length; i++)
        {
            if (fromSrc[i] < 0) gi[i] = g[i];
            else gs[fromSrc[i]] = g[i];
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], gInput, engine);
        DifferentiableOps.AccumulateGrad(grads, inputs[1], gSrc, engine);
    }

    /// <inheritdoc/>
    public virtual bool TensorIsFloatingPoint<T>(Tensor<T> tensor) => MathHelper.IsFloatingPoint<T>();

    /// <inheritdoc/>
    public virtual bool TensorIsSigned<T>(Tensor<T> tensor)
        => !(typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(uint) || typeof(T) == typeof(ulong) || typeof(T) == typeof(bool));

    /// <inheritdoc/>
    public virtual bool TensorIsComplex<T>(Tensor<T> tensor)
        => typeof(T).IsGenericType && typeof(T).GetGenericTypeDefinition() == typeof(Complex<>);

    /// <inheritdoc/>
    public virtual bool TensorIsNonzero<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length != 1) throw new InvalidOperationException("TensorIsNonzero is defined only for a single-element tensor.");
        var ops = MathHelper.GetNumericOperations<T>();
        return !ops.Equals(tensor.GetFlat(0), ops.Zero);
    }

    /// <inheritdoc/>
    public virtual bool TensorIsSameSize<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        return a._shape.SequenceEqual(b._shape);
    }

    /// <inheritdoc/>
    public virtual int[] BroadcastShapes(params int[][] shapes)
    {
        if (shapes == null) throw new ArgumentNullException(nameof(shapes));
        int rank = shapes.Length == 0 ? 0 : shapes.Max(s => s.Length);
        var result = Enumerable.Repeat(1, rank).ToArray();
        foreach (var shape in shapes)
            for (int k = 0; k < shape.Length; k++)
            {
                int r = rank - shape.Length + k, d = shape[k];
                if (d == 1) continue;
                if (result[r] != 1 && result[r] != d)
                    throw new ArgumentException($"Shapes cannot broadcast: {result[r]} vs {d} at dimension {r}.");
                result[r] = d;
            }
        return result;
    }
}
