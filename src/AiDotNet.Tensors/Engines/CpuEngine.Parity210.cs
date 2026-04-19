using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Parity-210 op additions — movement, cumulative, comparison, clamp,
/// special math, element-wise binary, indexing completeness. Kept in a
/// separate partial to avoid bloating CpuEngine.cs.
/// </summary>
public partial class CpuEngine
{
    // ==================================================================
    // Movement ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRoll<T>(Tensor<T> tensor, int[] shifts, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (shifts == null) throw new ArgumentNullException(nameof(shifts));
        if (axes == null) throw new ArgumentNullException(nameof(axes));
        if (shifts.Length != axes.Length)
            throw new ArgumentException("shifts and axes must have the same length");

        int rank = tensor.Rank;
        var shape = tensor._shape;

        // Per-axis effective shift, normalised into [0, dim).
        var effShift = new int[rank];
        foreach (var a in axes) if (a < 0 || a >= rank)
            throw new ArgumentOutOfRangeException(nameof(axes), $"axis {a} out of range [0, {rank})");
        for (int i = 0; i < axes.Length; i++)
        {
            int a = axes[i];
            int d = shape[a];
            if (d == 0) continue;
            int s = shifts[i] % d;
            if (s < 0) s += d;
            effShift[a] = (effShift[a] + s) % d;
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        // Iterate every index; destination position = (i_k + shift_k) mod d_k.
        var idx = new int[rank];
        int total = tensor.Length;
        var strides = ComputeRowMajorStrides(shape);
        for (int linear = 0; linear < total; linear++)
        {
            // Compute destination linear index applying rolls.
            int dstLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int newIdx = idx[k] + effShift[k];
                if (newIdx >= shape[k]) newIdx -= shape[k];
                dstLinear += newIdx * strides[k];
            }
            dst[dstLinear] = src[linear];
            // increment idx
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorRoll", result, tensor,
            BackwardFunctions<T>.RollBackward,
            savedState: new object[] { (int[])shifts.Clone(), (int[])axes.Clone() });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFlip<T>(Tensor<T> tensor, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (axes == null) throw new ArgumentNullException(nameof(axes));

        int rank = tensor.Rank;
        foreach (var a in axes) if (a < 0 || a >= rank)
            throw new ArgumentOutOfRangeException(nameof(axes), $"axis {a} out of range");
        var flipSet = new System.Collections.Generic.HashSet<int>(axes);

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var shape = tensor._shape;
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var idx = new int[rank];
        int total = tensor.Length;
        var strides = ComputeRowMajorStrides(shape);
        for (int linear = 0; linear < total; linear++)
        {
            int dstLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int newIdx = flipSet.Contains(k) ? shape[k] - 1 - idx[k] : idx[k];
                dstLinear += newIdx * strides[k];
            }
            dst[dstLinear] = src[linear];
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorFlip", result, tensor,
            BackwardFunctions<T>.FlipBackward,
            savedState: new object[] { (int[])axes.Clone() });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRepeatInterleave<T>(Tensor<T> tensor, int repeats, int dim)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats), "repeats must be >= 1");
        int rank = tensor.Rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var inShape = tensor._shape;
        var outShape = (int[])inShape.Clone();
        outShape[dim] = inShape[dim] * repeats;

        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(outShape);
        var dst = result.AsWritableSpan();

        // Iterate over the output's index space, map each to the source.
        var idx = new int[rank];
        int total = result.Length;
        var srcStrides = ComputeRowMajorStrides(inShape);
        for (int linear = 0; linear < total; linear++)
        {
            // Source index = same, except dim is idx[dim] / repeats.
            int srcLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int i = (k == dim) ? idx[k] / repeats : idx[k];
                srcLinear += i * srcStrides[k];
            }
            dst[linear] = src[srcLinear];
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < outShape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorRepeatInterleave", result, tensor,
            BackwardFunctions<T>.RepeatInterleaveBackward,
            savedState: new object[] { repeats, dim });
        return result;
    }

    // ==================================================================
    // Cumulative ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumProd<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis, MathHelper.GetNumericOperations<T>().One,
            (a, b) => MathHelper.GetNumericOperations<T>().Multiply(a, b),
            "TensorCumProd", BackwardFunctions<T>.CumProdBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMax<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: true),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.GreaterThan(a, b) ? a : b;
            },
            "TensorCumMax", BackwardFunctions<T>.CumMaxBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMin<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: false),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.LessThan(a, b) ? a : b;
            },
            "TensorCumMin", BackwardFunctions<T>.CumMinBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogCumSumExp<T>(Tensor<T> tensor, int axis)
    {
        // log-sum-exp cumulative scan: running m = max, running s = sum exp(x-m)
        // result[i] = m + log(s)
        // Numerically stable across wide dynamic ranges.
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var shape = tensor._shape;
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var strides = ComputeRowMajorStrides(shape);
        int axisLen = shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= shape[k];
        int axisStride = strides[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int basePos = outer * axisLen * innerSize;
            for (int inner = 0; inner < innerSize; inner++)
            {
                T runningMax = default!;
                T runningSum = ops.Zero;
                bool first = true;
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = basePos + i * axisStride + inner;
                    T x = src[pos];
                    if (first)
                    {
                        runningMax = x;
                        runningSum = ops.One; // exp(x - x) = 1
                        first = false;
                    }
                    else if (ops.GreaterThan(x, runningMax))
                    {
                        // scale existing sum by exp(old_max - new_max)
                        var scale = ops.Exp(ops.Subtract(runningMax, x));
                        runningSum = ops.Add(ops.Multiply(runningSum, scale), ops.One);
                        runningMax = x;
                    }
                    else
                    {
                        runningSum = ops.Add(runningSum, ops.Exp(ops.Subtract(x, runningMax)));
                    }
                    dst[pos] = ops.Add(runningMax, ops.Log(runningSum));
                }
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorLogCumSumExp", result, tensor,
            BackwardFunctions<T>.LogCumSumExpBackward,
            savedState: new object[] { axis });
        return result;
    }

    // ==================================================================
    // Comparison / predicate ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException($"IsClose requires matching shapes");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var result = new Bit[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            bool aNan = ops.IsNaN(av[i]);
            bool bNan = ops.IsNaN(bv[i]);
            if (aNan || bNan)
            {
                result[i] = (equalNan && aNan && bNan) ? Bit.True : Bit.False;
                continue;
            }
            var diff = ops.Abs(ops.Subtract(av[i], bv[i]));
            var tol = ops.Add(atol, ops.Multiply(rtol, ops.Abs(bv[i])));
            result[i] = ops.LessThanOrEquals(diff, tol) ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, a._shape);
    }

    /// <inheritdoc/>
    public virtual bool TensorAllClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        var mask = TensorIsClose(a, b, rtol, atol, equalNan);
        var span = mask.AsSpan();
        for (int i = 0; i < span.Length; i++)
            if (!(bool)span[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsIn<T>(Tensor<T> elements, Tensor<T> testElements, bool invert = false)
    {
        if (elements == null) throw new ArgumentNullException(nameof(elements));
        if (testElements == null) throw new ArgumentNullException(nameof(testElements));

        var set = new System.Collections.Generic.HashSet<T>(testElements.AsSpan().ToArray());
        if (!elements.IsContiguous) elements = elements.Contiguous();
        var src = elements.AsSpan();
        var result = new Bit[elements.Length];
        for (int i = 0; i < src.Length; i++)
        {
            bool contained = set.Contains(src[i]);
            result[i] = (contained ^ invert) ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, elements._shape);
    }

    // ==================================================================
    // Clamp family
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClampMin<T>(Tensor<T> tensor, T min)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.LessThan(src[i], min) ? min : src[i];

        // Box min as object; T can be any numeric here so a direct cast is
        // safe. The backward unboxes via savedState[0].
        var minBox = (object?)min ?? throw new InvalidOperationException(
            "Clamp min must not be null");
        DifferentiableOps.RecordUnary(
            "TensorClampMin", result, tensor,
            BackwardFunctions<T>.ClampMinBackward,
            savedState: new[] { minBox });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClampMax<T>(Tensor<T> tensor, T max)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.GreaterThan(src[i], max) ? max : src[i];

        var maxBox = (object?)max ?? throw new InvalidOperationException(
            "Clamp max must not be null");
        DifferentiableOps.RecordUnary(
            "TensorClampMax", result, tensor,
            BackwardFunctions<T>.ClampMaxBackward,
            savedState: new[] { maxBox });
        return result;
    }

    /// <inheritdoc/>
    public virtual (T Min, T Max) TensorAminmax<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Aminmax requires a non-empty tensor");
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        T min = src[0], max = src[0];
        for (int i = 1; i < src.Length; i++)
        {
            if (ops.LessThan(src[i], min)) min = src[i];
            if (ops.GreaterThan(src[i], max)) max = src[i];
        }
        return (min, max);
    }

    // ==================================================================
    // Element-wise binary math
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHypot<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Sqrt(ops.Add(ops.Multiply(ax, ax), ops.Multiply(bx, bx)));
        }, "TensorHypot");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCopysign<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            var sign = ops.SignOrZero(bx);
            var mag = ops.Abs(ax);
            // Treat sign==0 (zero b) as positive, matching IEEE copysign on +0.
            return ops.LessThan(sign, ops.Zero) ? ops.Negate(mag) : mag;
        }, "TensorCopysign");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFmod<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            // IEEE: result has same sign as dividend (a).
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = TruncateTowardZero(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorFmod");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRemainder<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            // Python-style: result has same sign as divisor (b).
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = ops.Floor(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorRemainder");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFloatPower<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) =>
            MathHelper.GetNumericOperations<T>().Power(ax, bx), "TensorFloatPower");

    // ==================================================================
    // Special math
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErfc<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Subtract(ops.One, MathHelper.Erf(x));
        }, "TensorErfc");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlogy<T>(Tensor<T> x, Tensor<T> y)
        => ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            // x * log(y) with the convention 0 * log(0) = 0.
            return ops.Equals(xv, ops.Zero) ? ops.Zero : ops.Multiply(xv, ops.Log(yv));
        }, "TensorXlogy");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlog1py<T>(Tensor<T> x, Tensor<T> y)
        => ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            // x * log(1 + y) with 0 * log(...) = 0.
            return ops.Equals(xv, ops.Zero)
                ? ops.Zero
                : ops.Multiply(xv, ops.Log(ops.Add(ops.One, yv)));
        }, "TensorXlog1py");

    // ==================================================================
    // Helpers
    // ==================================================================

    private static int[] ComputeRowMajorStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    private Tensor<T> CumulativeAlongAxis<T>(
        Tensor<T> tensor,
        int axis,
        T initial,
        Func<T, T, T> combine,
        string opName,
        BackwardFunction<T> backward)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var shape = tensor._shape;
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var strides = ComputeRowMajorStrides(shape);
        int axisLen = shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= shape[k];
        int axisStride = strides[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int basePos = outer * axisLen * innerSize;
            for (int inner = 0; inner < innerSize; inner++)
            {
                T acc = initial;
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = basePos + i * axisStride + inner;
                    acc = (i == 0) ? src[pos] : combine(acc, src[pos]);
                    dst[pos] = acc;
                }
            }
        }

        DifferentiableOps.RecordUnary(
            opName, result, tensor,
            backward,
            savedState: new object[] { axis });
        return result;
    }

    private static T CumulativeInitial<T>(bool max)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // Not actually used — cumulative fns handle the first element specially.
        return ops.Zero;
    }

    private static Tensor<T> ElementwiseUnary<T>(
        Tensor<T> tensor, Func<T, T> f, string opName)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = f(src[i]);
        return result;
    }

    private static Tensor<T> ElementwiseBinary<T>(
        Tensor<T> a, Tensor<T> b, Func<T, T, T> f, string opName)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException(
                $"{opName}: shape mismatch [{string.Join(", ", a._shape)}] vs [{string.Join(", ", b._shape)}]");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(a._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < av.Length; i++) dst[i] = f(av[i], bv[i]);
        return result;
    }

    private static T TruncateTowardZero<T>(T value)
    {
        // Truncate toward zero: floor for positive, ceil for negative.
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.LessThan(value, ops.Zero) ? ops.Ceiling(value) : ops.Floor(value);
    }
}
