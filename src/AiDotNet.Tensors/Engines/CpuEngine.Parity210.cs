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
    // Indexing family
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexAdd<T>(
        Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (indices.Rank != 1) throw new ArgumentException("indices must be 1-D");
        if (source._shape[axis] != indices._shape[0])
            throw new ArgumentException(
                $"source.shape[{axis}]={source._shape[axis]} must match indices.length={indices._shape[0]}");
        for (int k = 0; k < rank; k++)
        {
            if (k != axis && source._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"source.shape[{k}]={source._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();
        var idxData = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int dstAxis = tensor._shape[axis];
        int srcAxis = source._shape[axis];
        int dstAxisStride = innerSize;
        int srcAxisStride = innerSize;

        for (int outer = 0; outer < outerSize; outer++)
        {
            int dstOuter = outer * dstAxis * dstAxisStride;
            int srcOuter = outer * srcAxis * srcAxisStride;
            for (int i = 0; i < idxData.Length; i++)
            {
                int target = idxData[i];
                if (target < 0 || target >= dstAxis)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {dstAxis}");
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstPos = dstOuter + target * dstAxisStride + inner;
                    int srcPos = srcOuter + i * srcAxisStride + inner;
                    dst[dstPos] = ops.Add(dst[dstPos], srcData[srcPos]);
                }
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexFill<T>(
        Tensor<T> tensor, int axis, Tensor<int> indices, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (indices.Rank != 1) throw new ArgumentException("indices must be 1-D");

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var idxData = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int axisSize = tensor._shape[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int outerBase = outer * axisSize * innerSize;
            for (int i = 0; i < idxData.Length; i++)
            {
                int target = idxData[i];
                if (target < 0 || target >= axisSize)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {axisSize}");
                for (int inner = 0; inner < innerSize; inner++)
                    dst[outerBase + target * innerSize + inner] = value;
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMaskedScatter<T>(
        Tensor<T> tensor, Tensor<Bit> mask, Tensor<T> source)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (!tensor._shape.SequenceEqual(mask._shape))
            throw new ArgumentException(
                $"mask shape [{string.Join(", ", mask._shape)}] must match tensor shape [{string.Join(", ", tensor._shape)}]");

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var maskData = mask.AsSpan();
        var srcData = source.AsSpan();

        int sourceCursor = 0;
        for (int i = 0; i < maskData.Length; i++)
        {
            if ((bool)maskData[i])
            {
                if (sourceCursor >= srcData.Length)
                    throw new ArgumentException("source has fewer elements than mask-true count");
                dst[i] = srcData[sourceCursor++];
            }
        }
        return result;
    }

    // ==================================================================
    // Sort / order statistics
    // ==================================================================

    /// <inheritdoc/>
    public virtual (Tensor<T> Values, Tensor<int> Indices) TensorSort<T>(
        Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var numOps = MathHelper.GetNumericOperations<T>();
        var valuesOut = AutoTensorCache.RentOrAllocate<T>(input._shape);
        var indicesOut = new Tensor<int>(input._shape);

        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var valDst = valuesOut.AsWritableSpan();
        var idxDst = indicesOut.AsWritableSpan();

        int axisSize = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];

        var buffer = new (T value, int index)[axisSize];
        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                int baseIdx = outer * axisSize * innerSize + inner;
                for (int a = 0; a < axisSize; a++)
                    buffer[a] = (src[baseIdx + a * innerSize], a);

                Array.Sort(buffer, descending
                    ? (x, y) => numOps.Compare(y.value, x.value)
                    : (Comparison<(T value, int index)>)((x, y) => numOps.Compare(x.value, y.value)));

                for (int a = 0; a < axisSize; a++)
                {
                    int pos = baseIdx + a * innerSize;
                    valDst[pos] = buffer[a].value;
                    idxDst[pos] = buffer[a].index;
                }
            }
        return (valuesOut, indicesOut);
    }

    /// <inheritdoc/>
    public virtual (T Value, int Index) TensorKthvalue<T>(Tensor<T> input, int k)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (k < 1 || k > input.Length) throw new ArgumentOutOfRangeException(nameof(k));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var pairs = new (T value, int index)[src.Length];
        for (int i = 0; i < src.Length; i++) pairs[i] = (src[i], i);
        Array.Sort(pairs, (x, y) => numOps.Compare(x.value, y.value));
        return pairs[k - 1];
    }

    /// <inheritdoc/>
    public virtual T TensorMedian<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("Median requires a non-empty tensor");
        // PyTorch returns the lower median for even-length; keep the same behaviour.
        int k = (input.Length + 1) / 2;
        var (value, _) = TensorKthvalue(input, k);
        return value;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUnique<T>(Tensor<T> input, bool sorted = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();

        // HashSet<T> preserves first-insertion order in .NET; we sort afterwards if requested.
        var seen = new System.Collections.Generic.HashSet<T>();
        var order = new System.Collections.Generic.List<T>();
        for (int i = 0; i < src.Length; i++)
        {
            if (seen.Add(src[i])) order.Add(src[i]);
        }

        if (sorted)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            order.Sort((a, b) => numOps.Compare(a, b));
        }

        var outArr = order.ToArray();
        return new Tensor<T>(outArr, new[] { outArr.Length });
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorSearchSorted<T>(
        Tensor<T> sortedSequence, Tensor<T> values, bool right = false)
    {
        if (sortedSequence == null) throw new ArgumentNullException(nameof(sortedSequence));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (sortedSequence.Rank != 1)
            throw new ArgumentException("SearchSorted expects a 1-D sorted sequence");

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!sortedSequence.IsContiguous) sortedSequence = sortedSequence.Contiguous();
        if (!values.IsContiguous) values = values.Contiguous();
        var seq = sortedSequence.AsSpan();
        var vs = values.AsSpan();
        var result = new Tensor<int>(values._shape);
        var dst = result.AsWritableSpan();

        for (int i = 0; i < vs.Length; i++)
        {
            // Branchless-ready binary search; returns insertion index.
            int lo = 0, hi = seq.Length;
            var v = vs[i];
            while (lo < hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                // right=false → lower bound: split at seq[mid] >= v (v <= seq[mid]).
                // right=true  → upper bound: split at seq[mid] >  v (v <  seq[mid]).
                bool beforeMid = right
                    ? numOps.LessThan(v, seq[mid])
                    : numOps.LessThanOrEquals(v, seq[mid]);
                if (beforeMid) hi = mid; else lo = mid + 1;
            }
            dst[i] = lo;
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorHistogram<T>(Tensor<T> input, int bins, T min, T max)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bins < 1) throw new ArgumentOutOfRangeException(nameof(bins));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (numOps.GreaterThanOrEquals(min, max))
            throw new ArgumentException("Histogram requires min < max");
        if (!input.IsContiguous) input = input.Contiguous();

        var result = new Tensor<int>(new[] { bins });
        var dst = result.AsWritableSpan();
        var src = input.AsSpan();

        var width = numOps.Divide(numOps.Subtract(max, min), numOps.FromDouble(bins));
        for (int i = 0; i < src.Length; i++)
        {
            var v = src[i];
            if (numOps.LessThan(v, min) || numOps.GreaterThan(v, max)) continue;
            // Bin index: floor((v - min) / width). Clamp the last-bin edge so
            // the upper boundary maps into the final bin.
            int idx;
            if (numOps.Equals(v, max)) idx = bins - 1;
            else
            {
                var f = numOps.Divide(numOps.Subtract(v, min), width);
                idx = numOps.ToInt32(numOps.Floor(f));
                if (idx >= bins) idx = bins - 1;
                if (idx < 0) idx = 0;
            }
            dst[idx]++;
        }
        return result;
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
