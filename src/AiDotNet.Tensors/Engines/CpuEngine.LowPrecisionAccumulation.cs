using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Engines;

/// <summary>Which normalised exponential a low-precision softmax kernel produces.</summary>
internal enum LowPrecisionSoftmaxForm
{
    /// <summary><c>exp(x - max) / sum(exp(x - max))</c>.</summary>
    Softmax,

    /// <summary><c>(x - max) - log(sum(exp(x - max)))</c>.</summary>
    LogSoftmax,
}

/// <summary>
/// Float32 accumulation for the sums, dot products, matmuls, cumulative sums, norms and softmax denominators of Half
/// and BFloat16 tensors.
/// </summary>
/// <remarks>
/// <para>Accumulating in the element type stalls: Half has an 11-bit significand, so once a running sum reaches
/// 2048 adding 1 rounds back to 2048 (a Half sum of 4096 ones is 2048), and BFloat16 stalls at 256. Every product
/// and partial sum also rounds, so a long dot product keeps only a few correct bits. PyTorch accumulates these
/// reductions in float32 on CPU and CUDA (<c>at::opmath_type&lt;Half&gt;</c>, cuBLAS fp32 compute) and NumPy sums
/// float16 pairwise in float32; both round to the storage type once at the end. These kernels do the same: elements
/// are widened once (Half through the AVX2 decode in <see cref="SimdKernels.ConvertToSingle"/>), all arithmetic runs
/// in float32 and each output is narrowed once.</para>
/// <para>Long sums are taken in float32 blocks of <see cref="AccumulationBlock"/> elements with the SIMD float32
/// kernels and the block totals are added in double, so the float32 rounding error stays that of a 4096-element sum
/// however long the input is (PyTorch's cascade sum has the same bound).</para>
/// </remarks>
public partial class CpuEngine
{
    /// <summary>Elements widened and summed per float32 block before the block total is added in double.</summary>
    private const int AccumulationBlock = 4096;

    /// <summary>Widens one Half or BFloat16 value to float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float LowPrecisionToFloat<T>(T value)
    {
        if (typeof(T) == typeof(Half)) return (float)Unsafe.As<T, Half>(ref value);
        if (typeof(T) == typeof(BFloat16)) return BFloat16.ToFloat(Unsafe.As<T, BFloat16>(ref value));
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Rounds one float32 value to Half or BFloat16 (round to nearest even).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T LowPrecisionFromFloat<T>(float value)
    {
        if (typeof(T) == typeof(Half))
        {
            var h = (Half)value;
            return Unsafe.As<Half, T>(ref h);
        }
        if (typeof(T) == typeof(BFloat16))
        {
            var b = BFloat16.FromFloat(value);
            return Unsafe.As<BFloat16, T>(ref b);
        }
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Widens <paramref name="length"/> elements of <paramref name="source"/> from <paramref name="sourceOffset"/>.</summary>
    internal static void WidenInto<T>(T[] source, int sourceOffset, float[] destination, int destinationOffset, int length)
    {
        if (typeof(T) == typeof(Half))
        {
            var half = Unsafe.As<T[], Half[]>(ref source);
            SimdKernels.ConvertToSingle(new ReadOnlySpan<Half>(half, sourceOffset, length),
                new Span<float>(destination, destinationOffset, length));
            return;
        }
        if (typeof(T) == typeof(BFloat16))
        {
            var bf16 = Unsafe.As<T[], BFloat16[]>(ref source);
            for (int i = 0; i < length; i++) destination[destinationOffset + i] = BFloat16.ToFloat(bf16[sourceOffset + i]);
            return;
        }
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Rounds <paramref name="length"/> float32 values once each into <paramref name="destination"/>.</summary>
    internal static void NarrowInto<T>(float[] source, int sourceOffset, T[] destination, int destinationOffset, int length)
    {
        if (typeof(T) == typeof(Half))
        {
            var half = Unsafe.As<T[], Half[]>(ref destination);
            SimdKernels.ConvertToHalf(new ReadOnlySpan<float>(source, sourceOffset, length),
                new Span<Half>(half, destinationOffset, length));
            return;
        }
        if (typeof(T) == typeof(BFloat16))
        {
            var bf16 = Unsafe.As<T[], BFloat16[]>(ref destination);
            for (int i = 0; i < length; i++) bf16[destinationOffset + i] = BFloat16.FromFloat(source[sourceOffset + i]);
            return;
        }
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Widens a whole contiguous run into a new float32 array.</summary>
    private static float[] WidenRun<T>(T[] source, int offset, int length)
    {
        var result = new float[length];
        WidenInto(source, offset, result, 0, length);
        return result;
    }

    /// <summary>Sum of <paramref name="length"/> elements from <paramref name="offset"/>, accumulated in float32 blocks.</summary>
    internal static float LowPrecisionSum<T>(T[] data, int offset, int length)
    {
        var block = new float[Math.Min(length, AccumulationBlock)];
        double total = 0;
        for (int start = 0; start < length; start += AccumulationBlock)
        {
            int count = Math.Min(AccumulationBlock, length - start);
            WidenInto(data, offset + start, block, 0, count);
            total += SimdKernels.Sum(new ReadOnlySpan<float>(block, 0, count));
        }
        return (float)total;
    }

    /// <summary>
    /// Dot product of <paramref name="length"/> elements of two runs, products and sums in float32 blocks. Passing the
    /// same array and offset twice gives the sum of squares.
    /// </summary>
    internal static float LowPrecisionDot<T>(T[] a, int aOffset, T[] b, int bOffset, int length)
    {
        int blockLength = Math.Min(length, AccumulationBlock);
        var aBlock = new float[blockLength];
        var bBlock = ReferenceEquals(a, b) && aOffset == bOffset ? aBlock : new float[blockLength];
        double total = 0;
        for (int start = 0; start < length; start += AccumulationBlock)
        {
            int count = Math.Min(AccumulationBlock, length - start);
            WidenInto(a, aOffset + start, aBlock, 0, count);
            if (!ReferenceEquals(aBlock, bBlock)) WidenInto(b, bOffset + start, bBlock, 0, count);
            total += SimdKernels.DotProduct(new ReadOnlySpan<float>(aBlock, 0, count), new ReadOnlySpan<float>(bBlock, 0, count));
        }
        return (float)total;
    }

    /// <summary>
    /// Sum (or sum of squares, square-rooted when <paramref name="squareRoot"/> is set: the L2 norm) over
    /// <paramref name="normalizedAxes"/> of a logical row-major array, in the reduced shape's row-major order,
    /// accumulated in float32 (compensated per slot) and rounded to T once.
    /// </summary>
    internal static T[] LowPrecisionAxisSum<T>(T[] data, int[] shape, int[] normalizedAxes, int outputSize,
        bool squares, bool squareRoot)
    {
        int length = 1;
        foreach (int dim in shape) length *= dim;
        var x = WidenRun(data, 0, length);
        var strideForDim = ReducedSlotStrides(shape, normalizedAxes, out _);
        var acc = new float[outputSize];
        AccumulateSlots(x, shape, strideForDim, acc, squares ? new float[outputSize] : null);
        if (squareRoot)
        {
            for (int o = 0; o < outputSize; o++) acc[o] = (float)Math.Sqrt(acc[o]);
        }
        var result = new T[outputSize];
        NarrowInto(acc, 0, result, 0, outputSize);
        return result;
    }

    // One odometer pass over a logical row-major array: visit(i, slot) for every element, slot being its reduced output.
    private static void ForEachSlot(int[] shape, int[] strideForDim, int length, Action<int, int> visit)
    {
        int rank = shape.Length;
        var coord = new int[rank];
        int o = 0;
        for (int i = 0; i < length; i++)
        {
            visit(i, o);
            for (int d = rank - 1; d >= 0; d--)
            {
                coord[d]++; o += strideForDim[d];
                if (coord[d] < shape[d]) break;
                coord[d] = 0; o -= strideForDim[d] * shape[d];
            }
        }
    }

    /// <summary>
    /// <c>torch.nanmean</c> over <paramref name="normalizedAxes"/> for Half/BFloat16: the non-NaN elements are summed in
    /// float32 (compensated per slot), counted in an int, divided in float32 and rounded to T once. Holding the sum or
    /// the count in Half overflows past 65,504 (70,000 Half ones gave +Inf / +Inf = NaN). A slot with no non-NaN
    /// element is 0 / 0 = NaN, as in PyTorch.
    /// </summary>
    internal static T[] LowPrecisionNanMean<T>(T[] data, int[] shape, int[] normalizedAxes, int outputSize)
    {
        int length = 1;
        foreach (int dim in shape) length *= dim;
        var x = WidenRun(data, 0, length);
        var strideForDim = ReducedSlotStrides(shape, normalizedAxes, out _);
        var sum = new float[outputSize];
        var compensation = new float[outputSize];
        var count = new int[outputSize];
        ForEachSlot(shape, strideForDim, length, (i, o) =>
        {
            float v = x[i];
            if (float.IsNaN(v)) return;
            count[o]++;
            float y = v - compensation[o];
            float t = sum[o] + y;
            compensation[o] = float.IsInfinity(t) ? 0f : (t - sum[o]) - y;
            sum[o] = t;
        });
        for (int o = 0; o < outputSize; o++) sum[o] /= count[o];
        var result = new T[outputSize];
        NarrowInto(sum, 0, result, 0, outputSize);
        return result;
    }

    /// <summary>
    /// The nanmean backward for Half/BFloat16: each non-NaN input receives its slot's upstream gradient divided by the
    /// slot's int count in float32, NaN inputs receive 0, and each gradient is rounded once.
    /// </summary>
    internal static T[] LowPrecisionNanMeanBackward<T>(T[] input, T[] gradOutput, int[] shape, int[] normalizedAxes)
    {
        int length = 1;
        foreach (int dim in shape) length *= dim;
        var x = WidenRun(input, 0, length);
        var strideForDim = ReducedSlotStrides(shape, normalizedAxes, out _);
        int slots = 1;
        foreach (int dim in LowPrecisionReducedShape(shape, normalizedAxes, keepDims: true)) slots *= dim;
        var g = WidenRun(gradOutput, 0, slots);
        var count = new int[slots];
        ForEachSlot(shape, strideForDim, length, (i, o) => { if (!float.IsNaN(x[i])) count[o]++; });
        var result = new float[length];
        ForEachSlot(shape, strideForDim, length, (i, o) => result[i] = float.IsNaN(x[i]) ? 0f : g[o] / count[o]);
        var narrowed = new T[length];
        NarrowInto(result, 0, narrowed, 0, length);
        return narrowed;
    }

    /// <summary>
    /// Softmax or log-softmax along the middle axis of an <c>[outer, axisSize, inner]</c> contiguous layout: the max,
    /// the exponentials, their sum and the normalisation all run in float32, and each output is rounded once.
    /// </summary>
    internal static T[] LowPrecisionSoftmax<T>(T[] data, int outerSize, int axisSize, int innerSize, LowPrecisionSoftmaxForm form)
    {
        int length = outerSize * axisSize * innerSize;
        var x = WidenRun(data, 0, length);
        var y = new float[length];
        CpuParallelSettings.ParallelForOrSerial(0, outerSize * innerSize, (long)length, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;
            int baseIndex = outer * axisSize * innerSize + inner;
            float max = float.NegativeInfinity;
            for (int i = 0; i < axisSize; i++) max = Math.Max(max, x[baseIndex + i * innerSize]);
            // An all -Inf row has no finite max; shifting by 0 keeps exp(-Inf) = 0 instead of exp(NaN).
            if (float.IsNegativeInfinity(max)) max = 0f;
            float sum = 0f;
            for (int i = 0; i < axisSize; i++)
            {
                int at = baseIndex + i * innerSize;
                float e = (float)Math.Exp(x[at] - max);
                y[at] = e;
                sum += e;
            }
            if (form == LowPrecisionSoftmaxForm.Softmax)
            {
                for (int i = 0; i < axisSize; i++) y[baseIndex + i * innerSize] /= sum;
            }
            else
            {
                float logSum = (float)Math.Log(sum);
                for (int i = 0; i < axisSize; i++)
                {
                    int at = baseIndex + i * innerSize;
                    y[at] = x[at] - max - logSum;
                }
            }
        });
        var result = new T[length];
        NarrowInto(y, 0, result, 0, length);
        return result;
    }

    /// <summary>
    /// <c>log(sum(exp(x)))</c> along the middle axis of an <c>[outer, axisSize, inner]</c> contiguous layout, max-shifted
    /// and summed in float32, one output per (outer, inner) pair rounded once.
    /// </summary>
    internal static T[] LowPrecisionLogSumExp<T>(T[] data, int outerSize, int axisSize, int innerSize)
    {
        var x = WidenRun(data, 0, outerSize * axisSize * innerSize);
        var y = new float[outerSize * innerSize];
        CpuParallelSettings.ParallelForOrSerial(0, outerSize * innerSize, (long)x.Length, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;
            int baseIndex = outer * axisSize * innerSize + inner;
            float max = float.NegativeInfinity;
            for (int i = 0; i < axisSize; i++) max = Math.Max(max, x[baseIndex + i * innerSize]);
            if (float.IsInfinity(max))
            {
                y[idx] = max;
                return;
            }
            float sum = 0f;
            for (int i = 0; i < axisSize; i++) sum += (float)Math.Exp(x[baseIndex + i * innerSize] - max);
            y[idx] = max + (float)Math.Log(sum);
        });
        var result = new T[y.Length];
        NarrowInto(y, 0, result, 0, y.Length);
        return result;
    }

    /// <summary>
    /// Inclusive prefix sum along the middle axis of an <c>[outer, axisSize, inner]</c> contiguous layout with a float32
    /// running sum; each prefix is rounded to T once, never fed back into the running sum.
    /// </summary>
    internal static T[] LowPrecisionCumSum<T>(T[] data, int outerSize, int axisSize, int innerSize)
    {
        int length = outerSize * axisSize * innerSize;
        var x = WidenRun(data, 0, length);
        var y = new float[length];
        CpuParallelSettings.ParallelForOrSerial(0, outerSize * innerSize, (long)length, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;
            int baseIndex = outer * axisSize * innerSize + inner;
            float running = 0f;
            for (int i = 0; i < axisSize; i++)
            {
                int at = baseIndex + i * innerSize;
                running += x[at];
                y[at] = running;
            }
        });
        var result = new T[length];
        NarrowInto(y, 0, result, 0, length);
        return result;
    }
}
