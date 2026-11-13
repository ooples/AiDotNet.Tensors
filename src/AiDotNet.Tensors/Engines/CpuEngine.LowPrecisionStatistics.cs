using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Float32 accumulation for the mean-type statistics of Half and BFloat16 tensors.
/// </summary>
/// <remarks>
/// <para>Half's largest finite value is 65,504, so an element count above that becomes +Inf when converted to T,
/// and a running sum held in Half or BFloat16 overflows (Half) or stops growing once the sum's spacing exceeds the
/// addend (256 + 1 == 256 in BFloat16). PyTorch accumulates both types in their accumulation type float32
/// (<c>at::acc_type</c> / <c>at::opmath_type</c>), divides by the count in float32 and rounds to the storage type
/// once; these helpers do the same. The count is never converted to T.</para>
/// <para>The data is widened to float32 once with a loop specialised for the element type, all arithmetic runs on
/// float spans, and results are narrowed once. Sums are compensated (Kahan) so a float32 accumulator over hundreds of
/// thousands of elements keeps its error far below one unit in the last place of the 8- or 11-bit-mantissa result,
/// as PyTorch's cascade sum does. Float and double never take these paths, so their results are unchanged.</para>
/// </remarks>
public partial class CpuEngine
{
    /// <summary>True for the 16-bit floating-point element types whose accumulation type is float32.</summary>
    internal static bool IsLowPrecisionFloat<T>() => typeof(T) == typeof(Half) || typeof(T) == typeof(BFloat16);

    /// <summary>Widens the first <paramref name="length"/> Half or BFloat16 elements to float32.</summary>
    internal static float[] WidenToFloat<T>(T[] data, int length)
    {
        var result = new float[length];
        if (typeof(T) == typeof(Half))
        {
            var half = Unsafe.As<T[], Half[]>(ref data);
            for (int i = 0; i < length; i++) result[i] = (float)half[i];
        }
        else if (typeof(T) == typeof(BFloat16))
        {
            var bf16 = Unsafe.As<T[], BFloat16[]>(ref data);
            for (int i = 0; i < length; i++) result[i] = BFloat16.ToFloat(bf16[i]);
        }
        else
        {
            throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
        }
        return result;
    }

    /// <summary>Widens element <paramref name="index"/> of a Half or BFloat16 array to float32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float WidenElement<T>(T[] data, int index)
    {
        if (typeof(T) == typeof(Half)) return (float)Unsafe.As<T[], Half[]>(ref data)[index];
        if (typeof(T) == typeof(BFloat16)) return BFloat16.ToFloat(Unsafe.As<T[], BFloat16[]>(ref data)[index]);
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Rounds float32 values to Half or BFloat16 (round to nearest even), once each.</summary>
    internal static T[] NarrowFromFloat<T>(float[] values)
    {
        if (typeof(T) == typeof(Half))
        {
            var half = new Half[values.Length];
            for (int i = 0; i < values.Length; i++) half[i] = (Half)values[i];
            return Unsafe.As<Half[], T[]>(ref half);
        }
        if (typeof(T) == typeof(BFloat16))
        {
            var bf16 = new BFloat16[values.Length];
            for (int i = 0; i < values.Length; i++) bf16[i] = BFloat16.FromFloat(values[i]);
            return Unsafe.As<BFloat16[], T[]>(ref bf16);
        }
        throw new NotSupportedException($"{typeof(T).Name} is not a 16-bit floating-point type.");
    }

    /// <summary>Rounds one float32 value to Half or BFloat16.</summary>
    internal static T NarrowFromFloat<T>(float value) => NarrowFromFloat<T>(new[] { value })[0];

    // The float32 statistics a low-precision norm forward computed, keyed by the narrowed mean and variance tensors it
    // returned. Narrowing loses what the backward needs: a Half variance above 65,504 is +Inf (so 1/sqrt(var + eps) is
    // 0 and every gradient vanishes), and a rounded mean skews the centring. A backward handed those same tensors
    // reads the exact values back; any other statistic, or one written since, is widened as given.
    private static readonly ConditionalWeakTable<object, float[]> s_float32Statistics = new ConditionalWeakTable<object, float[]>();

    /// <summary>Wraps narrowed statistics in a tensor and remembers their float32 values for the backward.</summary>
    internal static Tensor<T> RentLowPrecisionStatistic<T>(int[] shape, float[] exact)
    {
        var statistic = TensorAllocator.Rent<T>(shape, NarrowFromFloat<T>(exact));
        // Replace, not Add: a statistic a compiled replay returned to the pool can come back here as the same object.
        lock (s_float32Statistics)
        {
            s_float32Statistics.Remove(statistic);
            s_float32Statistics.Add(statistic, exact);
        }
        return statistic;
    }

    /// <summary>
    /// Moves the float32 values remembered for <paramref name="fresh"/> onto <paramref name="refreshed"/>, the statistic a
    /// compiled replay has just copied <paramref name="fresh"/>'s narrowed values into. Without this the refreshed tensor
    /// keeps the trace's float32 values, and a step whose new statistics narrow to the same Half values (two variances
    /// both +Inf, say) would hand the backward the trace's statistics instead of its own.
    /// </summary>
    internal static void CarryLowPrecisionStatistic<T>(Tensor<T> fresh, Tensor<T> refreshed)
    {
        lock (s_float32Statistics)
        {
            s_float32Statistics.Remove(refreshed);
            if (s_float32Statistics.TryGetValue(fresh, out var exact))
                s_float32Statistics.Add(refreshed, (float[])exact.Clone());
        }
    }

    /// <summary>
    /// The float32 values of a statistic: the forward's exact values when <paramref name="statistic"/> is the tensor a
    /// low-precision norm forward returned and each value still rounds to its current element, else its elements widened.
    /// </summary>
    internal static float[] LowPrecisionStatisticValues<T>(Tensor<T> statistic, T[] data, int length)
    {
        if (s_float32Statistics.TryGetValue(statistic, out var exact) && exact.Length == length)
        {
            var narrowed = NarrowFromFloat<T>(exact);
            var comparer = EqualityComparer<T>.Default;
            bool current = true;
            for (int i = 0; i < length && current; i++) current = comparer.Equals(narrowed[i], data[i]);
            if (current) return exact;
        }
        return WidenToFloat(data, length);
    }

    /// <summary>Compensated float32 sum.</summary>
    private static float CompensatedSum(float[] values)
    {
        float sum = 0f, compensation = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            float y = values[i] - compensation;
            float t = sum + y;
            compensation = float.IsInfinity(t) ? 0f : (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    /// <summary>Mean of the first <paramref name="length"/> elements: float32 sum / count, rounded to T once.</summary>
    internal static T LowPrecisionMean<T>(T[] data, int length)
        => NarrowFromFloat<T>(CompensatedSum(WidenToFloat(data, length)) / length);

    /// <summary>Two-pass float32 mean and sum of squared deviations of the first <paramref name="length"/> elements.</summary>
    internal static void LowPrecisionMoments<T>(T[] data, int length, out float mean, out float sumSquaredDeviation)
    {
        var x = WidenToFloat(data, length);
        mean = CompensatedSum(x) / length;
        float sum = 0f, compensation = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            float d = x[i] - mean;
            float y = d * d - compensation;
            float t = sum + y;
            compensation = float.IsInfinity(t) ? 0f : (t - sum) - y;
            sum = t;
        }
        sumSquaredDeviation = sum;
    }

    /// <summary>
    /// Mean over all elements of <c>term(a[i], b[i])</c>, with the terms evaluated, summed (compensated) and divided by
    /// the element count in float32, then rounded to T once. Used by the mean-reduction losses.
    /// </summary>
    internal static T LowPrecisionPairMean<T>(Tensor<T> a, Tensor<T> b, Func<float, float, float> term)
    {
        int length = a.Length;
        var x = a.GetFlattenedData();
        var y = b.GetFlattenedData();
        // One streaming pass: each pair is widened, its term added to a compensated float32 sum, and nothing of the
        // input's size is allocated in float32.
        float sum = 0f, compensation = 0f;
        for (int i = 0; i < length; i++)
        {
            float v = term(WidenElement(x, i), WidenElement(y, i)) - compensation;
            float t = sum + v;
            compensation = float.IsInfinity(t) ? 0f : (t - sum) - v;
            sum = t;
        }
        return NarrowFromFloat<T>(sum / length);
    }

    /// <summary>
    /// BatchNorm's [batch, channels, spatial] view of a rank-2 [N, F], rank-3 [C, H, W] or rank-4 [N, C, H, W] input,
    /// matching the layouts the generic 2D, 3D and 4D paths normalise.
    /// </summary>
    internal static void GetLowPrecisionBatchNormDims(int[] shape, out int batch, out int channels, out int spatial)
    {
        if (shape.Length == 4) { batch = shape[0]; channels = shape[1]; spatial = shape[2] * shape[3]; }
        else if (shape.Length == 3) { batch = 1; channels = shape[0]; spatial = shape[1] * shape[2]; }
        else { batch = shape[0]; channels = shape[1]; spatial = 1; }
    }

    /// <summary>The shape left after reducing <paramref name="normalizedAxes"/>, with or without kept size-1 dims.</summary>
    internal static int[] LowPrecisionReducedShape(int[] shape, int[] normalizedAxes, bool keepDims)
    {
        var isReduced = new bool[shape.Length];
        foreach (int a in normalizedAxes) isReduced[a] = true;
        var result = new List<int>(shape.Length);
        for (int d = 0; d < shape.Length; d++)
        {
            if (!isReduced[d]) result.Add(shape[d]);
            else if (keepDims) result.Add(1);
        }
        return result.ToArray();
    }

    // Shape with every reduced dim kept as 1 (output slots follow its row-major order, same as without kept dims),
    // the element count reduced into each slot, and the odometer stride of each input dim into that shape.
    private static int[] ReducedSlotStrides(int[] shape, int[] normalizedAxes, out int reduceCount)
    {
        var keptShape = LowPrecisionReducedShape(shape, normalizedAxes, keepDims: true);
        reduceCount = 1;
        var isReduced = new bool[shape.Length];
        foreach (int a in normalizedAxes) isReduced[a] = true;
        for (int d = 0; d < shape.Length; d++) if (isReduced[d]) reduceCount *= shape[d];
        return BuildReducedOutStrideForDim(shape, keptShape, normalizedAxes);
    }

    // One odometer pass: acc[slot] += x (centre == null) or (x - centre[slot])^2, Kahan-compensated per slot.
    private static void AccumulateSlots(float[] x, int[] shape, int[] strideForDim, float[] acc, float[]? centre)
    {
        int rank = shape.Length;
        var compensation = new float[acc.Length];
        var coord = new int[rank];
        int o = 0;
        for (int i = 0; i < x.Length; i++)
        {
            float term;
            if (centre is null) term = x[i];
            else { float d = x[i] - centre[o]; term = d * d; }
            float y = term - compensation[o];
            float t = acc[o] + y;
            compensation[o] = float.IsInfinity(t) ? 0f : (t - acc[o]) - y;
            acc[o] = t;
            for (int d = rank - 1; d >= 0; d--)
            {
                coord[d]++; o += strideForDim[d];
                if (coord[d] < shape[d]) break;
                coord[d] = 0; o -= strideForDim[d] * shape[d];
            }
        }
    }

    /// <summary>
    /// Mean (or variance with the given correction, optionally square-rooted) over <paramref name="normalizedAxes"/> of
    /// a logical row-major array, in the reduced shape's row-major order, computed in float32 and rounded to T once.
    /// Variance divides the float32 sum of squared deviations by <c>count - correction</c> in float32.
    /// </summary>
    internal static T[] LowPrecisionAxisStatistic<T>(T[] data, int[] shape, int[] normalizedAxes, int outputSize,
        bool variance, int correction, bool squareRoot)
    {
        int length = 1;
        foreach (int dim in shape) length *= dim;
        var x = WidenToFloat(data, length);
        var strideForDim = ReducedSlotStrides(shape, normalizedAxes, out int reduceCount);
        var mean = new float[outputSize];
        AccumulateSlots(x, shape, strideForDim, mean, null);
        for (int o = 0; o < outputSize; o++) mean[o] /= reduceCount;
        if (!variance) return NarrowFromFloat<T>(mean);

        var sumSq = new float[outputSize];
        AccumulateSlots(x, shape, strideForDim, sumSq, mean);
        float denominator = reduceCount - correction;
        for (int o = 0; o < outputSize; o++)
        {
            float v = sumSq[o] / denominator;
            sumSq[o] = squareRoot ? (float)Math.Sqrt(Math.Max(v, 0f)) : v;
        }
        return NarrowFromFloat<T>(sumSq);
    }

    /// <summary>
    /// ReduceMean backward in float32: every input position receives its reduced slot's gradient divided by the
    /// count (1/count is subnormal or zero in Half once the count passes 65,504).
    /// </summary>
    internal static T[] LowPrecisionMeanBackward<T>(T[] gradOutput, int[] inputShape, int[] normalizedAxes)
    {
        var strideForDim = ReducedSlotStrides(inputShape, normalizedAxes, out int reduceCount);
        int slots = 1;
        foreach (int dim in LowPrecisionReducedShape(inputShape, normalizedAxes, keepDims: true)) slots *= dim;
        var g = WidenToFloat(gradOutput, slots);
        int inputSize = 1;
        foreach (int dim in inputShape) inputSize *= dim;
        var result = new float[inputSize];
        int rank = inputShape.Length;
        var coord = new int[rank];
        int o = 0;
        for (int i = 0; i < inputSize; i++)
        {
            result[i] = g[o] / reduceCount;
            for (int d = rank - 1; d >= 0; d--)
            {
                coord[d]++; o += strideForDim[d];
                if (coord[d] < inputShape[d]) break;
                coord[d] = 0; o -= strideForDim[d] * inputShape[d];
            }
        }
        return NarrowFromFloat<T>(result);
    }

    /// <summary>
    /// ReduceVariance (population) backward in float32: <c>2 (x - mean) gradOutput / count</c>, rounded once.
    /// <paramref name="gradOutput"/> is indexed by reduced slot. The mean is recomputed per slot in float32 from the
    /// input, as the forward computed it: the saved T-typed mean is rounded (BFloat16 [1, 1.0078125] has float32 mean
    /// 1.00390625 but a saved mean of 1), and centring on it skews every gradient.
    /// </summary>
    internal static T[] LowPrecisionVarianceBackward<T>(T[] input, T[] gradOutput, int[] inputShape, int[] normalizedAxes)
    {
        var strideForDim = ReducedSlotStrides(inputShape, normalizedAxes, out int reduceCount);
        int slots = 1;
        foreach (int dim in LowPrecisionReducedShape(inputShape, normalizedAxes, keepDims: true)) slots *= dim;
        int inputSize = 1;
        foreach (int dim in inputShape) inputSize *= dim;
        var x = WidenToFloat(input, inputSize);
        var m = new float[slots];
        AccumulateSlots(x, inputShape, strideForDim, m, null);
        for (int s = 0; s < slots; s++) m[s] /= reduceCount;
        var g = WidenToFloat(gradOutput, slots);
        var result = new float[inputSize];
        int rank = inputShape.Length;
        var coord = new int[rank];
        int o = 0;
        for (int i = 0; i < inputSize; i++)
        {
            result[i] = 2f * (x[i] - m[o]) * g[o] / reduceCount;
            for (int d = rank - 1; d >= 0; d--)
            {
                coord[d]++; o += strideForDim[d];
                if (coord[d] < inputShape[d]) break;
                coord[d] = 0; o -= strideForDim[d] * inputShape[d];
            }
        }
        return NarrowFromFloat<T>(result);
    }
}
