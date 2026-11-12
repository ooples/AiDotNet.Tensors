using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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

    /// <summary>Compensated float32 sum.</summary>
    private static float CompensatedSum(float[] values)
    {
        float sum = 0f, compensation = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            float y = values[i] - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;
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
            compensation = (t - sum) - y;
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
        var x = WidenToFloat(a.GetFlattenedData(), length);
        var y = WidenToFloat(b.GetFlattenedData(), length);
        var terms = new float[length];
        for (int i = 0; i < length; i++) terms[i] = term(x[i], y[i]);
        return NarrowFromFloat<T>(CompensatedSum(terms) / length);
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
            compensation[o] = (t - acc[o]) - y;
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
    /// <paramref name="mean"/> and <paramref name="gradOutput"/> are indexed by reduced slot.
    /// </summary>
    internal static T[] LowPrecisionVarianceBackward<T>(T[] input, T[] mean, T[] gradOutput, int[] inputShape, int[] normalizedAxes)
    {
        var strideForDim = ReducedSlotStrides(inputShape, normalizedAxes, out int reduceCount);
        int slots = 1;
        foreach (int dim in LowPrecisionReducedShape(inputShape, normalizedAxes, keepDims: true)) slots *= dim;
        int inputSize = 1;
        foreach (int dim in inputShape) inputSize *= dim;
        var x = WidenToFloat(input, inputSize);
        var m = WidenToFloat(mean, slots);
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
