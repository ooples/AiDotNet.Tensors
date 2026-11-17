using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    internal static (int Rows, int Columns) ValidateWeightNorm<T>(Tensor<T> direction, Tensor<T> gain, double epsilon)
    {
        if (direction is null) throw new ArgumentNullException(nameof(direction));
        if (gain is null) throw new ArgumentNullException(nameof(gain));
        if (double.IsNaN(epsilon) || double.IsInfinity(epsilon) || epsilon < 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Weight normalization epsilon must be finite and nonnegative.");
        if (direction.Rank < 1) throw new ArgumentException("Direction must have a row dimension.", nameof(direction));
        int rows = direction._shape[0], columns = 1;
        for (int i = 1; i < direction.Rank; i++) columns = checked(columns * direction._shape[i]);
        if (gain.Rank != 1 || gain.Length != rows)
            throw new ArgumentException("Gain must be rank one with one value per first-dimension row.", nameof(gain));
        return (rows, columns);
    }

    internal static (int Rows, int Columns) ValidateWeightNormBackward<T>(Tensor<T> gradOutput,
        Tensor<T> direction, Tensor<T> gain, Tensor<T>? gradDirection, Tensor<T>? gradGain, double epsilon)
    {
        var geometry = ValidateWeightNorm(direction, gain, epsilon);
        RequireWeightNormShape(gradOutput, direction, nameof(gradOutput));
        if (gradDirection is not null) RequireWeightNormShape(gradDirection, direction, nameof(gradDirection));
        if (gradGain is not null) RequireWeightNormShape(gradGain, gain, nameof(gradGain));
        return geometry;
    }

    private static void RequireWeightNormShape<T>(Tensor<T> value, Tensor<T> expected, string name)
    {
        if (value is null) throw new ArgumentNullException(name);
        if (!value._shape.AsSpan().SequenceEqual(expected._shape))
            throw new ArgumentException("Weight normalization destination and upstream shapes must match their corresponding source exactly.", name);
    }

    /// <inheritdoc/>
    public virtual void TensorWeightNormInto<T>(Tensor<T> output, Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24)
    {
        var (rows, columns) = ValidateWeightNorm(direction, gain, epsilon);
        RequireWeightNormShape(output, direction, nameof(output));
        using var keepDirection = new KeepAliveScope(direction);
        using var keepGain = new KeepAliveScope(gain);
        using var keepOutput = new KeepAliveScope(output);
        if (!output.IsContiguous || !direction.IsContiguous || !gain.IsContiguous)
        {
            WeightNormForwardFallback(output, direction, gain, rows, columns, epsilon);
            return;
        }
        var source = direction.ReadOnlyDataUnmarked;
        var scales = gain.ReadOnlyDataUnmarked;
        var destination = output.DataUnmarked;
        if (source.Span.Overlaps(destination.Span) || scales.Span.Overlaps(destination.Span))
        {
            WeightNormForwardFallback(output, direction, gain, rows, columns, epsilon);
            return;
        }
        WeightNormForwardRows(source, scales, destination, rows, columns, epsilon);
        output.IncrementVersion();
    }

    /// <inheritdoc/>
    public virtual void TensorWeightNormBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> direction, Tensor<T> gain,
        Tensor<T>? gradDirection, Tensor<T>? gradGain, double epsilon = 1e-24,
        bool accumulateDirection = false, bool accumulateGain = false)
    {
        var (rows, columns) = ValidateWeightNormBackward(gradOutput, direction, gain, gradDirection, gradGain, epsilon);
        if (gradDirection is null && gradGain is null) return;
        using var keepDirection = new KeepAliveScope(direction);
        using var keepGain = new KeepAliveScope(gain);
        using var keepGradOutput = new KeepAliveScope(gradOutput);
        using var keepGradDirection = new KeepAliveScope(gradDirection);
        using var keepGradGain = new KeepAliveScope(gradGain);
        if (!direction.IsContiguous || !gain.IsContiguous || !gradOutput.IsContiguous
            || gradDirection is { IsContiguous: false } || gradGain is { IsContiguous: false })
        {
            WeightNormBackwardFallback(gradOutput, direction, gain, gradDirection, gradGain, rows, columns,
                epsilon, accumulateDirection, accumulateGain);
            return;
        }
        var source = direction.ReadOnlyDataUnmarked;
        var scales = gain.ReadOnlyDataUnmarked;
        var upstream = gradOutput.ReadOnlyDataUnmarked;
        var directionDestination = gradDirection is null ? Memory<T>.Empty : gradDirection.DataUnmarked;
        var gainDestination = gradGain is null ? Memory<T>.Empty : gradGain.DataUnmarked;
        bool OverlapsInput(Memory<T> destination) => source.Span.Overlaps(destination.Span)
            || scales.Span.Overlaps(destination.Span) || upstream.Span.Overlaps(destination.Span);
        if (OverlapsInput(directionDestination) || OverlapsInput(gainDestination)
            || directionDestination.Span.Overlaps(gainDestination.Span))
        {
            WeightNormBackwardFallback(gradOutput, direction, gain, gradDirection, gradGain, rows, columns,
                epsilon, accumulateDirection, accumulateGain);
            return;
        }
        WeightNormBackwardRows(upstream, source, scales, directionDestination, gainDestination,
            gradDirection is not null, gradGain is not null, rows, columns, epsilon, accumulateDirection, accumulateGain);
        gradDirection?.IncrementVersion();
        gradGain?.IncrementVersion();
    }

    private static void WeightNormForwardRows<T>(ReadOnlyMemory<T> direction, ReadOnlyMemory<T> gain,
        Memory<T> output, int rows, int columns, double epsilon)
    {
        if (rows == 0 || columns == 0) return;
        if (typeof(T) == typeof(float))
        {
            var v = AsFloatReadOnlyMemory(direction);
            var g = AsFloatReadOnlyMemory(gain);
            var w = AsFloatMemory(output);
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
            {
                var values = v.Span.Slice(row * columns, columns);
                float inverse = (float)(1.0 / Math.Sqrt(Simd.SimdKernels.DotProduct(values, values) + (float)epsilon));
                Simd.SimdKernels.MultiplyScalar(values, g.Span[row] * inverse, w.Span.Slice(row * columns, columns));
            }, deterministicSafe: true);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var v = AsDoubleReadOnlyMemory(direction);
            var g = AsDoubleReadOnlyMemory(gain);
            var w = AsDoubleMemory(output);
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
            {
                var values = v.Span.Slice(row * columns, columns);
                double inverse = 1.0 / Math.Sqrt(Simd.SimdKernels.DotProduct(values, values) + epsilon);
                Simd.SimdKernels.MultiplyScalar(values, g.Span[row] * inverse, w.Span.Slice(row * columns, columns));
            }, deterministicSafe: true);
            return;
        }
        var ops = MathHelper.GetNumericOperations<T>();
        var epsilonValue = ops.FromDouble(epsilon);
        CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
        {
            var values = direction.Span.Slice(row * columns, columns);
            var result = output.Span.Slice(row * columns, columns);
            var squared = ops.Zero;
            for (int i = 0; i < columns; i++) squared = ops.Add(squared, ops.Multiply(values[i], values[i]));
            var scale = ops.Divide(gain.Span[row], ops.Sqrt(ops.Add(squared, epsilonValue)));
            for (int i = 0; i < columns; i++) result[i] = ops.Multiply(values[i], scale);
        }, deterministicSafe: true);
    }

    private static void WeightNormBackwardRows<T>(ReadOnlyMemory<T> gradOutput, ReadOnlyMemory<T> direction,
        ReadOnlyMemory<T> gain, Memory<T> gradDirection, Memory<T> gradGain, bool writeDirection, bool writeGain,
        int rows, int columns, double epsilon, bool accumulateDirection, bool accumulateGain)
    {
        if (rows == 0) return;
        if (columns == 0)
        {
            if (writeGain && !accumulateGain) gradGain.Span.Fill(MathHelper.GetNumericOperations<T>().Zero);
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var v = AsFloatReadOnlyMemory(direction);
            var g = AsFloatReadOnlyMemory(gain);
            var h = AsFloatReadOnlyMemory(gradOutput);
            var dv = AsFloatMemory(gradDirection);
            var dg = AsFloatMemory(gradGain);
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
            {
                var values = v.Span.Slice(row * columns, columns);
                var upstream = h.Span.Slice(row * columns, columns);
                float inverse = (float)(1.0 / Math.Sqrt(Simd.SimdKernels.DotProduct(values, values) + (float)epsilon));
                float dot = Simd.SimdKernels.DotProduct(upstream, values);
                if (writeDirection)
                {
                    var destination = dv.Span.Slice(row * columns, columns);
                    float scale = g.Span[row] * inverse;
                    float projection = dot * inverse * inverse;
                    if (accumulateDirection)
                    {
                        Simd.SimdKernels.ScalarMultiplyAdd(destination, upstream, scale, destination);
                        Simd.SimdKernels.ScalarMultiplyAdd(destination, values, -scale * projection, destination);
                    }
                    else
                    {
                        Simd.SimdKernels.ScalarMultiplyAdd(upstream, values, -projection, destination);
                        Simd.SimdKernels.MultiplyScalar(destination, scale, destination);
                    }
                }
                if (writeGain) dg.Span[row] = (accumulateGain ? dg.Span[row] : 0f) + inverse * dot;
            }, deterministicSafe: true);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            var v = AsDoubleReadOnlyMemory(direction);
            var g = AsDoubleReadOnlyMemory(gain);
            var h = AsDoubleReadOnlyMemory(gradOutput);
            var dv = AsDoubleMemory(gradDirection);
            var dg = AsDoubleMemory(gradGain);
            CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
            {
                var values = v.Span.Slice(row * columns, columns);
                var upstream = h.Span.Slice(row * columns, columns);
                double inverse = 1.0 / Math.Sqrt(Simd.SimdKernels.DotProduct(values, values) + epsilon);
                double dot = Simd.SimdKernels.DotProduct(upstream, values);
                if (writeDirection)
                {
                    var destination = dv.Span.Slice(row * columns, columns);
                    double scale = g.Span[row] * inverse;
                    double projection = dot * inverse * inverse;
                    if (accumulateDirection)
                    {
                        Simd.SimdKernels.ScalarMultiplyAdd(destination, upstream, scale, destination);
                        Simd.SimdKernels.ScalarMultiplyAdd(destination, values, -scale * projection, destination);
                    }
                    else
                    {
                        Simd.SimdKernels.ScalarMultiplyAdd(upstream, values, -projection, destination);
                        Simd.SimdKernels.MultiplyScalar(destination, scale, destination);
                    }
                }
                if (writeGain) dg.Span[row] = (accumulateGain ? dg.Span[row] : 0.0) + inverse * dot;
            }, deterministicSafe: true);
            return;
        }
        var ops = MathHelper.GetNumericOperations<T>();
        var epsilonValue = ops.FromDouble(epsilon);
        CpuParallelSettings.ParallelForOrSerial(0, rows, (long)rows * columns, row =>
        {
            var values = direction.Span.Slice(row * columns, columns);
            var upstream = gradOutput.Span.Slice(row * columns, columns);
            var squared = ops.Zero;
            var dot = ops.Zero;
            for (int i = 0; i < columns; i++)
            {
                squared = ops.Add(squared, ops.Multiply(values[i], values[i]));
                dot = ops.Add(dot, ops.Multiply(upstream[i], values[i]));
            }
            var inverse = ops.Divide(ops.One, ops.Sqrt(ops.Add(squared, epsilonValue)));
            if (writeDirection)
            {
                var result = gradDirection.Span.Slice(row * columns, columns);
                var scale = ops.Multiply(gain.Span[row], inverse);
                var projection = ops.Multiply(ops.Multiply(dot, inverse), inverse);
                for (int i = 0; i < columns; i++)
                {
                    var value = ops.Multiply(scale, ops.Subtract(upstream[i], ops.Multiply(values[i], projection)));
                    result[i] = accumulateDirection ? ops.Add(result[i], value) : value;
                }
            }
            if (writeGain)
            {
                var value = ops.Multiply(inverse, dot);
                gradGain.Span[row] = accumulateGain ? ops.Add(gradGain.Span[row], value) : value;
            }
        }, deterministicSafe: true);
    }

    // Actual overlaps or strided layouts take owned logical snapshots. The ordinary contiguous path above never
    // allocates a weight-sized intermediate, even when accumulating into an existing gradient.
    private static void WeightNormForwardFallback<T>(Tensor<T> output, Tensor<T> direction, Tensor<T> gain,
        int rows, int columns, double epsilon)
    {
        var result = new T[direction.Length];
        WeightNormForwardRows<T>(direction.ToArray(), gain.ToArray(), result, rows, columns, epsilon);
        CommitWeightNormResult(output, result, false);
    }

    private static void WeightNormBackwardFallback<T>(Tensor<T> gradOutput, Tensor<T> direction, Tensor<T> gain,
        Tensor<T>? gradDirection, Tensor<T>? gradGain, int rows, int columns, double epsilon,
        bool accumulateDirection, bool accumulateGain)
    {
        var dv = gradDirection is null ? Array.Empty<T>() : new T[direction.Length];
        var dg = gradGain is null ? Array.Empty<T>() : new T[gain.Length];
        WeightNormBackwardRows<T>(gradOutput.ToArray(), direction.ToArray(), gain.ToArray(), dv, dg,
            gradDirection is not null, gradGain is not null, rows, columns, epsilon, false, false);
        // Preserve complete logical dV-then-dG writes when caller destinations themselves overlap.
        if (gradDirection is not null) CommitWeightNormResult(gradDirection, dv, accumulateDirection);
        if (gradGain is not null) CommitWeightNormResult(gradGain, dg, accumulateGain);
    }

    private static void CommitWeightNormResult<T>(Tensor<T> output, T[] values, bool accumulate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (output.IsContiguous)
        {
            var destination = output.AsWritableSpanUnmarked();
            if (accumulate) ops.Add(destination, values.AsSpan(), destination);
            else values.AsSpan().CopyTo(destination);
        }
        else
        {
            for (int i = 0; i < values.Length; i++) output[i] = accumulate ? ops.Add(output[i], values[i]) : values[i];
        }
        output.IncrementVersion();
    }
}
