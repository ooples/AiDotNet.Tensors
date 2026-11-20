using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    internal static (int Rows, int Length, int Bins) ValidateRfftMagnitude<T>(Tensor<T> input, Tensor<T>? output, double epsilon)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank < 1 || input._shape[input.Rank - 1] <= 0)
            throw new ArgumentException("RFFT magnitude requires a positive last dimension.", nameof(input));
        if (double.IsNaN(epsilon) || double.IsInfinity(epsilon) || epsilon < 0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Magnitude epsilon must be finite and nonnegative.");
        int length = input._shape[input.Rank - 1], rows = 1;
        for (int axis = 0; axis < input.Rank - 1; axis++) rows = checked(rows * input._shape[axis]);
        int bins = length / 2 + 1;
        if (output is not null)
        {
            if (output.Rank != input.Rank || output._shape[output.Rank - 1] != bins)
                throw new ArgumentException("RFFT magnitude output must retain leading dimensions and have N/2+1 bins.", nameof(output));
            for (int axis = 0; axis < input.Rank - 1; axis++)
                if (output._shape[axis] != input._shape[axis])
                    throw new ArgumentException("RFFT magnitude output must retain every leading dimension.", nameof(output));
        }
        return (rows, length, bins);
    }

    internal static (int Rows, int Length, int Bins) ValidateRfftMagnitudeBackward<T>(Tensor<T> gradOutput,
        Tensor<T> input, Tensor<T> gradInput, double epsilon)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradInput is null) throw new ArgumentNullException(nameof(gradInput));
        var geometry = ValidateRfftMagnitude(input, gradOutput, epsilon);
        if (!gradInput._shape.AsSpan().SequenceEqual(input._shape))
            throw new ArgumentException("RFFT magnitude input gradient must match input shape exactly.", nameof(gradInput));
        return geometry;
    }

    internal static void RfftMagnitudeIntoCore<T>(Tensor<T> output, Tensor<T> input, double epsilon)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        var (rows, length, bins) = ValidateRfftMagnitude(input, output, epsilon);
        using var keepInput = new KeepAliveScope(input);
        using var keepOutput = new KeepAliveScope(output);
        var destination = output.IsContiguous ? output.DataUnmarked : default;
        // Separate FromMemory wrappers can physically alias without sharing a storage-identity object.
        // A strided destination has no single logical span to test, so snapshot that rare fallback conservatively.
        ReadOnlyMemory<T> source = input.IsContiguous && output.IsContiguous ? input.ReadOnlyDataUnmarked : input.ToArray();
        if (output.IsContiguous && source.Span.Overlaps(destination.Span))
            source = input.ToArray();
        RfftMagnitudeRows(source, default, destination, output.IsContiguous ? null : output,
            rows, length, bins, epsilon, backward: false, accumulate: false);
        output.IncrementVersion();
    }

    internal static void RfftMagnitudeBackwardIntoCore<T>(Tensor<T> gradOutput, Tensor<T> input,
        Tensor<T> gradInput, double epsilon, bool accumulate)
    {
        var (rows, length, bins) = ValidateRfftMagnitudeBackward(gradOutput, input, gradInput, epsilon);
        using var keepInput = new KeepAliveScope(input);
        using var keepUpstream = new KeepAliveScope(gradOutput);
        using var keepDestination = new KeepAliveScope(gradInput);
        var destination = gradInput.IsContiguous ? gradInput.DataUnmarked : default;
        ReadOnlyMemory<T> source = input.IsContiguous && gradInput.IsContiguous ? input.ReadOnlyDataUnmarked : input.ToArray();
        ReadOnlyMemory<T> upstream = gradOutput.IsContiguous && gradInput.IsContiguous ? gradOutput.ReadOnlyDataUnmarked : gradOutput.ToArray();
        if (gradInput.IsContiguous && source.Span.Overlaps(destination.Span)) source = input.ToArray();
        if (gradInput.IsContiguous && upstream.Span.Overlaps(destination.Span)) upstream = gradOutput.ToArray();
        RfftMagnitudeRows(source, upstream, destination, gradInput.IsContiguous ? null : gradInput,
            rows, length, bins, epsilon, backward: true, accumulate);
        gradInput.IncrementVersion();
    }

    private static void RfftMagnitudeRows<T>(ReadOnlyMemory<T> source, ReadOnlyMemory<T> upstream,
        Memory<T> destination, Tensor<T>? stridedDestination, int rows, int length, int bins,
        double epsilon, bool backward, bool accumulate)
    {
        if (rows == 0) return;
        int scratchLength = checked(2 * length);
        if (stridedDestination is null && TryMklRfftMagnitudeRows(source, upstream, destination,
                rows, length, bins, epsilon, backward, accumulate)) return;

        // Logical tensor writes can privatize COW storage on their first write. Keep that rare strided
        // path serial; contiguous destinations were privatized before entering the parallel region.
        int workers = stridedDestination is null ? Math.Max(1, Math.Min(rows, CpuParallelSettings.MaxDegreeOfParallelism)) : 1;
        CpuParallelSettings.ParallelForOrSerial(0, workers, (long)rows * length * 16, worker =>
        {
            // Reuse the existing CPU radix2/Bluestein implementation. Only one row of FFT scratch per worker
            // remains live; ordinary contiguous inputs/outputs do not materialize a batch-sized spectrum.
            var scratch = CpuStorageAllocationDiagnostics.RentWorkspace<double>(scratchLength, "workspace.rfft_magnitude_fft");
            try
            {
                var spectrum = scratch.AsSpan(0, scratchLength);
                for (int row = worker; row < rows; row += workers)
                {
                    LoadRfftMagnitudeRealRow(source, row * length, spectrum, length);
                    FftKernels.Transform1D(spectrum, length, inverse: false, FftNorm.Backward);
                    if (backward)
                    {
                        SeedRfftMagnitudeAdjoint(upstream, row * bins, spectrum, bins, epsilon);
                        spectrum.Slice(2 * bins).Clear();
                        // The loss observes positive bins only. Its real-input adjoint is the real part of
                        // the UNNORMALIZED inverse of a zero-padded complex seed, with no Hermitian factor2.
                        FftKernels.Transform1D(spectrum, length, inverse: true, FftNorm.Forward);
                        StoreRfftMagnitudeRow(destination, stridedDestination, row * length, spectrum,
                            length, epsilon, backward: true, accumulate);
                    }
                    else
                        StoreRfftMagnitudeRow(destination, stridedDestination, row * bins, spectrum,
                            bins, epsilon, backward: false, accumulate: false);
                }
            }
            finally { ArrayPool<double>.Shared.Return(scratch); }
        }, deterministicSafe: true);
    }

    private static void LoadRfftMagnitudeRealRow<T>(ReadOnlyMemory<T> source, int start, Span<double> spectrum, int length)
    {
        spectrum.Clear();
        if (typeof(T) == typeof(float))
        {
            var values = AsFloatReadOnlyMemory(source).Span.Slice(start, length);
            for (int i = 0; i < length; i++) spectrum[2 * i] = values[i];
        }
        else if (typeof(T) == typeof(double))
        {
            var values = AsDoubleReadOnlyMemory(source).Span.Slice(start, length);
            for (int i = 0; i < length; i++) spectrum[2 * i] = values[i];
        }
        else LoadRfftMagnitudeGenericRealRow(source.Span.Slice(start, length), spectrum);
    }

    [ScalarKernel("Non-float/double CPU FFT input uses numeric conversions into the existing double-interleaved FFT core; native float/double copies are specialized above.")]
    private static void LoadRfftMagnitudeGenericRealRow<T>(ReadOnlySpan<T> source, Span<double> spectrum)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < source.Length; i++) spectrum[2 * i] = ops.ToDouble(source[i]);
    }

    private static void SeedRfftMagnitudeAdjoint<T>(ReadOnlyMemory<T> upstream, int start,
        Span<double> spectrum, int bins, double epsilon)
    {
        if (typeof(T) == typeof(float))
        {
            var values = AsFloatReadOnlyMemory(upstream).Span.Slice(start, bins);
            for (int bin = 0; bin < bins; bin++)
            {
                double re = spectrum[2 * bin], im = spectrum[2 * bin + 1];
                double scale = values[bin] / Math.Sqrt(re * re + im * im + epsilon);
                spectrum[2 * bin] = re * scale;
                spectrum[2 * bin + 1] = im * scale;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var values = AsDoubleReadOnlyMemory(upstream).Span.Slice(start, bins);
            for (int bin = 0; bin < bins; bin++)
            {
                double re = spectrum[2 * bin], im = spectrum[2 * bin + 1];
                double scale = values[bin] / Math.Sqrt(re * re + im * im + epsilon);
                spectrum[2 * bin] = re * scale;
                spectrum[2 * bin + 1] = im * scale;
            }
        }
        else SeedRfftMagnitudeGenericAdjoint(upstream.Span.Slice(start, bins), spectrum, epsilon);
    }

    [ScalarKernel("Non-float/double upstream values require their numeric conversions at the CPU FFT boundary; float/double spectral seeds are specialized above.")]
    private static void SeedRfftMagnitudeGenericAdjoint<T>(ReadOnlySpan<T> upstream, Span<double> spectrum, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int bin = 0; bin < upstream.Length; bin++)
        {
            double re = spectrum[2 * bin], im = spectrum[2 * bin + 1];
            double scale = ops.ToDouble(upstream[bin]) / Math.Sqrt(re * re + im * im + epsilon);
            spectrum[2 * bin] = re * scale;
            spectrum[2 * bin + 1] = im * scale;
        }
    }

    private static void StoreRfftMagnitudeRow<T>(Memory<T> destination, Tensor<T>? stridedDestination,
        int start, ReadOnlySpan<double> spectrum, int count, double epsilon, bool backward, bool accumulate)
    {
        if (stridedDestination is not null)
        {
            StoreRfftMagnitudeStridedRow(stridedDestination, start, spectrum, count, epsilon, backward, accumulate);
            return;
        }
        if (typeof(T) == typeof(float))
        {
            var values = AsFloatMemory(destination).Span.Slice(start, count);
            for (int i = 0; i < count; i++)
            {
                double re = spectrum[2 * i], im = spectrum[2 * i + 1];
                float value = (float)(backward ? re : Math.Sqrt(re * re + im * im + epsilon));
                values[i] = accumulate ? values[i] + value : value;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var values = AsDoubleMemory(destination).Span.Slice(start, count);
            for (int i = 0; i < count; i++)
            {
                double re = spectrum[2 * i], im = spectrum[2 * i + 1];
                double value = backward ? re : Math.Sqrt(re * re + im * im + epsilon);
                values[i] = accumulate ? values[i] + value : value;
            }
        }
        else StoreRfftMagnitudeGenericRow(destination.Span.Slice(start, count), spectrum, epsilon, backward, accumulate);
    }

    [ScalarKernel("Non-float/double outputs convert from the existing double-interleaved CPU FFT and use numeric addition only when accumulating; float/double destinations are specialized above.")]
    private static void StoreRfftMagnitudeGenericRow<T>(Span<T> output, ReadOnlySpan<double> spectrum,
        double epsilon, bool backward, bool accumulate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < output.Length; i++)
        {
            double re = spectrum[2 * i], im = spectrum[2 * i + 1];
            var value = ops.FromDouble(backward ? re : Math.Sqrt(re * re + im * im + epsilon));
            output[i] = accumulate ? ops.Add(output[i], value) : value;
        }
    }

    [ScalarKernel("Only non-contiguous FFT destinations require logical tensor index writes; contiguous float/double outputs use direct specialized spans.")]
    private static void StoreRfftMagnitudeStridedRow<T>(Tensor<T> output, int start, ReadOnlySpan<double> spectrum,
        int count, double epsilon, bool backward, bool accumulate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < count; i++)
        {
            double re = spectrum[2 * i], im = spectrum[2 * i + 1];
            var value = ops.FromDouble(backward ? re : Math.Sqrt(re * re + im * im + epsilon));
            output[start + i] = accumulate ? ops.Add(output[start + i], value) : value;
        }
    }
}
