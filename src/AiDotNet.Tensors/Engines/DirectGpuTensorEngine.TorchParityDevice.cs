using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for the parity ops the op-parity registry probes on the GPU: the real/imaginary views of a
/// complex tensor (copies out of the split device buffers), the indexed max pools (a gather of each window's taps
/// followed by a device max-reduction) and max unpooling (a gather from the input with a zero appended). Under an
/// active tape they defer to the CPU implementation, which records the backward, the convention the engine's other
/// pooling overrides follow; graph capture fails closed in both.
/// </summary>
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorReal<T>(Tensor<Complex<T>> input) => ComplexPart(input, imaginary: false) ?? base.TensorReal(input);

    /// <inheritdoc/>
    public override Tensor<T> TensorImag<T>(Tensor<Complex<T>> input) => ComplexPart(input, imaginary: true) ?? base.TensorImag(input);

    private Tensor<T>? ComplexPart<T>(Tensor<Complex<T>> input, bool imaginary)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, imaginary ? "TensorImag" : "TensorReal");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0 || !TryGetBackend(out var backend) || DirectGpuEngine.ShouldFallbackForPrecision<T>()) return null;
        try
        {
            int n = input.Length;
            var parts = GetOrAllocateSplitComplexBuffers(backend, input);
            using var re = parts.Real;
            using var im = parts.Imaginary;
            var source = imaginary ? im.Buffer : re.Buffer;
            var result = DispatchDeferredGpuOp<T>(backend, n, input.Shape.ToArray(), output => backend.Copy(source, output, n));
            backend.Synchronize();
            return result;
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(imaginary ? "TensorImag" : "TensorReal", ex);
            return null;
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorViewAsReal<T>(Tensor<Complex<T>> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0 || !TryGetBackend(out var backend) || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return base.TensorViewAsReal(input);
        try
        {
            int n = input.Length;
            var parts = GetOrAllocateSplitComplexBuffers(backend, input);
            using var re = parts.Real;
            using var im = parts.Imaginary;
            var shape = input.Shape.ToArray().Concat(new[] { 2 }).ToArray();
            var result = DispatchDeferredGpuOp<T>(backend, 2 * n, shape, output => backend.InterleaveComplex(re.Buffer, im.Buffer, output, n));
            backend.Synchronize();
            return result;
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorViewAsReal", ex);
            return base.TensorViewAsReal(input);
        }
    }

    /// <inheritdoc/>
    public override (Tensor<T> Output, Tensor<int> Indices) TensorAdaptiveMaxPoolWithIndices<T>(Tensor<T> input, int[] outputSize)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (IsTapeActive<T>()) return base.TensorAdaptiveMaxPoolWithIndices(input, outputSize);
        return DeviceWindowMax(input, AdaptivePoolPlan(input, outputSize), "TensorAdaptiveMaxPoolWithIndices")
            ?? base.TensorAdaptiveMaxPoolWithIndices(input, outputSize);
    }

    /// <inheritdoc/>
    public override (Tensor<T> Output, Tensor<int> Indices) TensorMaxPool1DWithIndices<T>(Tensor<T> input, int kernelSize,
        int stride = 0, int padding = 0, int dilation = 1, bool ceilMode = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (IsTapeActive<T>()) return base.TensorMaxPool1DWithIndices(input, kernelSize, stride, padding, dilation, ceilMode);
        return DeviceWindowMax(input, MaxPool1DPlan(input, kernelSize, stride, padding, dilation, ceilMode), "TensorMaxPool1DWithIndices")
            ?? base.TensorMaxPool1DWithIndices(input, kernelSize, stride, padding, dilation, ceilMode);
    }

    /// <inheritdoc/>
    public override (Tensor<T> Output, Tensor<int> Indices) TensorFractionalMaxPool<T>(Tensor<T> input, int[] kernelSize,
        int[] outputSize, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (IsTapeActive<T>()) return base.TensorFractionalMaxPool(input, kernelSize, outputSize, seed);
        // The plan draws the window starts; a fallback redraws them from the same seed (or freshly, with no seed).
        var plan = FractionalPoolPlan(input, kernelSize, outputSize, seed);
        return DeviceWindowMax(input, plan, "TensorFractionalMaxPool") ?? base.TensorFractionalMaxPool(input, kernelSize, outputSize, seed);
    }

    // Max over each window of a pooling plan, on the device: the taps of every window (built from shapes alone, padded
    // to a common width by repeating a tap, which cannot change a maximum) are gathered into an [outputs, width]
    // matrix and max-reduced along its rows. Returns null to fall back when no backend or an unsupported precision.
    private (Tensor<T> Output, Tensor<int> Indices)? DeviceWindowMax<T>(Tensor<T> x, PoolPlan plan, string opName)
    {
        if (x.Length == 0 || !TryGetBackend(out _) || DirectGpuEngine.ShouldFallbackForPrecision<T>()) return null;
        try
        {
            var table = WindowTapTable(x, plan, out int planeSize);
            int dims = plan.OutputSize.Length, n = table.Length, width = 0;
            int outPlane = plan.OutputSize.Aggregate(1, (a, b) => a * b);
            foreach (var taps in table)
            {
                if (taps.Count == 0) return null;
                width = Math.Max(width, taps.Count);
            }
            var flat = new int[n * width];
            for (int o = 0; o < n; o++)
            {
                int planeBase = o / outPlane * planeSize;
                var taps = table[o];
                for (int j = 0; j < width; j++) flat[o * width + j] = planeBase + taps[Math.Min(j, taps.Count - 1)];
            }
            var gathered = Reshape(TensorTake(Reshape(x, new[] { x.Length }), new Tensor<int>(flat, new[] { n * width })), new[] { n, width });
            var values = ReduceMax(gathered, new[] { 1 }, false, out int[] arg);
            var indices = new int[n];
            for (int o = 0; o < n; o++)
            {
                // The reduction reports either the column or the flat position in the matrix; both name the same tap.
                int j = arg[o] >= width ? arg[o] - o * width : arg[o];
                indices[o] = flat[o * width + j] - o / outPlane * planeSize;
            }
            var outShape = x._shape.Take(x.Rank - dims).Concat(plan.OutputSize).ToArray();
            return (Reshape(values, outShape), new Tensor<int>(indices, outShape));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMaxUnpool<T>(Tensor<T> input, Tensor<int> indices, int[] outputSize)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        if (IsTapeActive<T>() || input is null || input.Length == 0 || !TryGetBackend(out _) || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return base.TensorMaxUnpool(input ?? throw new ArgumentNullException(nameof(input)), indices, outputSize);
        var source = MaxUnpoolSource(input, indices, outputSize, out int[] shape);
        try
        {
            // Gather from the input with one zero appended; positions no index names read the zero.
            int zero = input.Length;
            var map = source.Select(s => s < 0 ? zero : s).ToArray();
            var padded = TensorConcatenate(new[] { Reshape(input, new[] { input.Length }), new Tensor<T>(new[] { 1 }) }, 0);
            return Reshape(TensorTake(padded, new Tensor<int>(map, new[] { map.Length })), shape);
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorMaxUnpool", ex);
            return base.TensorMaxUnpool(input, indices, outputSize);
        }
    }
}
