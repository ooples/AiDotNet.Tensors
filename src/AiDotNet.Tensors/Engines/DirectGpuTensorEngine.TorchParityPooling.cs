using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for the parity pooling ops that computed on the host (#1112 tier 1): adaptive max pooling over 1 and 3
/// axes reuses <see cref="DeviceWindowMax{T}"/>; adaptive average pooling over 1 and 3 axes and Lp pooling gather each
/// window's taps (padded with an appended zero) on the device, sum the rows, and finish with the window's 1/count or
/// the 1/p power -- the CPU's WindowPool reductions. Under an active tape they defer to the CPU implementation, which
/// records the backward.
/// </summary>
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveMaxPool1D<T>(Tensor<T> input, int outputSize)
        => AdaptiveMaxOnDevice(input, new[] { outputSize }, "TensorAdaptiveMaxPool1D") ?? HostFallback("TensorAdaptiveMaxPool1D", () => base.TensorAdaptiveMaxPool1D(input, outputSize));

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveMaxPool3D<T>(Tensor<T> input, int[] outputSize)
        => (outputSize is { Length: 3 } ? AdaptiveMaxOnDevice(input, outputSize, "TensorAdaptiveMaxPool3D") : null)
           ?? HostFallback("TensorAdaptiveMaxPool3D", () => base.TensorAdaptiveMaxPool3D(input, outputSize));

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveAvgPool1D<T>(Tensor<T> input, int outputSize)
        => AdaptiveAverageOnDevice(input, new[] { outputSize }, "TensorAdaptiveAvgPool1D") ?? HostFallback("TensorAdaptiveAvgPool1D", () => base.TensorAdaptiveAvgPool1D(input, outputSize));

    /// <inheritdoc/>
    public override Tensor<T> TensorAdaptiveAvgPool3D<T>(Tensor<T> input, int[] outputSize)
        => (outputSize is { Length: 3 } ? AdaptiveAverageOnDevice(input, outputSize, "TensorAdaptiveAvgPool3D") : null)
           ?? HostFallback("TensorAdaptiveAvgPool3D", () => base.TensorAdaptiveAvgPool3D(input, outputSize));

    /// <inheritdoc/>
    public override Tensor<T> TensorLpPool<T>(Tensor<T> input, double power, int[] kernelSize, int[]? stride = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        // The zero pad contributes 0^p, which is 0 only for p > 0; other powers (and invalid arguments) take the CPU path.
        if (input is not null && power > 0 && !double.IsInfinity(power) && kernelSize is { Length: > 0 }
            && (stride is null || stride.Length == kernelSize.Length)
            && kernelSize.All(k => k > 0) && (stride ?? kernelSize).All(s => s > 0)
            && input.Rank >= kernelSize.Length + 1)
        {
            int dims = kernelSize.Length;
            var steps = stride ?? kernelSize;
            var spatial = input._shape.Skip(input.Rank - dims).ToArray();
            if (Enumerable.Range(0, dims).All(d => kernelSize[d] <= spatial[d]))
            {
                var outSize = spatial.Select((n, d) => (n - kernelSize[d]) / steps[d] + 1).ToArray();
                var plan = new PoolPlan(outSize, (_, d, o) => new PoolWindow(o * steps[d], kernelSize[d]));
                if (WindowSumOnDevice(input, plan, "TensorLpPool", power) is { } onDevice) return onDevice;
            }
        }
        return HostFallback("TensorLpPool", () => base.TensorLpPool(input!, power, kernelSize, stride));
    }

    // The CPU path for an op whose device path declined: recorded, as every host-computed parity op is.
    private static Tensor<T> HostFallback<T>(string opName, Func<Tensor<T>> host)
    {
        GpuLaunchProbe.OnFallback(opName + ": no device kernel", null);
        return host();
    }

    private Tensor<T>? AdaptiveMaxOnDevice<T>(Tensor<T> input, int[] outputSize, string opName)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (input is null || IsTapeActive<T>(opName) || outputSize.Any(o => o <= 0) || input.Rank < outputSize.Length + 1) return null;
        return DeviceWindowMax(input, AdaptivePoolPlan(input, outputSize), opName)?.Output;
    }

    private Tensor<T>? AdaptiveAverageOnDevice<T>(Tensor<T> input, int[] outputSize, string opName)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (input is null || outputSize.Any(o => o <= 0) || input.Rank < outputSize.Length + 1) return null;
        return WindowSumOnDevice(input, AdaptivePoolPlan(input, outputSize), opName, power: null);
    }

    // Sum over each window of a pooling plan on the device, finished as an average (power null) or as the Lp norm
    // (sum of x^p, then ^(1/p)). The taps (built from shapes alone) are padded to a common width with the index of a
    // zero appended to the input, which contributes nothing to either sum.
    private Tensor<T>? WindowSumOnDevice<T>(Tensor<T> x, PoolPlan plan, string opName, double? power)
    {
        if (x.Length == 0 || IsTapeActive<T>(opName) || !TryGetBackend(out _) || DirectGpuEngine.ShouldFallbackForPrecision<T>()) return null;
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
            int zero = x.Length;
            var flat = new int[n * width];
            var inverseCount = new T[n];
            var ops = MathHelper.GetNumericOperations<T>();
            for (int o = 0; o < n; o++)
            {
                int planeBase = o / outPlane * planeSize;
                var taps = table[o];
                for (int j = 0; j < width; j++) flat[o * width + j] = j < taps.Count ? planeBase + taps[j] : zero;
                inverseCount[o] = ops.FromDouble(1.0 / taps.Count);
            }
            var padded = TensorConcatenate(new[] { Reshape(x, new[] { x.Length }), new Tensor<T>(new[] { 1 }) }, 0);
            var gathered = Reshape(TensorTake(padded, new Tensor<int>(flat, new[] { n * width })), new[] { n, width });
            if (power is double p) gathered = TensorPow(gathered, ops.FromDouble(p));
            var sums = ReduceSum(gathered, new[] { 1 }, keepDims: false);
            var reduced = power is double q
                ? TensorPow(sums, ops.FromDouble(1.0 / q))
                : TensorMultiply(sums, new Tensor<T>(inverseCount, new[] { n }));
            var outShape = x._shape.Take(x.Rank - dims).Concat(plan.OutputSize).ToArray();
            return Reshape(reduced, outShape);
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }
}
