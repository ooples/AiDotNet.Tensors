using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>Adaptive, Lp, fractional and indexed pooling, and max-unpooling (see the matching <see cref="IEngine"/> members).</summary>
public partial class CpuEngine
{
    private enum WindowReduce { Max, Average, Power }

    /// <summary>One window along one spatial axis: <c>Length</c> taps from <c>Start</c>, <c>Step</c> apart.</summary>
    private readonly struct PoolWindow
    {
        public PoolWindow(int start, int length, int step = 1) { Start = start; Length = length; Step = step; }
        public int Start { get; }
        public int Length { get; }
        public int Step { get; }
    }

    // Pools the trailing outputSize.Length axes of x; window(plane, axis, outputIndex) gives each window (taps outside
    // the input are skipped). Returns the pooled tensor and, per output, the argmax's flat index in its plane.
    private (Tensor<T> Output, Tensor<int> Indices) WindowPool<T>(string opName, Tensor<T> x, int[] outputSize,
        Func<int, int, int, PoolWindow> window, WindowReduce mode, double power = 2)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        int dims = outputSize.Length;
        if (x.Rank < dims + 1) throw new ArgumentException($"{opName} expects at least {dims + 1} dimensions.", nameof(x));
        var ops = MathHelper.GetNumericOperations<T>();
        var source = x.IsContiguous ? x : x.Contiguous();
        var values = source.AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();
        var spatial = source._shape.Skip(source.Rank - dims).ToArray();
        int planeSize = spatial.Aggregate(1, (a, b) => a * b), outPlane = outputSize.Aggregate(1, (a, b) => a * b);
        int planes = source.Length / planeSize;
        var result = new double[planes * outPlane];
        var argmax = new int[planes * outPlane];
        var taps = new List<int>();
        var position = new int[dims];
        for (int plane = 0; plane < planes; plane++)
        {
            for (int o = 0; o < outPlane; o++)
            {
                for (int d = dims - 1, rest = o; d >= 0; d--) { position[d] = rest % outputSize[d]; rest /= outputSize[d]; }
                taps.Clear();
                CollectTaps(plane, 0, 0, position, spatial, window, taps);
                double acc = mode == WindowReduce.Max ? double.NegativeInfinity : 0;
                int best = taps.Count > 0 ? taps[0] : -1;
                foreach (int t in taps)
                {
                    double v = values[plane * planeSize + t];
                    if (mode == WindowReduce.Max) { if (v > acc || double.IsNaN(v)) { acc = v; best = t; if (double.IsNaN(v)) break; } }
                    else acc += mode == WindowReduce.Power ? Math.Pow(v, power) : v;
                }
                result[plane * outPlane + o] = mode switch
                {
                    WindowReduce.Max => acc,
                    WindowReduce.Average => taps.Count == 0 ? 0 : acc / taps.Count,
                    _ => Math.Pow(acc, 1 / power),
                };
                argmax[plane * outPlane + o] = best;
            }
        }
        var outShape = source._shape.Take(source.Rank - dims).Concat(outputSize).ToArray();
        var output = FromDoubles<T>(outShape, i => result[i]);
        var state = new object[] { mode, power, window, outputSize, spatial, argmax };
        DifferentiableOps.RecordUnary(opName, output, x, WindowPoolBackward<T>, state);
        return (output, new Tensor<int>(argmax, outShape));
    }

    private static void CollectTaps(int plane, int axis, int offset, int[] position, int[] spatial,
        Func<int, int, int, PoolWindow> window, List<int> taps)
    {
        if (axis == spatial.Length) { taps.Add(offset); return; }
        var w = window(plane, axis, position[axis]);
        for (int k = 0; k < w.Length; k++)
        {
            int i = w.Start + k * w.Step;
            if (i < 0 || i >= spatial[axis]) continue;
            CollectTaps(plane, axis + 1, offset * spatial[axis] + i, position, spatial, window, taps);
        }
    }

    private static void WindowPoolBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var mode = (WindowReduce)savedState[0];
        double power = (double)savedState[1];
        var window = (Func<int, int, int, PoolWindow>)savedState[2];
        var outputSize = (int[])savedState[3];
        var spatial = (int[])savedState[4];
        var argmax = (int[])savedState[5];
        var ops = MathHelper.GetNumericOperations<T>();
        var x = inputs[0].IsContiguous ? inputs[0] : inputs[0].Contiguous();
        var xs = x.AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();
        var g = (gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous()).AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();
        var y = (output.IsContiguous ? output : output.Contiguous()).AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();
        int dims = outputSize.Length, planeSize = spatial.Aggregate(1, (a, b) => a * b), outPlane = outputSize.Aggregate(1, (a, b) => a * b);
        int planes = xs.Length / planeSize;
        var dx = new double[xs.Length];
        var taps = new List<int>();
        var position = new int[dims];
        for (int plane = 0; plane < planes; plane++)
            for (int o = 0; o < outPlane; o++)
            {
                int k = plane * outPlane + o;
                if (mode == WindowReduce.Max) { if (argmax[k] >= 0) dx[plane * planeSize + argmax[k]] += g[k]; continue; }
                for (int d = dims - 1, rest = o; d >= 0; d--) { position[d] = rest % outputSize[d]; rest /= outputSize[d]; }
                taps.Clear();
                CollectTaps(plane, 0, 0, position, spatial, window, taps);
                foreach (int t in taps)
                {
                    int i = plane * planeSize + t;
                    dx[i] += mode == WindowReduce.Average
                        ? g[k] / taps.Count
                        : g[k] * Math.Pow(xs[i], power - 1) * Math.Pow(y[k], 1 - power);   // ∂(Σxᵖ)^{1/p}/∂x = x^{p-1}·y^{1-p}
                }
            }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], FromDoubles<T>((int[])x._shape.Clone(), i => dx[i]), engine);
    }

    // Adaptive windows: output o covers [⌊o·I/O⌋, ⌈(o+1)·I/O⌉).
    private static PoolWindow AdaptiveWindow(int input, int output, int o)
    {
        int start = (int)Math.Floor((double)o * input / output), end = (int)Math.Ceiling((double)(o + 1) * input / output);
        return new PoolWindow(start, end - start);
    }

    private (Tensor<T> Output, Tensor<int> Indices) AdaptivePool<T>(string opName, Tensor<T> x, int[] outputSize, WindowReduce mode)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        var spatial = x._shape.Skip(x.Rank - outputSize.Length).ToArray();
        return WindowPool(opName, x, outputSize, (_, axis, o) => AdaptiveWindow(spatial[axis], outputSize[axis], o), mode);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAdaptiveAvgPool1D<T>(Tensor<T> input, int outputSize)
        => AdaptivePool("TensorAdaptiveAvgPool1D", input, new[] { outputSize }, WindowReduce.Average).Output;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAdaptiveAvgPool3D<T>(Tensor<T> input, int[] outputSize)
        => AdaptivePool("TensorAdaptiveAvgPool3D", input, CheckSize(outputSize, 3), WindowReduce.Average).Output;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAdaptiveMaxPool1D<T>(Tensor<T> input, int outputSize)
        => AdaptivePool("TensorAdaptiveMaxPool1D", input, new[] { outputSize }, WindowReduce.Max).Output;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAdaptiveMaxPool3D<T>(Tensor<T> input, int[] outputSize)
        => AdaptivePool("TensorAdaptiveMaxPool3D", input, CheckSize(outputSize, 3), WindowReduce.Max).Output;

    /// <inheritdoc/>
    public virtual (Tensor<T> Output, Tensor<int> Indices) TensorAdaptiveMaxPoolWithIndices<T>(Tensor<T> input, int[] outputSize)
        => AdaptivePool("TensorAdaptiveMaxPoolWithIndices", input, outputSize ?? throw new ArgumentNullException(nameof(outputSize)), WindowReduce.Max);

    private static int[] CheckSize(int[] size, int dims)
    {
        if (size == null || size.Length != dims) throw new ArgumentException($"expected {dims} output sizes.", nameof(size));
        return size;
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Output, Tensor<int> Indices) TensorMaxPool1DWithIndices<T>(Tensor<T> input, int kernelSize,
        int stride = 0, int padding = 0, int dilation = 1, bool ceilMode = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be positive.");
        if (stride < 0) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be positive (0 means the kernel size).");
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation), "dilation must be positive.");
        // PyTorch: padding at most half the effective kernel, so no window holds only padding.
        if (padding < 0 || padding > ((kernelSize - 1) * dilation + 1) / 2)
            throw new ArgumentOutOfRangeException(nameof(padding), "padding must be between 0 and half the effective kernel size.");
        int s = stride <= 0 ? kernelSize : stride, length = input._shape[input.Rank - 1];
        double span = length + 2.0 * padding - dilation * (kernelSize - 1) - 1;
        int outLength = (int)(ceilMode ? Math.Ceiling(span / s) : Math.Floor(span / s)) + 1;
        // With ceil mode, the last window must start inside the (left-padded) input.
        if (ceilMode && (outLength - 1) * s >= length + padding) outLength--;
        if (outLength < 1) throw new ArgumentException($"input length {length} is too short for kernel {kernelSize} with dilation {dilation} and padding {padding}.", nameof(input));
        return WindowPool("TensorMaxPool1DWithIndices", input, new[] { outLength },
            (_, _, o) => new PoolWindow(o * s - padding, kernelSize, dilation), WindowReduce.Max);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLpPool<T>(Tensor<T> input, double power, int[] kernelSize, int[]? stride = null)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelSize == null || kernelSize.Length == 0) throw new ArgumentException("kernelSize is required.", nameof(kernelSize));
        var steps = stride ?? kernelSize;
        int dims = kernelSize.Length;
        if (steps.Length != dims) throw new ArgumentException($"stride needs {dims} entries, one per kernel axis.", nameof(stride));
        if (kernelSize.Any(k => k <= 0) || steps.Any(v => v <= 0)) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernel sizes and strides must be positive.");
        if (power == 0 || double.IsNaN(power)) throw new ArgumentOutOfRangeException(nameof(power), "power must be a non-zero number.");
        if (input.Rank < dims + 1) throw new ArgumentException($"lp pooling over {dims} axes needs at least {dims + 1} dimensions.", nameof(input));
        var spatial = input._shape.Skip(input.Rank - dims).ToArray();
        for (int d = 0; d < dims; d++)
            if (kernelSize[d] > spatial[d]) throw new ArgumentException($"kernel {kernelSize[d]} is larger than input axis {spatial[d]}.", nameof(kernelSize));
        var outputSize = spatial.Select((n, d) => (n - kernelSize[d]) / steps[d] + 1).ToArray();
        return WindowPool("TensorLpPool", input, outputSize, (_, d, o) => new PoolWindow(o * steps[d], kernelSize[d]), WindowReduce.Power, power).Output;
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Output, Tensor<int> Indices) TensorFractionalMaxPool<T>(Tensor<T> input, int[] kernelSize, int[] outputSize, int? seed = null)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelSize == null || outputSize == null || kernelSize.Length != outputSize.Length)
            throw new ArgumentException("kernelSize and outputSize must have one entry per pooled axis.");
        int dims = kernelSize.Length;
        var spatial = input._shape.Skip(input.Rank - dims).ToArray();
        for (int d = 0; d < dims; d++)
            if (outputSize[d] + kernelSize[d] - 1 > spatial[d])
                throw new ArgumentException($"output size {outputSize[d]} with kernel {kernelSize[d]} does not fit input size {spatial[d]}.");
        int planes = input.Length / spatial.Aggregate(1, (a, b) => a * b);
        // PyTorch's pseudo-random intervals: per plane and axis, u ~ U[0,1), α = (in - k)/(out - 1),
        // start_i = ⌊(i + u)α⌋ - ⌊uα⌋, the last window flush with the end.
        var rng = RandomSource(seed);
        var starts = new int[planes, dims][];
        for (int plane = 0; plane < planes; plane++)
            for (int d = 0; d < dims; d++)
            {
                int outN = outputSize[d], k = kernelSize[d], inN = spatial[d];
                double u = rng.NextDouble(), alpha = outN > 1 ? (inN - k) / (double)(outN - 1) : 0;
                var seq = new int[outN];
                for (int i = 0; i < outN - 1; i++) seq[i] = (int)(Math.Floor((i + u) * alpha) - Math.Floor(u * alpha));
                seq[outN - 1] = inN - k;
                starts[plane, d] = seq;
            }
        return WindowPool("TensorFractionalMaxPool", input, outputSize,
            (plane, d, o) => new PoolWindow(starts[plane, d][o], kernelSize[d]), WindowReduce.Max);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMaxUnpool<T>(Tensor<T> input, Tensor<int> indices, int[] outputSize)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (outputSize == null || outputSize.Length == 0) throw new ArgumentException("outputSize is required.", nameof(outputSize));
        if (!indices._shape.SequenceEqual(input._shape)) throw new ArgumentException("indices must match the input's shape.", nameof(indices));
        if (input.Rank < outputSize.Length) throw new ArgumentException($"unpooling {outputSize.Length} axes needs an input of rank ≥ {outputSize.Length}, got {input.Rank}.", nameof(input));
        // As in PyTorch, an index repeated within a plane keeps the last value written (and its gradient).
        int dims = outputSize.Length, inPlane = input._shape.Skip(input.Rank - dims).Aggregate(1, (a, b) => a * b);
        int outPlane = outputSize.Aggregate(1, (a, b) => a * b), planes = input.Length / inPlane;
        var idx = (indices.IsContiguous ? indices : indices.Contiguous()).AsSpan().ToArray();
        // Each input element lands at its plane's flat index; the rest stay zero. The scatter's adjoint is the gather.
        var source = Enumerable.Repeat(-1, planes * outPlane).ToArray();
        for (int plane = 0; plane < planes; plane++)
            for (int i = 0; i < inPlane; i++)
            {
                int target = idx[plane * inPlane + i];
                if (target < 0 || target >= outPlane) throw new ArgumentOutOfRangeException(nameof(indices), $"index {target} is outside the output plane.");
                source[plane * outPlane + target] = plane * inPlane + i;
            }
        var shape = input._shape.Take(input.Rank - dims).Concat(outputSize).ToArray();
        return IndexMap("TensorMaxUnpool", input, shape, source);
    }
}
