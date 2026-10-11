using AiDotNet.Tensors.Engines.Autodiff;
using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for PyTorch-parity elementwise ops that computed on the host (#1112). They run on backends implementing
/// <see cref="ITorchParityElementwiseBackend"/>, which today is CUDA only; HIP, Metal, OpenCL, Vulkan and WebGPU take the
/// recorded host fallback until #1124, #1125, #1126, #1127 and #1128 land. Under an active tape or graph recording they
/// defer to the CPU implementation, which records the backward (the convention of the engine's other parity
/// overrides); so do non-float precisions, mismatched binary shapes and argument errors the CPU reports.
/// </summary>
public partial class DirectGpuTensorEngine
{
    private Tensor<T>? TryParityUnaryOnDevice<T>(Tensor<T> input, TorchParityUnaryOp op, float p0, float p1)
    {
        if (input is null || input.Length == 0 || IsTapeActive<T>(op.ToString()) || !TryGetBackend(out var backend)
            || backend is not ITorchParityElementwiseBackend kernels || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = input.Length;
            using var x = GetOrAllocateBuffer(backend, input);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])input._shape.Clone(),
                output => kernels.TorchParityUnary(op, x.Buffer, output, n, p0, p1));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback("Tensor" + op, ex);
            return null;
        }
    }

    private Tensor<T>? TryParityBinaryOnDevice<T>(Tensor<T> a, Tensor<T> b, TorchParityBinaryOp op)
    {
        if (a is null || b is null || a.Length == 0 || !a._shape.AsSpan().SequenceEqual(b._shape) || IsTapeActive<T>(op.ToString())
            || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = a.Length;
            using var x = GetOrAllocateBuffer(backend, a);
            using var y = GetOrAllocateBuffer(backend, b);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])a._shape.Clone(),
                output => kernels.TorchParityBinary(op, x.Buffer, y.Buffer, output, n));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback("Tensor" + op, ex);
            return null;
        }
    }

    // Orthogonal polynomials: x and the degree tensor share a shape (the CPU requires it too). The CPU rejects an
    // infinite or > 2^20 degree with an exception; the degree tensor is checked here (a host read of n, which is normally a
    // small host-side tensor) and such a call stays on the CPU path so it throws the same exception.
    private Tensor<T>? TryPolynomialOnDevice<T>(Tensor<T> x, Tensor<T> degree, int kind, bool shifted, string opName)
    {
        if (x is null || degree is null || x.Length == 0 || !x._shape.AsSpan().SequenceEqual(degree._shape) || IsTapeActive<T>(opName)
            || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        // Row-major copies of strided views: the degree scan reads a span and the kernel reads both row-major.
        if (!x.IsContiguous) x = x.Contiguous();
        if (!degree.IsContiguous) degree = degree.Contiguous();
        var ops = MathHelper.GetNumericOperations<T>();
        foreach (var value in degree.AsSpan())
        {
            double d = ops.ToDouble(value);
            if (double.IsInfinity(d) || d > (1 << 20)) return null;
        }
        try
        {
            int n = x.Length;
            using var xs = GetOrAllocateBuffer(backend, x);
            using var ds = GetOrAllocateBuffer(backend, degree);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])x._shape.Clone(),
                output => kernels.TorchParityPolynomial(kind, shifted, xs.Buffer, ds.Buffer, output, n));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }

    // Creation ops (#1112 tier 5) build their result on the device: no host buffer, no upload. Argument validation
    // stays with the CPU implementation (the overrides call it first, or fall back to it), so errors are identical.
    private Tensor<T>? TryGenerateOnDevice<T>(string opName, int[] shape, int n,
        Action<ITorchParityElementwiseBackend, IDirectGpuBackend, IGpuBuffer> generate)
    {
        if (n == 0) return new Tensor<T>(shape);
        if (IsTapeActive<T>(opName) || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            return DispatchDeferredGpuOp<T>(backend, n, shape, output => generate(kernels, backend, output));
        }
        catch (System.Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }

    private Tensor<T>? TryRangeOnDevice<T>(string opName, double start, double step, int count)
        => TryGenerateOnDevice<T>(opName, new[] { count }, count, (k, _, output) => k.TorchParityRange(output, count, start, step, 1.0));

    private Tensor<T>? TryWindowOnDevice<T>(string opName, int length, bool periodic, int kind, double alpha, double beta)
    {
        if (length < 0) return null;   // the CPU rejects it
        int m = periodic ? length + 1 : length;
        return TryGenerateOnDevice<T>(opName, new[] { length }, length, (k, _, output) => k.TorchParityWindow(output, length, kind, m, alpha, beta));
    }

    private Tensor<T>? TryTriIndicesOnDevice<T>(string opName, int row, int col, int offset, bool lower)
    {
        if (row < 0 || col < 0) return null;   // the CPU rejects it
        if (row > (1 << 24) || col > (1 << 24)) return null;   // indices must stay exact in the float device buffer
        long total = 0;
        for (int r = 0; r < row; r++)
        {
            long edge = (long)r + offset;
            long lo = lower ? 0 : Math.Max(0, edge), hi = lower ? Math.Min(col - 1, edge) : col - 1;
            if (hi >= lo) total += hi - lo + 1;
        }
        if (2 * total > int.MaxValue) return null;
        int n = (int)total;
        return TryGenerateOnDevice<T>(opName, new[] { 2, n }, 2 * n, (k, _, output) => k.TorchParityTriIndices(output, row, col, offset, lower, n));
    }

    private Tensor<T>? TryFillLikeOnDevice<T>(string opName, Tensor<T> like, double value)
    {
        if (like is null) return null;   // the CPU throws ArgumentNullException
        int n = like.Length;
        return TryGenerateOnDevice<T>(opName, (int[])like._shape.Clone(), n, (_, backend, output) => backend.Fill(output, (float)value, n));
    }

    // all / any: the non-zero indicator on the device, then the engine's device min / max reduction (constants, so no
    // gradient, as on the CPU).
    private Tensor<T>? TryTruthinessOnDevice<T>(string opName, Tensor<T> tensor, int[]? axes, bool keepDims, bool all)
    {
        if (tensor is null || tensor.Length == 0) return null;
        var indicator = TryParityUnaryOnDevice(tensor, TorchParityUnaryOp.NonZero, 0f, 0f);
        if (indicator is null) return null;
        var reduceAxes = axes ?? AllAxes(tensor.Rank);
        Tensor<T> reduced;
        using (new NoGradScope<T>())
            reduced = all ? TensorAmin(indicator, reduceAxes, keepDims) : ReduceMax(indicator, reduceAxes, keepDims);
        return reduced;
    }

    private Tensor<T>? TryDiagflatOnDevice<T>(Tensor<T> tensor, int offset)
    {
        if (tensor is null || IsTapeActive<T>("TensorDiagflat")) return null;
        int n = tensor.Length;
        long sizeLong = (long)n + Math.Abs((long)offset);
        if (n == 0 || sizeLong * sizeLong > int.MaxValue) return null;
        int size = (int)sizeLong;
        if (!TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
            using var x = GetOrAllocateBuffer(backend, source);
            return DispatchDeferredGpuOp<T>(backend, size * size, new[] { size, size },
                output => kernels.TorchParityDiagflat(x.Buffer, output, n, size, offset));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorDiagflat", ex);
            return null;
        }
    }

    // Invalid arguments return null so the CPU implementation raises its exception.
    private Tensor<T>? TryDiagonalScatterOnDevice<T>(Tensor<T> input, Tensor<T> src, int offset, int dim1, int dim2)
    {
        if (input is null || src is null || input.Rank < 2 || IsTapeActive<T>("TensorDiagonalScatter")) return null;
        int rank = input.Rank;
        int d1 = dim1 < 0 ? dim1 + rank : dim1, d2 = dim2 < 0 ? dim2 + rank : dim2;
        if (d1 < 0 || d1 >= rank || d2 < 0 || d2 >= rank || d1 == d2 || rank - 2 > 8) return null;
        var shape = input._shape;
        var strides = new int[rank];
        for (int k = rank - 1, s = 1; k >= 0; s *= shape[k], k--) strides[k] = s;
        int diagonal = Math.Max(0, Math.Min(shape[d1] - Math.Max(0, -offset), shape[d2] - Math.Max(0, offset)));
        var others = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Where(System.Linq.Enumerable.Range(0, rank), k => k != d1 && k != d2));
        int otherCount = 1;
        foreach (int k in others) otherCount *= shape[k];
        if (src.Length != otherCount * diagonal || input.Length == 0) return null;
        if (!TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            var inputSource = input.IsContiguous ? input : input.Contiguous();
            var srcSource = src.IsContiguous ? src : src.Contiguous();
            using var x = GetOrAllocateBuffer(backend, inputSource);
            using var s = GetOrAllocateBuffer(backend, srcSource);
            int n = input.Length, count = src.Length;
            var otherExtents = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(others, k => shape[k]));
            var otherStrides = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(others, k => strides[k]));
            return DispatchDeferredGpuOp<T>(backend, n, (int[])shape.Clone(), output =>
            {
                backend.Copy(x.Buffer, output, n);
                kernels.TorchParityDiagonalScatter(s.Buffer, output, count, otherExtents, otherStrides,
                    strides[d1], strides[d2], Math.Max(0, -offset), Math.Max(0, offset), diagonal);
            });
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorDiagonalScatter", ex);
            return null;
        }
    }

    // msort: transpose [extent, inner] to rows, bitonic-sort each row (torch order, NaN last), transpose back.
    private Tensor<T>? TryMsortOnDevice<T>(Tensor<T> tensor)
    {
        if (tensor is null || tensor.Rank == 0 || tensor.Length == 0 || typeof(T) != typeof(float)
            || IsTapeActive<T>("TensorMsort") || !TryGetBackend(out var backend))
            return null;
        try
        {
            int extent = tensor._shape[0], inner = tensor.Length / extent;
            var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
            using var x = GetOrAllocateBuffer(backend, source);
            using var rows = AllocateOutputBuffer(backend, tensor.Length);
            backend.Transpose(x.Buffer, rows.Buffer, extent, inner);
            var (sorted, indices) = BitonicSortRowsToBuffers(backend, rows.Buffer, inner, extent, descending: false);
            using (sorted)
            using (indices)
                return DispatchDeferredGpuOp<T>(backend, tensor.Length, (int[])tensor._shape.Clone(),
                    output => backend.Transpose(sorted.Buffer, output, inner, extent));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorMsort", ex);
            return null;
        }
    }

    // quantile / nanquantile: the reduced axis moved last ([outer, extent, inner] -> [outer, inner, extent]), each row
    // bitonic-sorted (NaN last), then one thread per row reads the order statistics. Invalid q / dim go to the CPU.
    private Tensor<T>? TryQuantileOnDevice<T>(string opName, Tensor<T> tensor, double q, int? dim, bool keepDim,
        QuantileInterpolation interpolation, bool ignoreNan)
    {
        if (tensor is null || tensor.Length == 0 || double.IsNaN(q) || q < 0 || q > 1 || typeof(T) != typeof(float)
            || IsTapeActive<T>(opName) || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels)
            return null;
        int rank = Math.Max(1, tensor.Rank);
        int d = 0;
        if (dim is not null)
        {
            d = dim.Value < 0 ? dim.Value + tensor.Rank : dim.Value;
            if (d < 0 || d >= tensor.Rank) return null;
        }
        int[] shape = dim is null ? new[] { tensor.Length } : tensor._shape;
        int extent = shape[d], outer = 1, inner = 1;
        for (int k = 0; k < d; k++) outer *= shape[k];
        for (int k = d + 1; k < shape.Length; k++) inner *= shape[k];
        int rowsCount = outer * inner;
        int[] outShape = keepDim
            ? (dim is null ? System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Repeat(1, tensor.Rank))
                : System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(shape, (s, k) => k == d ? 1 : s)))
            : System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Where(shape, (_, k) => k != d));
        try
        {
            var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
            using var x = GetOrAllocateBuffer(backend, source);
            using var rows = AllocateOutputBuffer(backend, tensor.Length);
            if (inner == 1) backend.Copy(x.Buffer, rows.Buffer, tensor.Length);
            else backend.Permute(x.Buffer, rows.Buffer, new[] { outer, extent, inner }, new[] { 0, 2, 1 });
            var (sorted, indices) = BitonicSortRowsToBuffers(backend, rows.Buffer, rowsCount, extent, descending: false);
            using (sorted)
            using (indices)
                return DispatchDeferredGpuOp<T>(backend, rowsCount, outShape,
                    output => kernels.TorchParityQuantileRows(sorted.Buffer, output, rowsCount, extent, q, (int)interpolation, ignoreNan));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }
}
