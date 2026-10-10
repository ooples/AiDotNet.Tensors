using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for PyTorch-parity elementwise ops that computed on the host (#1112). Under an active tape or graph
/// recording they defer to the CPU implementation, which records the backward (the convention of the engine's other
/// parity overrides); so do non-float precisions, mismatched binary shapes and backends without the kernels.
/// </summary>
public partial class DirectGpuTensorEngine
{
    private Tensor<T>? TryParityUnaryOnDevice<T>(Tensor<T> input, TorchParityUnaryOp op, float p0, float p1)
    {
        if (input.Length == 0 || IsTapeActive<T>(op.ToString()) || !TryGetBackend(out var backend)
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
        if (a.Length == 0 || !a._shape.AsSpan().SequenceEqual(b._shape) || IsTapeActive<T>(op.ToString())
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
        if (x.Length == 0 || !x._shape.AsSpan().SequenceEqual(degree._shape) || IsTapeActive<T>(opName)
            || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
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
}
