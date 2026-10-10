using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Launches for the PyTorch-parity elementwise kernels (Kernels/CudaTorchParityElementwiseKernels).
public sealed partial class CudaBackend : ITorchParityElementwiseBackend
{
    private static string UnaryKernelName(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Celu => "parity_celu",
        TorchParityUnaryOp.Hardtanh => "parity_hardtanh",
        TorchParityUnaryOp.LogSigmoid => "parity_logsigmoid",
        TorchParityUnaryOp.Softsign => "parity_softsign",
        TorchParityUnaryOp.Logit => "parity_logit",
        TorchParityUnaryOp.Sinc => "parity_sinc",
        TorchParityUnaryOp.Erf => "parity_erf",
        TorchParityUnaryOp.BesselJ0 => "parity_bessel_j0",
        TorchParityUnaryOp.BesselJ1 => "parity_bessel_j1",
        TorchParityUnaryOp.BesselY0 => "parity_bessel_y0",
        TorchParityUnaryOp.BesselY1 => "parity_bessel_y1",
        TorchParityUnaryOp.ModifiedBesselI0 => "parity_bessel_i0",
        TorchParityUnaryOp.ModifiedBesselI1 => "parity_bessel_i1",
        TorchParityUnaryOp.Erfcx => "parity_erfcx",
        TorchParityUnaryOp.Ndtr => "parity_ndtr",
        TorchParityUnaryOp.Ndtri => "parity_ndtri",
        TorchParityUnaryOp.Entr => "parity_entr",
        TorchParityUnaryOp.Signbit => "parity_signbit",
        TorchParityUnaryOp.IsPosInf => "parity_isposinf",
        TorchParityUnaryOp.IsNegInf => "parity_isneginf",
        TorchParityUnaryOp.LogNdtr => "parity_log_ndtr",
        TorchParityUnaryOp.ModifiedBesselK0 => "parity_bessel_k0",
        TorchParityUnaryOp.ModifiedBesselK1 => "parity_bessel_k1",
        TorchParityUnaryOp.ScaledModifiedBesselK0 => "parity_bessel_k0e",
        TorchParityUnaryOp.ScaledModifiedBesselK1 => "parity_bessel_k1e",
        TorchParityUnaryOp.SphericalBesselJ0 => "parity_spherical_bessel_j0",
        TorchParityUnaryOp.AiryAi => "parity_airy_ai",
        TorchParityUnaryOp.Mvlgamma => "parity_mvlgamma",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private static string BinaryKernelName(TorchParityBinaryOp op) => op switch
    {
        TorchParityBinaryOp.Fmax => "parity_fmax",
        TorchParityBinaryOp.Fmin => "parity_fmin",
        TorchParityBinaryOp.Heaviside => "parity_heaviside",
        TorchParityBinaryOp.GreaterEqual => "parity_greater_equal",
        TorchParityBinaryOp.LessEqual => "parity_less_equal",
        TorchParityBinaryOp.Igamma => "parity_igamma",
        TorchParityBinaryOp.Igammac => "parity_igammac",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    unsafe void ITorchParityElementwiseBackend.TorchParityUnary(
        TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
    {
        if (n <= 0) return;
        string name = UnaryKernelName(op);
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        using var _ = PushContext();
        IntPtr x = input.Handle, y = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &x; args[1] = &y; args[2] = &n; args[3] = &p0; args[4] = &p1;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityPolynomial(
        int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        if (!_kernelCache.TryGetValue("parity_polynomial", out var polynomialKernel))
            throw new InvalidOperationException("CUDA kernel not found: parity_polynomial");
        using var _ = PushContext();
        IntPtr px = x.Handle, pd = degree.Handle, py = output.Handle;
        int shift = shifted ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &px; args[1] = &pd; args[2] = &py; args[3] = &n; args[4] = &kind; args[5] = &shift;
        LaunchKernel(polynomialKernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    private IntPtr ParityKernel(string name)
        => _kernelCache.TryGetValue(name, out var found) ? found : throw new InvalidOperationException($"CUDA kernel not found: {name}");

    private static uint ParityGrid(int n) => (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);

    unsafe void ITorchParityElementwiseBackend.TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
    {
        if (n <= 0) return;
        IntPtr rangeKernel = ParityKernel("parity_range");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &py; args[1] = &n; args[2] = &start; args[3] = &span; args[4] = &denominator;
        LaunchKernel(rangeKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityLogspace(
        IGpuBuffer output, int n, double start, double span, double denominator, double logBase, bool single)
    {
        if (n <= 0) return;
        IntPtr logspaceKernel = ParityKernel("parity_logspace");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        int one = single ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &py; args[1] = &n; args[2] = &start; args[3] = &span; args[4] = &denominator; args[5] = &logBase; args[6] = &one;
        LaunchKernel(logspaceKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
    {
        if (n <= 0) return;
        IntPtr windowKernel = ParityKernel("parity_window");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &py; args[1] = &n; args[2] = &kind; args[3] = &m; args[4] = &alpha; args[5] = &beta;
        LaunchKernel(windowKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
    {
        if (row <= 0 || total <= 0) return;
        IntPtr triKernel = ParityKernel("parity_tri_indices");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        int isLower = lower ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &py; args[1] = &row; args[2] = &col; args[3] = &offset; args[4] = &isLower; args[5] = &total;
        LaunchKernel(triKernel, ParityGrid(row), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityBinary(
        TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        string name = BinaryKernelName(op);
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        using var _ = PushContext();
        IntPtr pa = a.Handle, pb = b.Handle, y = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pa; args[1] = &pb; args[2] = &y; args[3] = &n;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }
}
