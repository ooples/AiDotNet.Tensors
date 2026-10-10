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
        TorchParityUnaryOp.Angle => "parity_angle",
        TorchParityUnaryOp.NonZero => "parity_nonzero",
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
        TorchParityBinaryOp.FloorDivide => "parity_floor_divide",
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

    unsafe void ITorchParityElementwiseBackend.TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
    {
        int total = size * size;
        if (total <= 0) return;
        IntPtr diagflatKernel = ParityKernel("parity_diagflat");
        using var _ = PushContext();
        IntPtr px = input.Handle, py = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &px; args[1] = &py; args[2] = &total; args[3] = &size; args[4] = &offset; args[5] = &n;
        LaunchKernel(diagflatKernel, ParityGrid(total), DefaultBlockSize, args);
    }

    // Mirrors the kernel's by-value ParityDiagonalGeometry.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private unsafe struct ParityDiagonalGeometry
    {
        public int OtherRank;
        public fixed int OtherExtent[8];
        public fixed int OtherStride[8];
        public int Stride1, Stride2, Start1, Start2, Diagonal;
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count,
        int[] otherExtents, int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
    {
        if (count <= 0) return;
        if (otherExtents.Length > 8) throw new ArgumentOutOfRangeException(nameof(otherExtents), "at most 8 other axes.");
        IntPtr scatterKernel = ParityKernel("parity_diagonal_scatter");
        var geometry = new ParityDiagonalGeometry
        {
            OtherRank = otherExtents.Length, Stride1 = stride1, Stride2 = stride2, Start1 = start1, Start2 = start2, Diagonal = diagonal,
        };
        for (int k = 0; k < otherExtents.Length; k++) { geometry.OtherExtent[k] = otherExtents[k]; geometry.OtherStride[k] = otherStrides[k]; }
        using var _ = PushContext();
        IntPtr ps = src.Handle, py = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &ps; args[1] = &py; args[2] = &count; args[3] = &geometry;
        LaunchKernel(scatterKernel, ParityGrid(count), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length,
        double q, int mode, bool ignoreNan)
    {
        if (rows <= 0) return;
        IntPtr quantileKernel = ParityKernel("parity_quantile_rows");
        using var _ = PushContext();
        IntPtr ps = sorted.Handle, py = output.Handle;
        int ignore = ignoreNan ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &ps; args[1] = &py; args[2] = &rows; args[3] = &length; args[4] = &q; args[5] = &mode; args[6] = &ignore;
        LaunchKernel(quantileKernel, ParityGrid(rows), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityFlagReset(IGpuBuffer flag)
    {
        IntPtr resetKernel = ParityKernel("parity_flag_reset");
        using var _ = PushContext();
        IntPtr pf = flag.Handle;
        void** args = stackalloc void*[1];
        args[0] = &pf;
        LaunchKernel(resetKernel, 1, 1, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
    {
        if (n <= 0) return;
        IntPtr sampleKernel = ParityKernel("parity_sample");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &py; args[1] = &n; args[2] = &kind; args[3] = &a; args[4] = &b; args[5] = &seed;
        LaunchKernel(sampleKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        IntPtr bernoulliKernel = ParityKernel("parity_bernoulli");
        using var _ = PushContext();
        IntPtr pp = probabilities.Handle, py = output.Handle, pf = flag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pp; args[1] = &py; args[2] = &n; args[3] = &seed; args[4] = &pf;
        LaunchKernel(bernoulliKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        IntPtr poissonKernel = ParityKernel("parity_poisson");
        using var _ = PushContext();
        IntPtr pr = rates.Handle, py = output.Handle, pf = flag.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pr; args[1] = &py; args[2] = &n; args[3] = &seed; args[4] = &pf;
        LaunchKernel(poissonKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n,
        ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        IntPtr binomialKernel = ParityKernel("parity_binomial");
        using var _ = PushContext();
        IntPtr pc = counts.Handle, pp = probabilities.Handle, py = output.Handle, pf = flag.Handle;
        void** args = stackalloc void*[6];
        args[0] = &pc; args[1] = &pp; args[2] = &py; args[3] = &n; args[4] = &seed; args[5] = &pf;
        LaunchKernel(binomialKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows,
        int categories, int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
    {
        if (rows <= 0) return;
        IntPtr multinomialKernel = ParityKernel("parity_multinomial");
        using var _ = PushContext();
        IntPtr pw = weights.Handle, ps = scratch.Handle, py = output.Handle, pf = flag.Handle;
        int withReplacement = replacement ? 1 : 0;
        void** args = stackalloc void*[9];
        args[0] = &pw; args[1] = &ps; args[2] = &py; args[3] = &rows; args[4] = &categories; args[5] = &numSamples;
        args[6] = &withReplacement; args[7] = &seed; args[8] = &pf;
        LaunchKernel(multinomialKernel, ParityGrid(rows), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p,
        double a, double sat, double b, bool alpha, ulong seed)
    {
        if (n <= 0) return;
        IntPtr dropoutKernel = ParityKernel("parity_dropout_affine");
        using var _ = PushContext();
        IntPtr pscale = scale.Handle, poffset = offset.Handle;
        int isAlpha = alpha ? 1 : 0;
        void** args = stackalloc void*[10];
        args[0] = &pscale; args[1] = &poffset; args[2] = &n; args[3] = &block; args[4] = &p; args[5] = &a; args[6] = &sat;
        args[7] = &b; args[8] = &isAlpha; args[9] = &seed;
        LaunchKernel(dropoutKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper,
        bool training, ulong seed)
    {
        if (n <= 0) return;
        IntPtr rreluKernel = ParityKernel("parity_rrelu");
        using var _ = PushContext();
        IntPtr px = input.Handle, py = output.Handle;
        int isTraining = training ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &px; args[1] = &py; args[2] = &n; args[3] = &lower; args[4] = &upper; args[5] = &isTraining; args[6] = &seed;
        LaunchKernel(rreluKernel, ParityGrid(n), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
    {
        if (n <= 0) return;
        IntPtr keysKernel = ParityKernel("parity_random_keys");
        using var _ = PushContext();
        IntPtr py = output.Handle;
        void** args = stackalloc void*[3];
        args[0] = &py; args[1] = &n; args[2] = &seed;
        LaunchKernel(keysKernel, ParityGrid(n), DefaultBlockSize, args);
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
