using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Launches for the PyTorch-parity elementwise kernels (VulkanTorchParityElementwiseKernels), argument for argument with
// CudaBackend.TorchParityElementwise (#1127). Push constants are 32-bit words; doubles and 64-bit values go as lo / hi
// pairs, first, so each pair is 8-byte aligned (see VulkanTorchParityElementwiseKernels).
public sealed unsafe partial class VulkanBackend : ITorchParityElementwiseBackend
{
    private void ParityDispatch(string kernel, int threads, IGpuBuffer[] buffers, params uint[] push)
    {
        if (threads <= 0) return;
        GlslNaryOp(VulkanTorchParityElementwiseKernels.Source(kernel), buffers, threads, push,
            VulkanTorchParityElementwiseKernels.LocalSize);
    }

    private static void ParityRequire(IGpuBuffer buffer, long requiredElements, string parameter, string formula)
    {
        if (buffer is null) throw new ArgumentNullException(parameter);
        if (buffer.Size < requiredElements)
            throw new ArgumentException(
                $"{parameter} buffer too small: capacity={buffer.Size} elements, required={requiredElements} ({formula}).", parameter);
    }

    private static uint Lo(long bits) => unchecked((uint)bits);
    private static uint Hi(long bits) => unchecked((uint)(bits >> 32));
    private static uint DLo(double v) => Lo(BitConverter.DoubleToInt64Bits(v));
    private static uint DHi(double v) => Hi(BitConverter.DoubleToInt64Bits(v));
    private static uint SLo(ulong seed) => unchecked((uint)seed);
    private static uint SHi(ulong seed) => unchecked((uint)(seed >> 32));
    private static uint U(int v) => unchecked((uint)v);
    private static uint F(float v) => BitConverter.ToUInt32(BitConverter.GetBytes(v), 0);

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

    void ITorchParityElementwiseBackend.TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
    {
        if (n <= 0) return;
        ParityRequire(input, n, nameof(input), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch(UnaryKernelName(op), n, new[] { input, output }, U(n), F(p0), F(p1));
    }

    void ITorchParityElementwiseBackend.TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        ParityRequire(a, n, nameof(a), "n");
        ParityRequire(b, n, nameof(b), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch(BinaryKernelName(op), n, new[] { a, b, output }, U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        ParityRequire(x, n, nameof(x), "n");
        ParityRequire(degree, n, nameof(degree), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_polynomial", n, new[] { x, degree, output }, U(n), U(kind), shifted ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
    {
        if (n <= 0) return;
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_range", n, new[] { output }, DLo(start), DHi(start), DLo(span), DHi(span), DLo(denominator), DHi(denominator), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator, double logBase,
        bool single)
    {
        if (n <= 0) return;
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_logspace", n, new[] { output }, DLo(start), DHi(start), DLo(span), DHi(span), DLo(denominator),
            DHi(denominator), DLo(logBase), DHi(logBase), U(n), single ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
    {
        if (n <= 0) return;
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_window", n, new[] { output }, DLo(alpha), DHi(alpha), DLo(beta), DHi(beta), U(n), U(kind), U(m));
    }

    void ITorchParityElementwiseBackend.TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
    {
        if (row <= 0 || total <= 0) return;
        ParityRequire(output, 2L * total, nameof(output), "2*total");
        ParityDispatch("parity_tri_indices", row, new[] { output }, U(row), U(col), U(offset), lower ? 1u : 0u, U(total));
    }

    void ITorchParityElementwiseBackend.TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
    {
        if (size <= 0) return;
        int total = checked(size * size);
        ParityRequire(output, total, nameof(output), "size*size");
        ParityRequire(input, n, nameof(input), "n");
        ParityDispatch("parity_diagflat", total, new[] { input, output }, U(total), U(size), U(offset), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents, int[] otherStrides,
        int stride1, int stride2, int start1, int start2, int diagonal)
    {
        if (otherExtents is null) throw new ArgumentNullException(nameof(otherExtents));
        if (otherStrides is null) throw new ArgumentNullException(nameof(otherStrides));
        if (otherStrides.Length != otherExtents.Length)
            throw new ArgumentException("otherStrides must hold one stride per other axis.", nameof(otherStrides));
        if (count <= 0) return;
        if (otherExtents.Length > 8) throw new ArgumentOutOfRangeException(nameof(otherExtents), "at most 8 other axes.");
        if (diagonal <= 0) throw new ArgumentOutOfRangeException(nameof(diagonal));
        ParityRequire(src, count, nameof(src), "count");
        long furthest = (long)(diagonal - 1 + start1) * stride1 + (long)(diagonal - 1 + start2) * stride2;
        for (int i = 0; i < otherExtents.Length; i++) furthest += (long)(otherExtents[i] - 1) * otherStrides[i];
        ParityRequire(output, furthest + 1, nameof(output), "furthest diagonal offset + 1");
        var geometry = new int[22];
        geometry[0] = otherExtents.Length;
        for (int i = 0; i < otherExtents.Length; i++) { geometry[1 + i] = otherExtents[i]; geometry[9 + i] = otherStrides[i]; }
        geometry[17] = stride1; geometry[18] = stride2; geometry[19] = start1; geometry[20] = start2; geometry[21] = diagonal;
        using var g = AllocateIntBuffer(geometry);
        ParityDispatch("parity_diagonal_scatter", count, new[] { src, output, g }, U(count));
    }

    void ITorchParityElementwiseBackend.TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q, int mode, bool ignoreNan)
    {
        if (rows <= 0) return;
        ParityRequire(sorted, (long)rows * length, nameof(sorted), "rows*length");
        ParityRequire(output, rows, nameof(output), "rows");
        ParityDispatch("parity_quantile_rows", rows, new[] { sorted, output }, DLo(q), DHi(q), U(rows), U(length), U(mode), ignoreNan ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityFlagReset(IGpuBuffer flag)
    {
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_flag_reset", 1, new[] { flag }, 1u);
    }

    void ITorchParityElementwiseBackend.TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
    {
        if (n <= 0) return;
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_sample", n, new[] { output }, DLo(a), DHi(a), DLo(b), DHi(b), SLo(seed), SHi(seed), U(n), U(kind));
    }

    void ITorchParityElementwiseBackend.TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        ParityRequire(probabilities, n, nameof(probabilities), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_bernoulli", n, new[] { probabilities, output, flag }, SLo(seed), SHi(seed), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        ParityRequire(rates, n, nameof(rates), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_poisson", n, new[] { rates, output, flag }, SLo(seed), SHi(seed), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        ParityRequire(counts, n, nameof(counts), "n");
        ParityRequire(probabilities, n, nameof(probabilities), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_binomial", n, new[] { counts, probabilities, output, flag }, SLo(seed), SHi(seed), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows, int categories,
        int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
    {
        if (rows <= 0) return;
        ParityRequire(weights, (long)rows * categories, nameof(weights), "rows*categories");
        ParityRequire(scratch, 2L * rows * categories, nameof(scratch), "2*rows*categories (double prefix sums)");
        ParityRequire(output, (long)rows * numSamples, nameof(output), "rows*numSamples");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_multinomial", rows, new[] { weights, scratch, output, flag }, SLo(seed), SHi(seed), U(rows), U(categories),
            U(numSamples), replacement ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a, double sat,
        double b, bool alpha, ulong seed)
    {
        if (n <= 0) return;
        ParityRequire(scale, n, nameof(scale), "n");
        if (alpha) ParityRequire(offset, n, nameof(offset), "n");
        ParityDispatch("parity_dropout_affine", n, new[] { scale, offset }, DLo(p), DHi(p), DLo(a), DHi(a), DLo(sat), DHi(sat),
            DLo(b), DHi(b), SLo(seed), SHi(seed), U(n), U(block), alpha ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper, bool training, ulong seed)
    {
        if (n <= 0) return;
        ParityRequire(input, n, nameof(input), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_rrelu", n, new[] { input, output }, DLo(lower), DHi(lower), DLo(upper), DHi(upper), SLo(seed), SHi(seed),
            U(n), training ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
    {
        if (n <= 0) return;
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_random_keys", n, new[] { output }, SLo(seed), SHi(seed), U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag)
    {
        if (n <= 0) return;
        ParityRequire(a, n, nameof(a), "n");
        ParityRequire(b, n, nameof(b), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_int_binary", n, new[] { a, b, output, flag }, U(n), U(op));
    }

    void ITorchParityElementwiseBackend.TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        ParityRequire(input, n, nameof(input), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityDispatch("parity_int_not", n, new[] { input, output }, U(n));
    }

    void ITorchParityElementwiseBackend.TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        ParityRequire(input, n, nameof(input), "n");
        ParityRequire(output, n, nameof(output), "n");
        const int block = VulkanTorchParityElementwiseKernels.LocalSize;
        int blocks = (n + block - 1) / block;
        IGpuBuffer? sums = blocks > 1 ? AllocateBuffer(blocks) : null;
        IGpuBuffer? carries = blocks > 1 ? AllocateBuffer(blocks) : null;
        try
        {
            // Without block sums the output fills the unused binding (a descriptor cannot be empty).
            ParityDispatch("parity_scan_block", blocks * block, new[] { input, output, sums ?? output }, U(n), sums is null ? 0u : 1u);
            if (sums is null || carries is null) return;
            ((ITorchParityElementwiseBackend)this).TorchParityInclusiveScan(sums, carries, blocks);
            ParityDispatch("parity_scan_add", n, new[] { output, carries }, U(n));
        }
        finally
        {
            sums?.Dispose();
            carries?.Dispose();
        }
    }

    void ITorchParityElementwiseBackend.TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n, int size, int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (n <= 0) return;
        ParityRequire(input, n, nameof(input), "n");
        ParityRequire(positions, n, nameof(positions), "n");
        ParityRequire(output, (long)size * shape.Length, nameof(output), "size*rank");
        if (shape.Length > 8) throw new ArgumentOutOfRangeException(nameof(shape), "at most 8 axes.");
        var packed = new int[9];
        packed[0] = shape.Length;
        for (int i = 0; i < shape.Length; i++) packed[1 + i] = shape[i];
        using var s = AllocateIntBuffer(packed);
        ParityDispatch("parity_nonzero_static_scatter", n, new[] { input, positions, output, s }, U(n), U(size));
    }

    void ITorchParityElementwiseBackend.TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride, long extent,
        bool validate, IGpuBuffer flag)
    {
        if (n <= 0) return;
        ParityRequire(indices, n, nameof(indices), "n");
        ParityRequire(output, n, nameof(output), "n");
        ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
        ParityDispatch("parity_unravel_axis", n, new[] { indices, output, flag }, Lo(total), Hi(total), Lo(stride), Hi(stride), Lo(extent),
            Hi(extent), U(n), validate ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j, int length, int sliceCount)
    {
        if (rowLen <= 0) return;
        if (sliceCount > rowLen) throw new ArgumentOutOfRangeException(nameof(sliceCount), "more slices than order slots.");
        ParityRequire(order, rowLen, nameof(order), "rowLen");
        ParityRequire(slices, (long)sliceCount * length, nameof(slices), "sliceCount*length");
        ParityDispatch("parity_slice_bitonic_step", rowLen, new[] { slices, order }, U(rowLen), U(k), U(j), U(length));
    }

    void ITorchParityElementwiseBackend.TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
    {
        if (n <= 0) return;
        ParityRequire(order, n, nameof(order), "n");
        ParityRequire(flags, n, nameof(flags), "n");
        ParityRequire(slices, (long)n * length, nameof(slices), "n*length");
        ParityDispatch("parity_slice_first_flags", n, new[] { slices, order, flags }, U(n), U(length));
    }

    void ITorchParityElementwiseBackend.TorchParitySliceGatherUnique(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, IGpuBuffer positions,
        IGpuBuffer output, int n, int length, int outputRows)
    {
        int total = checked(n * length);
        ParityRequire(output, (long)outputRows * length, nameof(output), "outputRows*length");
        if (total <= 0) return;
        ParityRequire(slices, (long)n * length, nameof(slices), "n*length");
        ParityRequire(order, n, nameof(order), "n");
        ParityRequire(flags, n, nameof(flags), "n");
        ParityRequire(positions, n, nameof(positions), "n");
        ParityDispatch("parity_slice_gather_unique", total, new[] { slices, order, flags, positions, output }, U(total), U(length));
    }

    void ITorchParityElementwiseBackend.TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches, int channels,
        int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        ParityRequire(input, (long)batches * channels * depth * height * width, nameof(input), "N*C*D*H*W");
        ParityRequire(grid, 3L * batches * points, nameof(grid), "N*points*3");
        ParityRequire(output, (long)batches * channels * points, nameof(output), "N*C*points");
        ParityDispatch("parity_grid_sample_3d", threads, new[] { input, grid, output }, U(batches), U(channels), U(depth), U(height), U(width),
            U(points), U(mode), U(padding), alignCorners ? 1u : 0u);
    }

    void ITorchParityElementwiseBackend.TorchParityGridSample3DBackward(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad, IGpuBuffer inputGrad,
        IGpuBuffer gridGrad, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        long volume = (long)batches * channels * depth * height * width;
        ParityRequire(input, volume, nameof(input), "N*C*D*H*W");
        ParityRequire(inputGrad, volume, nameof(inputGrad), "N*C*D*H*W");
        ParityRequire(grid, 3L * batches * points, nameof(grid), "N*points*3");
        ParityRequire(gridGrad, 3L * batches * points, nameof(gridGrad), "N*points*3");
        ParityRequire(outputGrad, (long)batches * channels * points, nameof(outputGrad), "N*C*points");
        ParityDispatch("parity_grid_sample_3d_backward", threads, new[] { input, grid, outputGrad, inputGrad, gridGrad }, U(batches), U(channels),
            U(depth), U(height), U(width), U(points), U(mode), U(padding), alignCorners ? 1u : 0u);
    }
}