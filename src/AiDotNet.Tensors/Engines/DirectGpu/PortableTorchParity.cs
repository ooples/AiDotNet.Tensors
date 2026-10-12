// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="ITorchParityElementwiseBackend"/> over the FP32 kernels of <see cref="TorchParityF32Kernels"/> (#1125,
/// #1128), for backends without 64-bit floats: each member validates as CUDA's launcher does and launches the kernel
/// of the same contract through the backend's launcher.
/// </summary>
internal static class PortableTorchParity
{
    public static string UnaryKernel(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Celu => "tp_celu",
        TorchParityUnaryOp.Hardtanh => "tp_hardtanh",
        TorchParityUnaryOp.LogSigmoid => "tp_logsigmoid",
        TorchParityUnaryOp.Softsign => "tp_softsign",
        TorchParityUnaryOp.Logit => "tp_logit",
        TorchParityUnaryOp.Sinc => "tp_sinc",
        TorchParityUnaryOp.Erf => "tp_erf",
        TorchParityUnaryOp.BesselJ0 => "tp_bessel_j0",
        TorchParityUnaryOp.BesselJ1 => "tp_bessel_j1",
        TorchParityUnaryOp.BesselY0 => "tp_bessel_y0",
        TorchParityUnaryOp.BesselY1 => "tp_bessel_y1",
        TorchParityUnaryOp.ModifiedBesselI0 => "tp_bessel_i0",
        TorchParityUnaryOp.ModifiedBesselI1 => "tp_bessel_i1",
        TorchParityUnaryOp.Erfcx => "tp_erfcx",
        TorchParityUnaryOp.Ndtr => "tp_ndtr",
        TorchParityUnaryOp.Ndtri => "tp_ndtri",
        TorchParityUnaryOp.Entr => "tp_entr",
        TorchParityUnaryOp.Signbit => "tp_signbit",
        TorchParityUnaryOp.IsPosInf => "tp_isposinf",
        TorchParityUnaryOp.IsNegInf => "tp_isneginf",
        TorchParityUnaryOp.LogNdtr => "tp_log_ndtr",
        TorchParityUnaryOp.ModifiedBesselK0 => "tp_bessel_k0",
        TorchParityUnaryOp.ModifiedBesselK1 => "tp_bessel_k1",
        TorchParityUnaryOp.ScaledModifiedBesselK0 => "tp_bessel_k0e",
        TorchParityUnaryOp.ScaledModifiedBesselK1 => "tp_bessel_k1e",
        TorchParityUnaryOp.SphericalBesselJ0 => "tp_spherical_bessel_j0",
        TorchParityUnaryOp.AiryAi => "tp_airy_ai",
        TorchParityUnaryOp.Mvlgamma => "tp_mvlgamma",
        TorchParityUnaryOp.Angle => "tp_angle",
        TorchParityUnaryOp.NonZero => "tp_nonzero",
        _ => throw new System.ArgumentOutOfRangeException(nameof(op)),
    };

    public static string BinaryKernel(TorchParityBinaryOp op) => op switch
    {
        TorchParityBinaryOp.Fmax => "tp_fmax",
        TorchParityBinaryOp.Fmin => "tp_fmin",
        TorchParityBinaryOp.Heaviside => "tp_heaviside",
        TorchParityBinaryOp.GreaterEqual => "tp_greater_equal",
        TorchParityBinaryOp.LessEqual => "tp_less_equal",
        TorchParityBinaryOp.Igamma => "tp_igamma",
        TorchParityBinaryOp.Igammac => "tp_igammac",
        TorchParityBinaryOp.FloorDivide => "tp_floor_divide",
        _ => throw new System.ArgumentOutOfRangeException(nameof(op)),
    };

    private static void RequireCapacity(IGpuBuffer buffer, long required, string parameter, string formula)
    {
        if (buffer is null) throw new System.ArgumentNullException(parameter);
        if (buffer.Size < required)
            throw new System.ArgumentException(
                $"{parameter} buffer too small: capacity={buffer.Size} elements, required={required} ({formula}).", parameter);
    }

    public static void Unary(PortableGenericFft.Launcher launch, TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output,
        int n, float p0, float p1)
    {
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch(UnaryKernel(op), n, input, output, n, p0, p1);
    }

    public static void Binary(PortableGenericFft.Launcher launch, TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(a, n, nameof(a), "n");
        RequireCapacity(b, n, nameof(b), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch(BinaryKernel(op), n, a, b, output, n);
    }

    public static void Polynomial(PortableGenericFft.Launcher launch, int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree,
        IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(x, n, nameof(x), "n");
        RequireCapacity(degree, n, nameof(degree), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch("tp_polynomial", n, x, degree, output, n, kind, shifted ? 1 : 0);
    }

    /// <summary>A double as the float pair hi + lo the kernels take for values they must keep to double precision.</summary>
    private static (float Hi, float Lo) Split(double value)
    {
        float hi = (float)value;
        if (float.IsInfinity(hi) || float.IsNaN(hi)) return (hi, 0f);
        return (hi, (float)(value - hi));
    }

    public static void Range(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, double start, double span, double denominator)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (s0, s1) = Split(start); var (p0, p1) = Split(span); var (d0, d1) = Split(denominator);
        launch("tp_range", n, output, n, s0, s1, p0, p1, d0, d1);
    }

    public static void Logspace(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, double start, double span,
        double denominator, double logBase, bool single)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (s0, s1) = Split(start); var (p0, p1) = Split(span); var (d0, d1) = Split(denominator);
        var (l0, l1) = Split(Math.Log(logBase));
        launch("tp_logspace", n, output, n, s0, s1, p0, p1, d0, d1, l0, l1, single ? 1 : 0);
    }

    public static void Window(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        launch("tp_window", n, output, n, kind, m, (float)alpha, (float)beta);
    }

    public static void TriIndices(PortableGenericFft.Launcher launch, IGpuBuffer output, int row, int col, int offset, bool lower, int total)
    {
        if (row <= 0 || total <= 0) return;
        RequireCapacity(output, 2L * total, nameof(output), "2*total");
        launch("tp_tri_indices", row, output, row, col, offset, lower ? 1 : 0, total);
    }

    public static void Diagflat(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
    {
        if (size <= 0) return;
        int total = checked(size * size);
        RequireCapacity(output, total, nameof(output), "size*size");
        RequireCapacity(input, n, nameof(input), "n");
        launch("tp_diagflat", total, input, output, total, size, offset, n);
    }

    public static void DiagonalScatter(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer src, IGpuBuffer output,
        int count, int[] otherExtents, int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
    {
        if (otherExtents is null) throw new ArgumentNullException(nameof(otherExtents));
        if (otherStrides is null) throw new ArgumentNullException(nameof(otherStrides));
        if (otherStrides.Length != otherExtents.Length)
            throw new ArgumentException("otherStrides must hold one stride per other axis.", nameof(otherStrides));
        if (count <= 0) return;
        if (otherExtents.Length > 8) throw new ArgumentOutOfRangeException(nameof(otherExtents), "at most 8 other axes.");
        if (diagonal <= 0) throw new ArgumentOutOfRangeException(nameof(diagonal));
        RequireCapacity(src, count, nameof(src), "count");
        long furthest = (long)(diagonal - 1 + start1) * stride1 + (long)(diagonal - 1 + start2) * stride2;
        for (int k = 0; k < otherExtents.Length; k++) furthest += (long)(otherExtents[k] - 1) * otherStrides[k];
        RequireCapacity(output, furthest + 1, nameof(output), "furthest diagonal offset + 1");
        var geometry = new int[22];
        geometry[0] = otherExtents.Length;
        for (int k = 0; k < otherExtents.Length; k++) { geometry[1 + k] = otherExtents[k]; geometry[9 + k] = otherStrides[k]; }
        geometry[17] = stride1; geometry[18] = stride2; geometry[19] = start1; geometry[20] = start2; geometry[21] = diagonal;
        using var g = backend.AllocateIntBuffer(geometry);
        launch("tp_diagonal_scatter", count, src, output, g, count);
    }

    public static void QuantileRows(PortableGenericFft.Launcher launch, IGpuBuffer sorted, IGpuBuffer output, int rows, int length,
        double q, int mode, bool ignoreNan)
    {
        if (rows <= 0) return;
        RequireCapacity(sorted, (long)rows * length, nameof(sorted), "rows*length");
        RequireCapacity(output, rows, nameof(output), "rows");
        var (q0, q1) = Split(q);
        launch("tp_quantile_rows", rows, sorted, output, rows, length, q0, q1, mode, ignoreNan ? 1 : 0);
    }

    private static (int Lo, int Hi) SeedWords(ulong seed) => (unchecked((int)(uint)seed), unchecked((int)(uint)(seed >> 32)));

    /// <summary>
    /// Runs a kernel that reports invalid parameters: it lowers a fresh 32-bit rank (element &lt;&lt; 3 | code) with
    /// atomicMin, and tp_flag_merge folds that into the engine's 64-bit flag, since WGSL and MSL have no 64-bit atomics.
    /// </summary>
    private static void Flagged(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer flag, long elements,
        Action<IGpuBuffer> run)
    {
        RequireCapacity(flag, 2, nameof(flag), "2 (64-bit flag)");
        if (elements >= 1L << 28)
            throw new ArgumentOutOfRangeException(nameof(elements), "FP32 backends flag at most 2^28 elements per launch.");
        using var rank = backend.AllocateIntBuffer(new[] { -1 });
        run(rank);
        launch("tp_flag_merge", 1, rank, flag, 1);
    }

    public static void FlagReset(PortableGenericFft.Launcher launch, IGpuBuffer flag)
    {
        RequireCapacity(flag, 2, nameof(flag), "2 (64-bit flag)");
        launch("tp_flag_reset", 2, flag, 2);
    }

    public static void Sample(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        launch("tp_sample", n, output, n, kind, (float)a, (float)b, lo, hi);
    }

    public static void Bernoulli(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer probabilities, IGpuBuffer output,
        int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(probabilities, n, nameof(probabilities), "n");
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        Flagged(backend, launch, flag, n, rank => launch("tp_bernoulli", n, probabilities, output, rank, n, lo, hi));
    }

    public static void Poisson(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer rates, IGpuBuffer output, int n,
        ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(rates, n, nameof(rates), "n");
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        Flagged(backend, launch, flag, n, rank => launch("tp_poisson", n, rates, output, rank, n, lo, hi));
    }

    public static void Binomial(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer counts, IGpuBuffer probabilities,
        IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(counts, n, nameof(counts), "n");
        RequireCapacity(probabilities, n, nameof(probabilities), "n");
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        Flagged(backend, launch, flag, n, rank => launch("tp_binomial", n, counts, probabilities, output, rank, n, lo, hi));
    }

    public static void Multinomial(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer weights, IGpuBuffer scratch,
        IGpuBuffer output, int rows, int categories, int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
    {
        if (rows <= 0) return;
        RequireCapacity(weights, (long)rows * categories, nameof(weights), "rows*categories");
        RequireCapacity(scratch, 2L * rows * categories, nameof(scratch), "2*rows*categories (double prefix sums)");
        RequireCapacity(output, (long)rows * numSamples, nameof(output), "rows*numSamples");
        var (lo, hi) = SeedWords(seed);
        Flagged(backend, launch, flag, (long)rows * categories, rank =>
            launch("tp_multinomial", rows, weights, scratch, output, rank, rows, categories, numSamples, replacement ? 1 : 0, lo, hi));
    }

    public static void DropoutAffine(PortableGenericFft.Launcher launch, IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p,
        double a, double sat, double b, bool alpha, ulong seed)
    {
        if (n <= 0) return;
        if (block <= 0) throw new ArgumentOutOfRangeException(nameof(block));
        RequireCapacity(scale, n, nameof(scale), "n");
        var (lo, hi) = SeedWords(seed);
        if (!alpha)
        {
            launch("tp_dropout_scale", n, scale, n, block, (float)p, (float)a, lo, hi);
            return;
        }
        RequireCapacity(offset, n, nameof(offset), "n");
        launch("tp_dropout_affine", n, scale, offset, n, block, (float)p, (float)a, (float)(a * sat + b), (float)b, lo, hi);
    }

    public static void Rrelu(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper,
        bool training, ulong seed)
    {
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        launch("tp_rrelu", n, input, output, n, (float)lower, (float)upper, (float)((lower + upper) / 2.0), training ? 1 : 0, lo, hi);
    }

    public static void RandomKeys(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, ulong seed)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (lo, hi) = SeedWords(seed);
        launch("tp_random_keys", n, output, n, lo, hi);
    }

    public static void IntBinary(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, int op, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int n, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(a, n, nameof(a), "n");
        RequireCapacity(b, n, nameof(b), "n");
        RequireCapacity(output, n, nameof(output), "n");
        Flagged(backend, launch, flag, n, rank => launch("tp_int_binary", n, a, b, output, rank, n, op));
    }

    public static void IntNot(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch("tp_int_not", n, input, output, n);
    }

    public static void Int64Binary(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, int op, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int n, int typeCode, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(a, 2L * n, nameof(a), "2n (int64 words)");
        RequireCapacity(b, 2L * n, nameof(b), "2n (int64 words)");
        RequireCapacity(output, 2L * n, nameof(output), "2n (int64 words)");
        Flagged(backend, launch, flag, n, rank => launch("tp_int64_binary", n, a, b, output, rank, n, op, typeCode));
    }

    public static void Int64Not(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n, int typeCode)
    {
        if (n <= 0) return;
        RequireCapacity(input, 2L * n, nameof(input), "2n (int64 words)");
        RequireCapacity(output, 2L * n, nameof(output), "2n (int64 words)");
        launch("tp_int64_not", n, input, output, n, typeCode);
    }

    /// <summary>
    /// Inclusive prefix sum as ceil(log2 n) Hillis-Steele passes ping-ponged between the output and a scratch buffer,
    /// ordered so the last pass writes the output (and the first never writes the input it reads).
    /// </summary>
    public static void InclusiveScan(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(output, n, nameof(output), "n");
        int passes = 1;
        while ((1L << passes) < n) passes++;
        using var scratch = backend.AllocateBuffer(n);
        IGpuBuffer? copy = null;
        try
        {
            IGpuBuffer source = input;
            if (ReferenceEquals(input, output) && passes % 2 == 1)
            {
                copy = backend.AllocateBuffer(n);
                launch("tp_scan_step", n, input, copy, n, n);
                source = copy;
            }
            for (int pass = 0; pass < passes; pass++)
            {
                IGpuBuffer target = (passes - 1 - pass) % 2 == 0 ? output : scratch;
                launch("tp_scan_step", n, source, target, n, 1 << pass);
                source = target;
            }
        }
        finally
        {
            copy?.Dispose();
        }
    }

    public static void NonzeroStaticScatter(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer positions,
        IGpuBuffer output, int n, int size, int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(positions, n, nameof(positions), "n");
        RequireCapacity(output, (long)size * shape.Length, nameof(output), "size*rank");
        if (shape.Length > 8) throw new ArgumentOutOfRangeException(nameof(shape), "at most 8 axes.");
        var packed = new int[9];
        packed[0] = shape.Length;
        for (int i = 0; i < shape.Length; i++) packed[1 + i] = shape[i];
        using var s = backend.AllocateIntBuffer(packed);
        launch("tp_nonzero_static_scatter", n, input, positions, output, s, n, size);
    }

    public static void UnravelAxis(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer indices, IGpuBuffer output, int n,
        long total, long stride, long extent, bool validate, IGpuBuffer flag)
    {
        if (n <= 0) return;
        RequireCapacity(indices, n, nameof(indices), "n");
        RequireCapacity(output, n, nameof(output), "n");
        if (total <= 0 || total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(total), "FP32 backends unravel at most int.MaxValue elements.");
        if (stride <= 0 || stride > total) throw new ArgumentOutOfRangeException(nameof(stride));
        if (extent <= 0 || extent > total) throw new ArgumentOutOfRangeException(nameof(extent));
        Flagged(backend, launch, flag, n, rank =>
            launch("tp_unravel_axis", n, indices, output, rank, n, (int)total, (int)stride, (int)extent, validate ? 1 : 0));
    }

    public static void SliceBitonicStep(PortableGenericFft.Launcher launch, IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j,
        int length, int sliceCount)
    {
        if (rowLen <= 0) return;
        if (sliceCount > rowLen) throw new ArgumentOutOfRangeException(nameof(sliceCount), "more slices than order slots.");
        RequireCapacity(order, rowLen, nameof(order), "rowLen");
        RequireCapacity(slices, (long)sliceCount * length, nameof(slices), "sliceCount*length");
        launch("tp_slice_bitonic_step", rowLen, slices, order, rowLen, k, j, length);
    }

    public static void SliceFirstFlags(PortableGenericFft.Launcher launch, IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
    {
        if (n <= 0) return;
        RequireCapacity(order, n, nameof(order), "n");
        RequireCapacity(flags, n, nameof(flags), "n");
        RequireCapacity(slices, (long)n * length, nameof(slices), "n*length");
        launch("tp_slice_first_flags", n, slices, order, flags, n, length);
    }

    public static void SliceGatherUnique(PortableGenericFft.Launcher launch, IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags,
        IGpuBuffer positions, IGpuBuffer output, int n, int length, int outputRows)
    {
        int total = checked(n * length);
        RequireCapacity(output, (long)outputRows * length, nameof(output), "outputRows*length");
        if (total <= 0) return;
        RequireCapacity(slices, (long)n * length, nameof(slices), "n*length");
        RequireCapacity(order, n, nameof(order), "n");
        RequireCapacity(flags, n, nameof(flags), "n");
        RequireCapacity(positions, n, nameof(positions), "n");
        launch("tp_slice_gather_unique", total, slices, order, flags, positions, output, total, length);
    }

    public static void GridSample3D(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches,
        int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        RequireCapacity(input, (long)batches * channels * depth * height * width, nameof(input), "N*C*D*H*W");
        RequireCapacity(grid, 3L * batches * points, nameof(grid), "N*points*3");
        RequireCapacity(output, (long)batches * channels * points, nameof(output), "N*C*points");
        launch("tp_grid_sample_3d", threads, input, grid, output, batches, channels, depth, height, width, points, mode, padding,
            alignCorners ? 1 : 0);
    }

    public static void GridSample3DBackward(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad,
        IGpuBuffer inputGrad, IGpuBuffer gridGrad, int batches, int channels, int depth, int height, int width, int points, int mode,
        int padding, bool alignCorners)
    {
        int threads = checked(batches * points);
        if (threads <= 0) return;
        long volume = (long)batches * channels * depth * height * width;
        RequireCapacity(input, volume, nameof(input), "N*C*D*H*W");
        RequireCapacity(inputGrad, volume, nameof(inputGrad), "N*C*D*H*W");
        RequireCapacity(grid, 3L * batches * points, nameof(grid), "N*points*3");
        RequireCapacity(gridGrad, 3L * batches * points, nameof(gridGrad), "N*points*3");
        RequireCapacity(outputGrad, (long)batches * channels * points, nameof(outputGrad), "N*C*points");
        launch("tp_grid_sample_3d_backward", threads, input, grid, outputGrad, inputGrad, gridGrad, batches, channels, depth, height, width,
            points, mode, padding, alignCorners ? 1 : 0);
    }
}