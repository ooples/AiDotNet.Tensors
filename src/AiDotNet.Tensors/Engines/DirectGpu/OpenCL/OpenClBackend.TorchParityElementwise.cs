#if !NET462
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Launches for the PyTorch-parity elementwise kernels (OpenClTorchParityElementwiseKernels), argument for argument
    // with CudaBackend.TorchParityElementwise (#1126).
    public sealed partial class OpenClBackend : ITorchParityElementwiseBackend
    {
        private const int ParityLocalSize = 256;

        private DirectOpenClKernel ParityKernel(string name)
            => _kernelCache.TryGetValue(name, out var kernel)
                ? kernel
                : throw new InvalidOperationException($"OpenCL kernel not found: {name}. The parity module may have failed to compile.");

        private static void ParityRequire(IGpuBuffer buffer, long requiredElements, string parameter, string formula)
        {
            if (buffer is null) throw new ArgumentNullException(parameter);
            if (buffer.Size < requiredElements)
                throw new ArgumentException(
                    $"{parameter} buffer too small: capacity={buffer.Size} elements, required={requiredElements} ({formula}).", parameter);
        }

        private static void ParityLaunch(DirectOpenClKernel kernel, int threads)
            => kernel.Execute1D(RoundUpToMultiple(threads, ParityLocalSize), ParityLocalSize);

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
            var k = ParityKernel(UnaryKernelName(op));
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n); k.SetArg(3, p0); k.SetArg(4, p1);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
        {
            if (n <= 0) return;
            ParityRequire(a, n, nameof(a), "n");
            ParityRequire(b, n, nameof(b), "n");
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel(BinaryKernelName(op));
            k.SetArg(0, BufHandle(a)); k.SetArg(1, BufHandle(b)); k.SetArg(2, BufHandle(output)); k.SetArg(3, n);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
        {
            if (n <= 0) return;
            ParityRequire(x, n, nameof(x), "n");
            ParityRequire(degree, n, nameof(degree), "n");
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_polynomial");
            k.SetArg(0, BufHandle(x)); k.SetArg(1, BufHandle(degree)); k.SetArg(2, BufHandle(output)); k.SetArg(3, n);
            k.SetArg(4, kind); k.SetArg(5, shifted ? 1 : 0);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
        {
            if (n <= 0) return;
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_range");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, n); k.SetArg(2, start); k.SetArg(3, span); k.SetArg(4, denominator);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator,
            double logBase, bool single)
        {
            if (n <= 0) return;
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_logspace");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, n); k.SetArg(2, start); k.SetArg(3, span); k.SetArg(4, denominator);
            k.SetArg(5, logBase); k.SetArg(6, single ? 1 : 0);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
        {
            if (n <= 0) return;
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_window");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, n); k.SetArg(2, kind); k.SetArg(3, m); k.SetArg(4, alpha); k.SetArg(5, beta);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
        {
            if (row <= 0 || total <= 0) return;
            ParityRequire(output, 2L * total, nameof(output), "2*total");
            var k = ParityKernel("parity_tri_indices");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, row); k.SetArg(2, col); k.SetArg(3, offset); k.SetArg(4, lower ? 1 : 0); k.SetArg(5, total);
            ParityLaunch(k, row);
        }

        void ITorchParityElementwiseBackend.TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
        {
            if (size <= 0) return;
            int total = checked(size * size);
            ParityRequire(output, total, nameof(output), "size*size");
            ParityRequire(input, n, nameof(input), "n");
            var k = ParityKernel("parity_diagflat");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(output)); k.SetArg(2, total); k.SetArg(3, size); k.SetArg(4, offset); k.SetArg(5, n);
            ParityLaunch(k, total);
        }

        void ITorchParityElementwiseBackend.TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents,
            int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
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
            // The kernel reads the geometry CUDA passes by value: rank, 8 extents, 8 strides, the two strides and starts, diagonal.
            var geometry = new int[22];
            geometry[0] = otherExtents.Length;
            for (int i = 0; i < otherExtents.Length; i++) { geometry[1 + i] = otherExtents[i]; geometry[9 + i] = otherStrides[i]; }
            geometry[17] = stride1; geometry[18] = stride2; geometry[19] = start1; geometry[20] = start2; geometry[21] = diagonal;
            using var g = AllocateIntBuffer(geometry);
            var k = ParityKernel("parity_diagonal_scatter");
            k.SetArg(0, BufHandle(src)); k.SetArg(1, BufHandle(output)); k.SetArg(2, count); k.SetArg(3, BufHandle(g));
            ParityLaunch(k, count);
        }

        void ITorchParityElementwiseBackend.TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q, int mode,
            bool ignoreNan)
        {
            if (rows <= 0) return;
            ParityRequire(sorted, (long)rows * length, nameof(sorted), "rows*length");
            ParityRequire(output, rows, nameof(output), "rows");
            var k = ParityKernel("parity_quantile_rows");
            k.SetArg(0, BufHandle(sorted)); k.SetArg(1, BufHandle(output)); k.SetArg(2, rows); k.SetArg(3, length); k.SetArg(4, q);
            k.SetArg(5, mode); k.SetArg(6, ignoreNan ? 1 : 0);
            ParityLaunch(k, rows);
        }

        void ITorchParityElementwiseBackend.TorchParityFlagReset(IGpuBuffer flag)
        {
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_flag_reset");
            k.SetArg(0, BufHandle(flag));
            k.Execute1D(1, 1);
        }

        void ITorchParityElementwiseBackend.TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
        {
            if (n <= 0) return;
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_sample");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, n); k.SetArg(2, kind); k.SetArg(3, a); k.SetArg(4, b); k.SetArg(5, seed);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        {
            if (n <= 0) return;
            ParityRequire(probabilities, n, nameof(probabilities), "n");
            ParityRequire(output, n, nameof(output), "n");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_bernoulli");
            k.SetArg(0, BufHandle(probabilities)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n); k.SetArg(3, seed); k.SetArg(4, BufHandle(flag));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        {
            if (n <= 0) return;
            ParityRequire(rates, n, nameof(rates), "n");
            ParityRequire(output, n, nameof(output), "n");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_poisson");
            k.SetArg(0, BufHandle(rates)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n); k.SetArg(3, seed); k.SetArg(4, BufHandle(flag));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed,
            IGpuBuffer flag)
        {
            if (n <= 0) return;
            ParityRequire(counts, n, nameof(counts), "n");
            ParityRequire(probabilities, n, nameof(probabilities), "n");
            ParityRequire(output, n, nameof(output), "n");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_binomial");
            k.SetArg(0, BufHandle(counts)); k.SetArg(1, BufHandle(probabilities)); k.SetArg(2, BufHandle(output)); k.SetArg(3, n);
            k.SetArg(4, seed); k.SetArg(5, BufHandle(flag));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows, int categories,
            int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
        {
            if (rows <= 0) return;
            ParityRequire(weights, (long)rows * categories, nameof(weights), "rows*categories");
            ParityRequire(scratch, 2L * rows * categories, nameof(scratch), "2*rows*categories (double prefix sums)");
            ParityRequire(output, (long)rows * numSamples, nameof(output), "rows*numSamples");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_multinomial");
            k.SetArg(0, BufHandle(weights)); k.SetArg(1, BufHandle(scratch)); k.SetArg(2, BufHandle(output)); k.SetArg(3, rows);
            k.SetArg(4, categories); k.SetArg(5, numSamples); k.SetArg(6, replacement ? 1 : 0); k.SetArg(7, seed); k.SetArg(8, BufHandle(flag));
            ParityLaunch(k, rows);
        }

        void ITorchParityElementwiseBackend.TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a,
            double sat, double b, bool alpha, ulong seed)
        {
            if (n <= 0) return;
            ParityRequire(scale, n, nameof(scale), "n");
            if (alpha) ParityRequire(offset, n, nameof(offset), "n");
            var k = ParityKernel("parity_dropout_affine");
            k.SetArg(0, BufHandle(scale)); k.SetArg(1, BufHandle(offset)); k.SetArg(2, n); k.SetArg(3, block); k.SetArg(4, p); k.SetArg(5, a);
            k.SetArg(6, sat); k.SetArg(7, b); k.SetArg(8, alpha ? 1 : 0); k.SetArg(9, seed);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper, bool training,
            ulong seed)
        {
            if (n <= 0) return;
            ParityRequire(input, n, nameof(input), "n");
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_rrelu");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n); k.SetArg(3, lower); k.SetArg(4, upper);
            k.SetArg(5, training ? 1 : 0); k.SetArg(6, seed);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
        {
            if (n <= 0) return;
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_random_keys");
            k.SetArg(0, BufHandle(output)); k.SetArg(1, n); k.SetArg(2, seed);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag)
        {
            if (n <= 0) return;
            ParityRequire(a, n, nameof(a), "n");
            ParityRequire(b, n, nameof(b), "n");
            ParityRequire(output, n, nameof(output), "n");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_int_binary");
            k.SetArg(0, BufHandle(a)); k.SetArg(1, BufHandle(b)); k.SetArg(2, BufHandle(output)); k.SetArg(3, n); k.SetArg(4, op);
            k.SetArg(5, BufHandle(flag));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n)
        {
            if (n <= 0) return;
            ParityRequire(input, n, nameof(input), "n");
            ParityRequire(output, n, nameof(output), "n");
            var k = ParityKernel("parity_int_not");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n);
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n)
        {
            if (n <= 0) return;
            ParityRequire(input, n, nameof(input), "n");
            ParityRequire(output, n, nameof(output), "n");
            const int block = OpenClTorchParityElementwiseKernels.ScanBlock;
            int blocks = (n + block - 1) / block;
            IGpuBuffer? sums = blocks > 1 ? AllocateBuffer(blocks) : null;
            IGpuBuffer? carries = blocks > 1 ? AllocateBuffer(blocks) : null;
            try
            {
                var scan = ParityKernel("parity_scan_block");
                scan.SetArg(0, BufHandle(input)); scan.SetArg(1, BufHandle(output));
                // A kernel buffer argument cannot be null: without block sums the output doubles as the unused slot.
                scan.SetArg(2, BufHandle(sums ?? output)); scan.SetArg(3, n); scan.SetArg(4, sums is null ? 0 : 1);
                scan.Execute1D(blocks * block, block);
                if (sums is null || carries is null) return;
                ((ITorchParityElementwiseBackend)this).TorchParityInclusiveScan(sums, carries, blocks);
                var add = ParityKernel("parity_scan_add");
                add.SetArg(0, BufHandle(output)); add.SetArg(1, BufHandle(carries)); add.SetArg(2, n);
                ParityLaunch(add, n);
            }
            finally
            {
                sums?.Dispose();
                carries?.Dispose();
            }
        }

        void ITorchParityElementwiseBackend.TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n, int size,
            int[] shape)
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
            var k = ParityKernel("parity_nonzero_static_scatter");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(positions)); k.SetArg(2, BufHandle(output)); k.SetArg(3, n); k.SetArg(4, size);
            k.SetArg(5, BufHandle(s));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride, long extent,
            bool validate, IGpuBuffer flag)
        {
            if (n <= 0) return;
            ParityRequire(indices, n, nameof(indices), "n");
            ParityRequire(output, n, nameof(output), "n");
            ParityRequire(flag, 2, nameof(flag), "2 (64-bit flag)");
            var k = ParityKernel("parity_unravel_axis");
            k.SetArg(0, BufHandle(indices)); k.SetArg(1, BufHandle(output)); k.SetArg(2, n); k.SetArg(3, unchecked((ulong)total));
            k.SetArg(4, unchecked((ulong)stride)); k.SetArg(5, unchecked((ulong)extent)); k.SetArg(6, validate ? 1 : 0); k.SetArg(7, BufHandle(flag));
            ParityLaunch(k, n);
        }

        void ITorchParityElementwiseBackend.TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j, int length,
            int sliceCount)
        {
            if (rowLen <= 0) return;
            if (sliceCount > rowLen) throw new ArgumentOutOfRangeException(nameof(sliceCount), "more slices than order slots.");
            ParityRequire(order, rowLen, nameof(order), "rowLen");
            ParityRequire(slices, (long)sliceCount * length, nameof(slices), "sliceCount*length");
            var kernel = ParityKernel("parity_slice_bitonic_step");
            kernel.SetArg(0, BufHandle(slices)); kernel.SetArg(1, BufHandle(order)); kernel.SetArg(2, rowLen); kernel.SetArg(3, k);
            kernel.SetArg(4, j); kernel.SetArg(5, length);
            ParityLaunch(kernel, rowLen);
        }

        void ITorchParityElementwiseBackend.TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
        {
            if (n <= 0) return;
            ParityRequire(order, n, nameof(order), "n");
            ParityRequire(flags, n, nameof(flags), "n");
            ParityRequire(slices, (long)n * length, nameof(slices), "n*length");
            var k = ParityKernel("parity_slice_first_flags");
            k.SetArg(0, BufHandle(slices)); k.SetArg(1, BufHandle(order)); k.SetArg(2, BufHandle(flags)); k.SetArg(3, n); k.SetArg(4, length);
            ParityLaunch(k, n);
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
            var k = ParityKernel("parity_slice_gather_unique");
            k.SetArg(0, BufHandle(slices)); k.SetArg(1, BufHandle(order)); k.SetArg(2, BufHandle(flags)); k.SetArg(3, BufHandle(positions));
            k.SetArg(4, BufHandle(output)); k.SetArg(5, total); k.SetArg(6, length);
            ParityLaunch(k, total);
        }

        void ITorchParityElementwiseBackend.TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches, int channels,
            int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
        {
            int threads = checked(batches * points);
            if (threads <= 0) return;
            ParityRequire(input, (long)batches * channels * depth * height * width, nameof(input), "N*C*D*H*W");
            ParityRequire(grid, 3L * batches * points, nameof(grid), "N*points*3");
            ParityRequire(output, (long)batches * channels * points, nameof(output), "N*C*points");
            var k = ParityKernel("parity_grid_sample_3d");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(grid)); k.SetArg(2, BufHandle(output)); k.SetArg(3, batches);
            k.SetArg(4, channels); k.SetArg(5, depth); k.SetArg(6, height); k.SetArg(7, width); k.SetArg(8, points); k.SetArg(9, mode);
            k.SetArg(10, padding); k.SetArg(11, alignCorners ? 1 : 0);
            ParityLaunch(k, threads);
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
            var k = ParityKernel("parity_grid_sample_3d_backward");
            k.SetArg(0, BufHandle(input)); k.SetArg(1, BufHandle(grid)); k.SetArg(2, BufHandle(outputGrad)); k.SetArg(3, BufHandle(inputGrad));
            k.SetArg(4, BufHandle(gridGrad)); k.SetArg(5, batches); k.SetArg(6, channels); k.SetArg(7, depth); k.SetArg(8, height);
            k.SetArg(9, width); k.SetArg(10, points); k.SetArg(11, mode); k.SetArg(12, padding); k.SetArg(13, alignCorners ? 1 : 0);
            ParityLaunch(k, threads);
        }
    }
}
#endif