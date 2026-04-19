// Copyright (c) AiDotNet. All rights reserved.
// OpenCL launcher shims for the parity-210 kernels. Each method pulls the
// compiled DirectOpenClKernel from _kernelCache (populated at backend init)
// and dispatches via kernel.Execute1D with 256-thread workgroups.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        private const int Parity210LocalSize = 256;

        private DirectOpenClKernel GetParity210Kernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException(
                    $"OpenCL Parity-210 kernel not found: {name}. Module may have failed to compile.");
            return kernel;
        }

        private static int RoundUpToMultiple(int v, int m) => ((v + m - 1) / m) * m;

        private static IntPtr BufHandle(IGpuBuffer b) => ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        // -------------------------------------------------------------------
        // MOVEMENT
        // -------------------------------------------------------------------

        public void Parity210Roll1D(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize, int shift)
        {
            int total = outerSize * axisSize * innerSize;
            var k = GetParity210Kernel("parity210_roll_1d");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, outerSize);
            k.SetArg(3, axisSize);
            k.SetArg(4, innerSize);
            k.SetArg(5, shift);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }

        public void Parity210FlipAxis(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
        {
            int total = outerSize * axisSize * innerSize;
            var k = GetParity210Kernel("parity210_flip_axis");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, outerSize);
            k.SetArg(3, axisSize);
            k.SetArg(4, innerSize);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }

        public void Parity210Triu(IGpuBuffer input, IGpuBuffer output,
            int batchSize, int rows, int cols, int diagonal)
            => DispatchTriuTrilCl("parity210_triu", input, output, batchSize, rows, cols, diagonal);

        public void Parity210Tril(IGpuBuffer input, IGpuBuffer output,
            int batchSize, int rows, int cols, int diagonal)
            => DispatchTriuTrilCl("parity210_tril", input, output, batchSize, rows, cols, diagonal);

        private void DispatchTriuTrilCl(string name,
            IGpuBuffer input, IGpuBuffer output,
            int batchSize, int rows, int cols, int diagonal)
        {
            int total = batchSize * rows * cols;
            var k = GetParity210Kernel(name);
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, batchSize);
            k.SetArg(3, rows);
            k.SetArg(4, cols);
            k.SetArg(5, diagonal);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }

        public void Parity210DiagEmbed(IGpuBuffer input, IGpuBuffer output,
            int batchSize, int diagLen, int matSize, int offset)
        {
            int total = batchSize * matSize * matSize;
            var k = GetParity210Kernel("parity210_diag_embed");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, batchSize);
            k.SetArg(3, diagLen);
            k.SetArg(4, matSize);
            k.SetArg(5, offset);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }

        // -------------------------------------------------------------------
        // CUMULATIVE
        // -------------------------------------------------------------------

        public void Parity210CumSum(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
            => DispatchCumulativeCl("parity210_cumsum_axis", input, output, outerSize, axisSize, innerSize);

        public void Parity210CumProd(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
            => DispatchCumulativeCl("parity210_cumprod_axis", input, output, outerSize, axisSize, innerSize);

        public void Parity210CumMax(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
            => DispatchCumulativeCl("parity210_cummax_axis", input, output, outerSize, axisSize, innerSize);

        public void Parity210CumMin(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
            => DispatchCumulativeCl("parity210_cummin_axis", input, output, outerSize, axisSize, innerSize);

        public void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
            => DispatchCumulativeCl("parity210_logcumsumexp_axis", input, output, outerSize, axisSize, innerSize);

        private void DispatchCumulativeCl(string name,
            IGpuBuffer input, IGpuBuffer output,
            int outerSize, int axisSize, int innerSize)
        {
            int total = outerSize * innerSize;
            var k = GetParity210Kernel(name);
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, outerSize);
            k.SetArg(3, axisSize);
            k.SetArg(4, innerSize);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }

        // -------------------------------------------------------------------
        // ELEMENT-WISE BINARY
        // -------------------------------------------------------------------

        public void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_hypot", a, b, o, size);

        public void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_copysign", a, b, o, size);

        public void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_fmod", a, b, o, size);

        public void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_remainder", a, b, o, size);

        public void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_float_power", a, b, o, size);

        public void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_log_add_exp", a, b, o, size);

        public void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_log_add_exp2", a, b, o, size);

        public void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_xlogy", x, y, o, size);

        public void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
            => DispatchBinaryCl("parity210_xlog1py", x, y, o, size);

        private void DispatchBinaryCl(string name, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        {
            var k = GetParity210Kernel(name);
            k.SetArg(0, BufHandle(a));
            k.SetArg(1, BufHandle(b));
            k.SetArg(2, BufHandle(o));
            k.SetArg(3, size);
            k.Execute1D(RoundUpToMultiple(size, Parity210LocalSize), Parity210LocalSize);
        }

        // -------------------------------------------------------------------
        // ELEMENT-WISE UNARY SPECIAL
        // -------------------------------------------------------------------

        public void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_erfc", input, output, size);

        public void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_erfinv", input, output, size);

        public void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_lgamma_approx", input, output, size);

        public void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_digamma", input, output, size);

        public void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_i0", input, output, size);

        public void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_i1", input, output, size);

        public void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_i0e", input, output, size);

        public void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_i1e", input, output, size);

        public void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_is_finite", input, output, size);

        public void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_is_nan", input, output, size);

        public void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
            => DispatchUnaryCl("parity210_is_inf", input, output, size);

        private void DispatchUnaryCl(string name, IGpuBuffer input, IGpuBuffer output, int size)
        {
            var k = GetParity210Kernel(name);
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, size);
            k.Execute1D(RoundUpToMultiple(size, Parity210LocalSize), Parity210LocalSize);
        }

        public void Parity210NanToNum(IGpuBuffer input, IGpuBuffer output, int size,
            float nanVal, float posInfVal, float negInfVal)
        {
            var k = GetParity210Kernel("parity210_nan_to_num");
            k.SetArg(0, BufHandle(input));
            k.SetArg(1, BufHandle(output));
            k.SetArg(2, size);
            k.SetArg(3, nanVal);
            k.SetArg(4, posInfVal);
            k.SetArg(5, negInfVal);
            k.Execute1D(RoundUpToMultiple(size, Parity210LocalSize), Parity210LocalSize);
        }

        // -------------------------------------------------------------------
        // PAIRWISE
        // -------------------------------------------------------------------

        public void Parity210CosineSimilarityLast(
            IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
        {
            var k = GetParity210Kernel("parity210_cosine_similarity_last");
            k.SetArg(0, BufHandle(a));
            k.SetArg(1, BufHandle(b));
            k.SetArg(2, BufHandle(output));
            k.SetArg(3, n);
            k.SetArg(4, d);
            k.SetArg(5, eps);
            k.Execute1D(RoundUpToMultiple(n, Parity210LocalSize), Parity210LocalSize);
        }

        public void Parity210CdistL2(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
        {
            int total = n * m;
            var k = GetParity210Kernel("parity210_cdist_l2");
            k.SetArg(0, BufHandle(x1));
            k.SetArg(1, BufHandle(x2));
            k.SetArg(2, BufHandle(output));
            k.SetArg(3, n);
            k.SetArg(4, m);
            k.SetArg(5, d);
            k.Execute1D(RoundUpToMultiple(total, Parity210LocalSize), Parity210LocalSize);
        }
    }
}
#endif
