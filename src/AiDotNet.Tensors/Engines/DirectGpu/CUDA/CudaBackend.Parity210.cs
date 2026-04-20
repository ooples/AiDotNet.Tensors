// Copyright (c) AiDotNet. All rights reserved.
// CUDA launcher shims for the parity-210 kernels. Each method resolves the
// NVRTC-compiled kernel from _kernelCache (populated by CompileAllKernels)
// and dispatches it with the standard 256-thread block / grid-ceil launch
// geometry used elsewhere in this backend.
//
// These methods are called from DirectGpuTensorEngine's parity-210 overrides.
// Types are deliberately mapped to fp32 buffers — the tensor engine lifts
// generic T to float on entry and lowers on exit, matching the activation
// and elementwise paths.

using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IParity210Backend
{
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private IntPtr ResolveParity210Kernel(string name)
    {
        if (_parity210Module == IntPtr.Zero)
            throw new InvalidOperationException(
                "Parity-210 CUDA module was not compiled (older CUDA toolkit?). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    private unsafe void LaunchParity210Ternary(
        string kernelName,
        IntPtr arg0, IntPtr arg1, IntPtr arg2, IntPtr arg3,
        int p0, int p1, int p2, int p3, int total)
    {
        var kernel = ResolveParity210Kernel(kernelName);
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        var a0 = arg0; var a1 = arg1; var a2 = arg2; var a3 = arg3;
        int q0 = p0, q1 = p1, q2 = p2, q3 = p3;
        void** args = stackalloc void*[8];
        args[0] = &a0; args[1] = &a1; args[2] = &a2; args[3] = &a3;
        args[4] = &q0; args[5] = &q1; args[6] = &q2; args[7] = &q3;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------

    public unsafe void Parity210Roll1D(
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize, int shift)
    {
        // Empty-shape guard — avoids cuLaunchKernel with grid=0 which some
        // driver versions reject with CUDA_ERROR_INVALID_VALUE.
        if (outerSize <= 0 || axisSize <= 0 || innerSize <= 0) return;
        var kernel = ResolveParity210Kernel("parity210_roll_1d");
        using var _ = PushContext();
        int total = outerSize * axisSize * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize, s = shift;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i; args[5] = &s;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210FlipAxis(
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        if (outerSize <= 0 || axisSize <= 0 || innerSize <= 0) return;
        var kernel = ResolveParity210Kernel("parity210_flip_axis");
        using var _ = PushContext();
        int total = outerSize * axisSize * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210Triu(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        LaunchTriangularMask("parity210_triu", input, output, batchSize, rows, cols, diagonal);
    }

    public unsafe void Parity210Tril(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        LaunchTriangularMask("parity210_tril", input, output, batchSize, rows, cols, diagonal);
    }

    private unsafe void LaunchTriangularMask(
        string name, IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        if (batchSize <= 0 || rows <= 0 || cols <= 0) return;
        var kernel = ResolveParity210Kernel(name);
        using var _ = PushContext();
        int total = batchSize * rows * cols;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int b = batchSize, r = rows, c = cols, d = diagonal;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &b; args[3] = &r; args[4] = &c; args[5] = &d;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210DiagEmbed(
        IGpuBuffer input, IGpuBuffer output,
        int batchSize, int diagLen, int matSize, int offset)
    {
        if (batchSize <= 0 || matSize <= 0) return;
        var kernel = ResolveParity210Kernel("parity210_diag_embed");
        using var _ = PushContext();
        int total = batchSize * matSize * matSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int b = batchSize, d = diagLen, m = matSize, off = offset;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &b; args[3] = &d; args[4] = &m; args[5] = &off;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // CUMULATIVE
    // -----------------------------------------------------------------------

    public unsafe void Parity210CumSum(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        // Empty-axis shape (e.g. [outer, 0, inner]) has zero-length output;
        // skip launching before probing axisSize against the 1024 threshold.
        if (axisSize <= 0 || outerSize <= 0 || innerSize <= 0) return;
        // Block-level Hillis-Steele scan for axes up to 1024 — each line gets
        // one thread block with 2*blockDim.x floats of shared memory for
        // ping-pong buffers. Longer axes fall through to the per-line serial
        // kernel (multi-block carry scheme is a follow-up).
        if (axisSize <= 1024)
        {
            LaunchBlockScanCumSum(input, output, outerSize, axisSize, innerSize);
            return;
        }
        LaunchCumulativeLine("parity210_cumsum_axis", input, output, outerSize, axisSize, innerSize);
    }

    private unsafe void LaunchBlockScanCumSum(
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        var kernel = ResolveParity210Kernel("parity210_cumsum_block_hillis_steele");
        using var _ = PushContext();
        // Pick next power of two ≥ axisSize, capped at 1024 per block.
        uint block = 1;
        while (block < (uint)axisSize && block < 1024) block <<= 1;
        uint grid = (uint)(outerSize * innerSize);
        uint sharedBytes = block * 2 * sizeof(float);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i;
        LaunchKernelWithSharedMem(kernel, grid, block, sharedBytes, args);
    }

    public unsafe void Parity210CumProd(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLine("parity210_cumprod_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210CumMax(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLine("parity210_cummax_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210CumMin(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLine("parity210_cummin_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLine("parity210_logcumsumexp_axis", input, output, outerSize, axisSize, innerSize);

    private unsafe void LaunchCumulativeLine(
        string name, IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        // Empty-axis guard: for shapes like [outer, 0, inner] the output is
        // also empty — launching a kernel would read from zero-length buffers.
        if (axisSize <= 0 || outerSize <= 0 || innerSize <= 0) return;
        var kernel = ResolveParity210Kernel(name);
        using var _ = PushContext();
        int total = outerSize * innerSize;       // one thread per line
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // INDEXING
    // -----------------------------------------------------------------------

    public unsafe void Parity210TakeLinear(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outSize, int inputLinearLen)
    {
        var kernel = ResolveParity210Kernel("parity210_take_linear");
        using var _ = PushContext();
        uint grid = (uint)((outSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr outPtr = output.Handle;
        int o = outSize, l = inputLinearLen;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &idxPtr; args[2] = &outPtr;
        args[3] = &o; args[4] = &l;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210TakeAlongDim(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outerSize, int idxAxis, int innerSize, int srcAxis)
    {
        var kernel = ResolveParity210Kernel("parity210_take_along_dim");
        using var _ = PushContext();
        int total = outerSize * idxAxis * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, ia = idxAxis, i = innerSize, sa = srcAxis;
        void** args = stackalloc void*[7];
        args[0] = &inPtr; args[1] = &idxPtr; args[2] = &outPtr;
        args[3] = &o; args[4] = &ia; args[5] = &i; args[6] = &sa;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210IndexAdd(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_add");
        using var _ = PushContext();
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr srcPtr = source.Handle;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &srcPtr;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210IndexCopy(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_copy");
        using var _ = PushContext();
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr srcPtr = source.Handle;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &srcPtr;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210IndexFill(
        IGpuBuffer output, IGpuBuffer indicesInt32, float fillValue,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_fill");
        using var _ = PushContext();
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle;
        float f = fillValue;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &f;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210MaskedScatter(
        IGpuBuffer output, IGpuBuffer maskInt8, IGpuBuffer prefixSumInt32,
        IGpuBuffer source, int total)
    {
        var kernel = ResolveParity210Kernel("parity210_masked_scatter");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr maskPtr = maskInt8.Handle;
        IntPtr prefPtr = prefixSumInt32.Handle; IntPtr srcPtr = source.Handle;
        int n = total;
        int sourceLen = source.Size;
        void** args = stackalloc void*[6];
        args[0] = &outPtr; args[1] = &maskPtr; args[2] = &prefPtr; args[3] = &srcPtr;
        args[4] = &n; args[5] = &sourceLen;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE BINARY  (reuses LaunchElementwiseKernel for (a, b, out, size))
    // -----------------------------------------------------------------------

    public void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_hypot", a, b, output, size);

    public void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_copysign", a, b, output, size);

    public void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_fmod", a, b, output, size);

    public void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_remainder", a, b, output, size);

    public void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_float_power", a, b, output, size);

    public void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_log_add_exp", a, b, output, size);

    public void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_log_add_exp2", a, b, output, size);

    public void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_xlogy", x, y, output, size);

    public void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => LaunchElementwiseKernel("parity210_xlog1py", x, y, output, size);

    // -----------------------------------------------------------------------
    // ELEMENT-WISE UNARY SPECIAL  (reuses LaunchUnaryKernel)
    // -----------------------------------------------------------------------

    public void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_erfc", input, output, size);

    public void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_erfinv", input, output, size);

    public void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_lgamma_approx", input, output, size);

    public void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_digamma", input, output, size);

    public void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_i0", input, output, size);

    public void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_i1", input, output, size);

    public void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_i0e", input, output, size);

    public void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_i1e", input, output, size);

    public void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_is_finite", input, output, size);

    public void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_is_nan", input, output, size);

    public void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryKernel("parity210_is_inf", input, output, size);

    public unsafe void Parity210NanToNum(
        IGpuBuffer input, IGpuBuffer output, int size,
        float nanVal, float posInfVal, float negInfVal)
    {
        var kernel = ResolveParity210Kernel("parity210_nan_to_num");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int n = size;
        float nv = nanVal, pv = posInfVal, negV = negInfVal;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &n;
        args[3] = &nv; args[4] = &pv; args[5] = &negV;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // PAIRWISE
    // -----------------------------------------------------------------------

    public unsafe void Parity210CosineSimilarityLast(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
    {
        var kernel = ResolveParity210Kernel("parity210_cosine_similarity_last");
        using var _ = PushContext();
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle; IntPtr bPtr = b.Handle; IntPtr outPtr = output.Handle;
        int nn = n, dd = d;
        float ep = eps;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr;
        args[3] = &nn; args[4] = &dd; args[5] = &ep;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Parity210CdistL2(
        IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
    {
        var kernel = ResolveParity210Kernel("parity210_cdist_l2");
        using var _ = PushContext();
        int total = n * m;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr x1Ptr = x1.Handle; IntPtr x2Ptr = x2.Handle; IntPtr outPtr = output.Handle;
        int nn = n, mm = m, dd = d;
        void** args = stackalloc void*[6];
        args[0] = &x1Ptr; args[1] = &x2Ptr; args[2] = &outPtr;
        args[3] = &nn; args[4] = &mm; args[5] = &dd;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    // -----------------------------------------------------------------------
    // CLAMP (tensor-bounds)
    // -----------------------------------------------------------------------

    public unsafe void Parity210ClampMinMax(
        IGpuBuffer input, IGpuBuffer lo, IGpuBuffer hi, IGpuBuffer output,
        int size, bool hasLo, bool hasHi)
    {
        var kernel = ResolveParity210Kernel("parity210_clamp_min_max");
        using var _ = PushContext();
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr loPtr = lo?.Handle ?? input.Handle;   // unused branch handled in kernel via hasLo flag
        IntPtr hiPtr = hi?.Handle ?? input.Handle;
        IntPtr outPtr = output.Handle;
        int n = size, hl = hasLo ? 1 : 0, hh = hasHi ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &inPtr; args[1] = &loPtr; args[2] = &hiPtr; args[3] = &outPtr;
        args[4] = &n; args[5] = &hl; args[6] = &hh;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
}
