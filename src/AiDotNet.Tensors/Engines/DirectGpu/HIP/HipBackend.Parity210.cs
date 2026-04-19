// Copyright (c) AiDotNet. All rights reserved.
// HIP launcher shims for the parity-210 kernels. The HIP launch API takes
// the same void**-array as CUDA, so every method here is a direct port of
// its CudaBackend.Parity210.cs counterpart — only the runtime API name
// (hipModuleLaunchKernel vs cuLaunchKernel) differs.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IParity210Backend
{
    private IntPtr ResolveParity210Kernel(string name)
    {
        if (_parity210Module == IntPtr.Zero)
            throw new InvalidOperationException(
                "Parity-210 HIP module was not compiled (hipRTC rejected source?). Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
        return kernel;
    }

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------

    public unsafe void Parity210Roll1D(
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize, int shift)
    {
        var kernel = ResolveParity210Kernel("parity210_roll_1d");
        int total = outerSize * axisSize * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize, s = shift;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i; args[5] = &s;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210FlipAxis(
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        var kernel = ResolveParity210Kernel("parity210_flip_axis");
        int total = outerSize * axisSize * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210Triu(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => LaunchTriangularMaskP210("parity210_triu", input, output, batchSize, rows, cols, diagonal);

    public unsafe void Parity210Tril(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => LaunchTriangularMaskP210("parity210_tril", input, output, batchSize, rows, cols, diagonal);

    private unsafe void LaunchTriangularMaskP210(
        string name, IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        var kernel = ResolveParity210Kernel(name);
        int total = batchSize * rows * cols;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int b = batchSize, r = rows, c = cols, d = diagonal;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &b; args[3] = &r; args[4] = &c; args[5] = &d;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210DiagEmbed(
        IGpuBuffer input, IGpuBuffer output,
        int batchSize, int diagLen, int matSize, int offset)
    {
        var kernel = ResolveParity210Kernel("parity210_diag_embed");
        int total = batchSize * matSize * matSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int b = batchSize, d = diagLen, m = matSize, off = offset;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &b; args[3] = &d; args[4] = &m; args[5] = &off;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // CUMULATIVE
    // -----------------------------------------------------------------------

    public unsafe void Parity210CumSum(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLineP210("parity210_cumsum_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210CumProd(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLineP210("parity210_cumprod_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210CumMax(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLineP210("parity210_cummax_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210CumMin(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLineP210("parity210_cummin_axis", input, output, outerSize, axisSize, innerSize);

    public unsafe void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => LaunchCumulativeLineP210("parity210_logcumsumexp_axis", input, output, outerSize, axisSize, innerSize);

    private unsafe void LaunchCumulativeLineP210(
        string name, IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        var kernel = ResolveParity210Kernel(name);
        int total = outerSize * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, a = axisSize, i = innerSize;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &outPtr;
        args[2] = &o; args[3] = &a; args[4] = &i;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // INDEXING
    // -----------------------------------------------------------------------

    public unsafe void Parity210TakeLinear(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outSize, int inputLinearLen)
    {
        var kernel = ResolveParity210Kernel("parity210_take_linear");
        uint grid = (uint)((outSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr outPtr = output.Handle;
        int o = outSize, l = inputLinearLen;
        void** args = stackalloc void*[5];
        args[0] = &inPtr; args[1] = &idxPtr; args[2] = &outPtr;
        args[3] = &o; args[4] = &l;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210TakeAlongDim(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outerSize, int idxAxis, int innerSize, int srcAxis)
    {
        var kernel = ResolveParity210Kernel("parity210_take_along_dim");
        int total = outerSize * idxAxis * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr outPtr = output.Handle;
        int o = outerSize, ia = idxAxis, i = innerSize, sa = srcAxis;
        void** args = stackalloc void*[7];
        args[0] = &inPtr; args[1] = &idxPtr; args[2] = &outPtr;
        args[3] = &o; args[4] = &ia; args[5] = &i; args[6] = &sa;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210IndexAdd(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_add");
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr srcPtr = source.Handle;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &srcPtr;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210IndexCopy(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_copy");
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle; IntPtr srcPtr = source.Handle;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &srcPtr;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210IndexFill(
        IGpuBuffer output, IGpuBuffer indicesInt32, float fillValue,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        var kernel = ResolveParity210Kernel("parity210_index_fill");
        int total = outerSize * idxLen * innerSize;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr idxPtr = indicesInt32.Handle;
        float f = fillValue;
        int o = outerSize, da = dstAxis, i = innerSize, il = idxLen;
        void** args = stackalloc void*[7];
        args[0] = &outPtr; args[1] = &idxPtr; args[2] = &f;
        args[3] = &o; args[4] = &da; args[5] = &i; args[6] = &il;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210MaskedScatter(
        IGpuBuffer output, IGpuBuffer maskInt8, IGpuBuffer prefixSumInt32,
        IGpuBuffer source, int total)
    {
        var kernel = ResolveParity210Kernel("parity210_masked_scatter");
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr outPtr = output.Handle; IntPtr maskPtr = maskInt8.Handle;
        IntPtr prefPtr = prefixSumInt32.Handle; IntPtr srcPtr = source.Handle;
        int n = total;
        void** args = stackalloc void*[5];
        args[0] = &outPtr; args[1] = &maskPtr; args[2] = &prefPtr; args[3] = &srcPtr; args[4] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE BINARY
    // -----------------------------------------------------------------------

    public unsafe void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_hypot", a, b, output, size);

    public unsafe void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_copysign", a, b, output, size);

    public unsafe void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_fmod", a, b, output, size);

    public unsafe void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_remainder", a, b, output, size);

    public unsafe void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_float_power", a, b, output, size);

    public unsafe void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_log_add_exp", a, b, output, size);

    public unsafe void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_log_add_exp2", a, b, output, size);

    public unsafe void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_xlogy", x, y, output, size);

    public unsafe void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size)
        => LaunchBinaryP210("parity210_xlog1py", x, y, output, size);

    private unsafe void LaunchBinaryP210(string name, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        var kernel = ResolveParity210Kernel(name);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle; IntPtr bPtr = b.Handle; IntPtr outPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[4];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr; args[3] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE UNARY SPECIAL
    // -----------------------------------------------------------------------

    public unsafe void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_erfc", input, output, size);

    public unsafe void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_erfinv", input, output, size);

    public unsafe void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_lgamma_approx", input, output, size);

    public unsafe void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_digamma", input, output, size);

    public unsafe void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_i0", input, output, size);

    public unsafe void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_i1", input, output, size);

    public unsafe void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_i0e", input, output, size);

    public unsafe void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_i1e", input, output, size);

    public unsafe void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_is_finite", input, output, size);

    public unsafe void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_is_nan", input, output, size);

    public unsafe void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
        => LaunchUnaryP210("parity210_is_inf", input, output, size);

    private unsafe void LaunchUnaryP210(string name, IGpuBuffer input, IGpuBuffer output, int size)
    {
        var kernel = ResolveParity210Kernel(name);
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int n = size;
        void** args = stackalloc void*[3];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &n;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210NanToNum(
        IGpuBuffer input, IGpuBuffer output, int size,
        float nanVal, float posInfVal, float negInfVal)
    {
        var kernel = ResolveParity210Kernel("parity210_nan_to_num");
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle; IntPtr outPtr = output.Handle;
        int n = size;
        float nv = nanVal, pv = posInfVal, negV = negInfVal;
        void** args = stackalloc void*[6];
        args[0] = &inPtr; args[1] = &outPtr; args[2] = &n;
        args[3] = &nv; args[4] = &pv; args[5] = &negV;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // PAIRWISE
    // -----------------------------------------------------------------------

    public unsafe void Parity210CosineSimilarityLast(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
    {
        var kernel = ResolveParity210Kernel("parity210_cosine_similarity_last");
        uint grid = (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr aPtr = a.Handle; IntPtr bPtr = b.Handle; IntPtr outPtr = output.Handle;
        int nn = n, dd = d;
        float ep = eps;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr;
        args[3] = &nn; args[4] = &dd; args[5] = &ep;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Parity210CdistL2(
        IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
    {
        var kernel = ResolveParity210Kernel("parity210_cdist_l2");
        int total = n * m;
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr x1Ptr = x1.Handle; IntPtr x2Ptr = x2.Handle; IntPtr outPtr = output.Handle;
        int nn = n, mm = m, dd = d;
        void** args = stackalloc void*[6];
        args[0] = &x1Ptr; args[1] = &x2Ptr; args[2] = &outPtr;
        args[3] = &nn; args[4] = &mm; args[5] = &dd;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    // -----------------------------------------------------------------------
    // CLAMP
    // -----------------------------------------------------------------------

    public unsafe void Parity210ClampMinMax(
        IGpuBuffer input, IGpuBuffer lo, IGpuBuffer hi, IGpuBuffer output,
        int size, bool hasLo, bool hasHi)
    {
        var kernel = ResolveParity210Kernel("parity210_clamp_min_max");
        uint grid = (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inPtr = input.Handle;
        IntPtr loPtr = lo?.Handle ?? input.Handle;
        IntPtr hiPtr = hi?.Handle ?? input.Handle;
        IntPtr outPtr = output.Handle;
        int n = size, hl = hasLo ? 1 : 0, hh = hasHi ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &inPtr; args[1] = &loPtr; args[2] = &hiPtr; args[3] = &outPtr;
        args[4] = &n; args[5] = &hl; args[6] = &hh;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }
}
