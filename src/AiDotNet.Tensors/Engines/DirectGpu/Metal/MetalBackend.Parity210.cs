// Copyright (c) AiDotNet. All rights reserved.
// Metal launcher shims for the parity-210 kernels. Uses the same
// GetPipeline / encoder.SetBuffer / SetBytes / DispatchThreadgroups
// pattern as the rest of MetalBackend — just threaded through the
// parity210 library handle.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IParity210Backend
{
    private const string Parity210LibName = "Parity210";

    private MetalPipelineState GetParity210Pipeline(string kernelName)
    {
        if (_parity210Library == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal Parity-210 library was not compiled. Falling back to CPU reference.");
        return GetPipeline(Parity210LibName, _parity210Library, kernelName);
    }

    private void RequireMetalPair(IGpuBuffer a, IGpuBuffer b,
        out MetalGpuBuffer aBuf, out MetalGpuBuffer bBuf)
    {
        if (a is not MetalGpuBuffer ma || b is not MetalGpuBuffer mb)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        aBuf = ma; bBuf = mb;
    }

    private void RequireMetal3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        out MetalGpuBuffer aBuf, out MetalGpuBuffer bBuf, out MetalGpuBuffer cBuf)
    {
        if (a is not MetalGpuBuffer ma || b is not MetalGpuBuffer mb || c is not MetalGpuBuffer mc)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        aBuf = ma; bBuf = mb; cBuf = mc;
    }

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------

    public void Parity210Roll1D(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize, int shift)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        int total = outerSize * axisSize * innerSize;
        var pipeline = GetParity210Pipeline("parity210_roll_1d");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(outerSize, 2);
        encoder.SetBytes(axisSize, 3);
        encoder.SetBytes(innerSize, 4);
        encoder.SetBytes(shift, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Parity210FlipAxis(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        int total = outerSize * axisSize * innerSize;
        var pipeline = GetParity210Pipeline("parity210_flip_axis");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(outerSize, 2);
        encoder.SetBytes(axisSize, 3);
        encoder.SetBytes(innerSize, 4);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Parity210Triu(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => DispatchTriuTril("parity210_triu", input, output, batchSize, rows, cols, diagonal);

    public void Parity210Tril(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
        => DispatchTriuTril("parity210_tril", input, output, batchSize, rows, cols, diagonal);

    private void DispatchTriuTril(string name,
        IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        int total = batchSize * rows * cols;
        var pipeline = GetParity210Pipeline(name);
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(batchSize, 2);
        encoder.SetBytes(rows, 3);
        encoder.SetBytes(cols, 4);
        encoder.SetBytes(diagonal, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210DiagEmbed(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int diagLen, int matSize, int offset)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        int total = batchSize * matSize * matSize;
        var pipeline = GetParity210Pipeline("parity210_diag_embed");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(batchSize, 2);
        encoder.SetBytes(diagLen, 3);
        encoder.SetBytes(matSize, 4);
        encoder.SetBytes(offset, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // CUMULATIVE  (one thread per (outer, inner) line)
    // -----------------------------------------------------------------------

    public void Parity210CumSum(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulative("parity210_cumsum_axis", input, output, outerSize, axisSize, innerSize);

    public void Parity210CumProd(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulative("parity210_cumprod_axis", input, output, outerSize, axisSize, innerSize);

    public void Parity210CumMax(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulative("parity210_cummax_axis", input, output, outerSize, axisSize, innerSize);

    public void Parity210CumMin(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulative("parity210_cummin_axis", input, output, outerSize, axisSize, innerSize);

    public void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulative("parity210_logcumsumexp_axis", input, output, outerSize, axisSize, innerSize);

    private void DispatchCumulative(string name,
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        ThrowIfDisposed();
        // Empty-axis guard — matches CUDA / HIP / OpenCL.
        if (axisSize <= 0 || outerSize <= 0 || innerSize <= 0) return;
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        int total = outerSize * innerSize;
        var pipeline = GetParity210Pipeline(name);
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(outerSize, 2);
        encoder.SetBytes(axisSize, 3);
        encoder.SetBytes(innerSize, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE BINARY
    // -----------------------------------------------------------------------

    public void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_hypot", a, b, o, size);

    public void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_copysign", a, b, o, size);

    public void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_fmod", a, b, o, size);

    public void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_remainder", a, b, o, size);

    public void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_float_power", a, b, o, size);

    public void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_log_add_exp", a, b, o, size);

    public void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => DispatchBinary("parity210_log_add_exp2", a, b, o, size);

    public void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => DispatchBinary("parity210_xlogy", x, y, o, size);

    public void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => DispatchBinary("parity210_xlog1py", x, y, o, size);

    private void DispatchBinary(string name, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
    {
        ThrowIfDisposed();
        RequireMetal3(a, b, o, out var aBuf, out var bBuf, out var oBuf);
        var pipeline = GetParity210Pipeline(name);
        var (tgr, tpg) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuf, 0);
        encoder.SetBuffer(bBuf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE UNARY SPECIAL
    // -----------------------------------------------------------------------

    public void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_erfc", input, output, size);

    public void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_erfinv", input, output, size);

    public void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_lgamma_approx", input, output, size);

    public void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_digamma", input, output, size);

    public void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_i0", input, output, size);

    public void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_i1", input, output, size);

    public void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_i0e", input, output, size);

    public void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_i1e", input, output, size);

    public void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_is_finite", input, output, size);

    public void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_is_nan", input, output, size);

    public void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
        => DispatchUnary("parity210_is_inf", input, output, size);

    private void DispatchUnary(string name, IGpuBuffer input, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        var pipeline = GetParity210Pipeline(name);
        var (tgr, tpg) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(size, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210NanToNum(IGpuBuffer input, IGpuBuffer output, int size,
        float nanVal, float posInfVal, float negInfVal)
    {
        ThrowIfDisposed();
        RequireMetalPair(input, output, out var inBuf, out var outBuf);
        var pipeline = GetParity210Pipeline("parity210_nan_to_num");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(size, 2);
        encoder.SetBytes(nanVal, 3);
        encoder.SetBytes(posInfVal, 4);
        encoder.SetBytes(negInfVal, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // PAIRWISE
    // -----------------------------------------------------------------------

    public void Parity210CosineSimilarityLast(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
    {
        ThrowIfDisposed();
        RequireMetal3(a, b, output, out var aBuf, out var bBuf, out var oBuf);
        var pipeline = GetParity210Pipeline("parity210_cosine_similarity_last");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(n);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuf, 0);
        encoder.SetBuffer(bBuf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes(n, 3);
        encoder.SetBytes(d, 4);
        encoder.SetBytes(eps, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210CdistL2(
        IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
    {
        ThrowIfDisposed();
        RequireMetal3(x1, x2, output, out var x1Buf, out var x2Buf, out var oBuf);
        int total = n * m;
        var pipeline = GetParity210Pipeline("parity210_cdist_l2");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(x1Buf, 0);
        encoder.SetBuffer(x2Buf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes(n, 3);
        encoder.SetBytes(m, 4);
        encoder.SetBytes(d, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // Indexing (take / index_add / index_copy / index_fill / masked_scatter)
    // Uses Metal's atomic_float for index_add; requires MSL 2.4+.
    // -----------------------------------------------------------------------

    public void Parity210TakeLinear(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outSize, int inputLinearLen)
    {
        ThrowIfDisposed();
        RequireMetal3(input, indicesInt32, output, out var inBuf, out var idxBuf, out var oBuf);
        var pipeline = GetParity210Pipeline("parity210_take_linear");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(outSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(idxBuf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes(outSize, 3);
        encoder.SetBytes(inputLinearLen, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210TakeAlongDim(
        IGpuBuffer input, IGpuBuffer indicesInt32, IGpuBuffer output,
        int outerSize, int idxAxis, int innerSize, int srcAxis)
    {
        ThrowIfDisposed();
        RequireMetal3(input, indicesInt32, output, out var inBuf, out var idxBuf, out var oBuf);
        int total = outerSize * idxAxis * innerSize;
        var pipeline = GetParity210Pipeline("parity210_take_along_dim");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(idxBuf, 1);
        encoder.SetBuffer(oBuf, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(idxAxis, 4);
        encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(srcAxis, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210IndexAdd(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        ThrowIfDisposed();
        RequireMetal3(output, indicesInt32, source, out var oBuf, out var idxBuf, out var sBuf);
        int total = outerSize * idxLen * innerSize;
        var pipeline = GetParity210Pipeline("parity210_index_add");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(oBuf, 0);
        encoder.SetBuffer(idxBuf, 1);
        encoder.SetBuffer(sBuf, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(dstAxis, 4);
        encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(idxLen, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210IndexCopy(
        IGpuBuffer output, IGpuBuffer indicesInt32, IGpuBuffer source,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        ThrowIfDisposed();
        RequireMetal3(output, indicesInt32, source, out var oBuf, out var idxBuf, out var sBuf);
        int total = outerSize * idxLen * innerSize;
        var pipeline = GetParity210Pipeline("parity210_index_copy");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(oBuf, 0);
        encoder.SetBuffer(idxBuf, 1);
        encoder.SetBuffer(sBuf, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(dstAxis, 4);
        encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(idxLen, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210IndexFill(
        IGpuBuffer output, IGpuBuffer indicesInt32, float fillValue,
        int outerSize, int dstAxis, int innerSize, int idxLen)
    {
        ThrowIfDisposed();
        RequireMetalPair(output, indicesInt32, out var oBuf, out var idxBuf);
        int total = outerSize * idxLen * innerSize;
        var pipeline = GetParity210Pipeline("parity210_index_fill");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(oBuf, 0);
        encoder.SetBuffer(idxBuf, 1);
        encoder.SetBytes(fillValue, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(dstAxis, 4);
        encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(idxLen, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Parity210MaskedScatter(
        IGpuBuffer output, IGpuBuffer maskInt8, IGpuBuffer prefixSumInt32,
        IGpuBuffer source, int total)
    {
        ThrowIfDisposed();
        if (output is not MetalGpuBuffer oBuf || maskInt8 is not MetalGpuBuffer mBuf
            || prefixSumInt32 is not MetalGpuBuffer pBuf || source is not MetalGpuBuffer sBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetParity210Pipeline("parity210_masked_scatter");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(oBuf, 0);
        encoder.SetBuffer(mBuf, 1);
        encoder.SetBuffer(pBuf, 2);
        encoder.SetBuffer(sBuf, 3);
        encoder.SetBytes(total, 4);
        encoder.SetBytes(source.Size, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    // -----------------------------------------------------------------------
    // Clamp tensor-bounds
    // -----------------------------------------------------------------------

    public void Parity210ClampMinMax(
        IGpuBuffer input, IGpuBuffer lo, IGpuBuffer hi, IGpuBuffer output,
        int size, bool hasLo, bool hasHi)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");

        var pipeline = GetParity210Pipeline("parity210_clamp_min_max");
        var (tgr, tpg) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inBuf, 0);
        // lo / hi may be null when hasLo / hasHi are false — reuse the input
        // buffer in that slot so the kernel has valid pointers even though
        // the has* flag tells it to skip those reads.
        var loBuf = (lo as MetalGpuBuffer) ?? inBuf;
        var hiBuf = (hi as MetalGpuBuffer) ?? inBuf;
        encoder.SetBuffer(loBuf, 1);
        encoder.SetBuffer(hiBuf, 2);
        encoder.SetBuffer(oBuf, 3);
        encoder.SetBytes(size, 4);
        encoder.SetBytes(hasLo ? 1 : 0, 5);
        encoder.SetBytes(hasHi ? 1 : 0, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
}
