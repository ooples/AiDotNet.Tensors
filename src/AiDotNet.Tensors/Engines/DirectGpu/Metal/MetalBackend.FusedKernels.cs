namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    // All methods use Metal GPU dispatch via GetPipeline + DispatchThreadgroups.
    // CPU fallback only for ops where MSL kernel source doesn't exist yet.

    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("Reduction", _reductionLibrary, "mean_axis", i, o, sz); }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) { SimpleUnary("Activation", _activationLibrary, "clamp", i, o, sz); }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "pow_scalar", i, o, sz); }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "frac_kernel", i, o, sz); }
    public void EyeKernel(IGpuBuffer o, int n) { SimpleGenerate("Reduction", _reductionLibrary, "eye_kernel", o, n*n); }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { SimpleUnary("Reduction", _reductionLibrary, "one_hot_kernel", idx, o, bs*nc); }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "masked_fill_kernel", i, m, o, sz); }
    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "equals_kernel", a, b, o, sz); }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { SimpleBinary("ElementWise", _elementWiseLibrary, "not_equals_kernel", a, b, o, sz); }
    // IEEE classify (isnan/isinf/isfinite): a real Metal MSL kernel is a HW-validation follow-up; the
    // engine catches this and falls back to the correct CPU path until then.
    public void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size) => throw new NotSupportedException("ClassifyFloat not yet implemented on the Metal backend.");
    public void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn) => throw new NotSupportedException("TakeAlongDim not yet implemented on the Metal backend.");
    public void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) => throw new NotSupportedException("Cross3 not yet implemented on the Metal backend.");
    public void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size) => throw new NotSupportedException("Ldexp not yet implemented on the Metal backend.");
    public void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq) => throw new NotSupportedException("Kron2D not yet implemented on the Metal backend.");
    public void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right) => throw new NotSupportedException("SearchSorted not yet implemented on the Metal backend.");
    public void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) => throw new NotSupportedException("NextAfter not yet implemented on the Metal backend.");
    public void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis) => throw new NotSupportedException("IndexWrite not yet implemented on the Metal backend.");
    public void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p) => throw new NotSupportedException("CDist not yet implemented on the Metal backend.");
    public void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p) => throw new NotSupportedException("PDist not yet implemented on the Metal backend.");
    public void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx) => throw new NotSupportedException("Histc not yet implemented on the Metal backend.");
    public void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending) => throw new NotSupportedException("BitonicStep not yet implemented on the Metal backend.");
    public void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen) => throw new NotSupportedException("CopyRows not yet implemented on the Metal backend.");
    public void IotaPad(IGpuBuffer idx, int l, int p, int numRows) => throw new NotSupportedException("IotaPad not yet implemented on the Metal backend.");
    public void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim) => throw new NotSupportedException("Rwkv7Forward not yet implemented on the Metal backend.");
    public void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses) => throw new NotSupportedException("HierarchicalSoftmaxPaths not yet implemented on the Metal backend.");
    public void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen) => throw new NotSupportedException("IsIn not yet implemented on the Metal backend.");
    public void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff) => throw new NotSupportedException("CopyBlock2D not yet implemented on the Metal backend.");
    public void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size) => throw new NotSupportedException("Zeta not yet implemented on the Metal backend.");
    public void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size) => throw new NotSupportedException("Polygamma not yet implemented on the Metal backend.");
    public void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n) => throw new NotSupportedException("ShiftedDiff not yet implemented on the Metal backend.");
    public void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d) => throw new NotSupportedException("HistogramDD not yet implemented on the Metal backend.");
    public void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w) => throw new NotSupportedException("MasksToBoxes not yet implemented on the Metal backend.");
    public void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode) => throw new NotSupportedException("ScatterReduce not yet implemented on the Metal backend.");
    public void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step) => throw new NotSupportedException("Unfold not yet implemented on the Metal backend.");
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) { SimpleBinary("Reduction", _reductionLibrary, "outer_product", a, b, o, M*N); }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) { SimpleBinary("Reduction", _reductionLibrary, "batch_dot_product", a, b, o, bs); }
    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "glu_forward", i, o, os*hd); }
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "geglu_forward", i, o, os*hd); }
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "reglu_forward", i, o, os*hd); }
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) { SimpleUnary("Activation", _activationLibrary, "swiglu_forward", i, o, os*hd); }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { SimpleBinary("Loss", _lossLibrary, "bce_loss", p, t, l, sz); }
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { SimpleUnary("ElementWise", _elementWiseLibrary, "sub_scalar", i, o, sz); }
    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_add_last", a, b, o, os*isz); }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_sub_last", a, b, o, os*isz); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_mul_last", a, b, o, os*isz); }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { SimpleBinary("ElementWise", _elementWiseLibrary, "broadcast_div_last", a, b, o, os*isz); }

    private void SimpleUnary(string lib, System.IntPtr library, string kernel, IGpuBuffer i, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (i is not MetalGpuBuffer ib || o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ib, 0);
        encoder.SetBuffer(ob, 1);
        encoder.SetBytes((uint)sz, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void SimpleBinary(string lib, System.IntPtr library, string kernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer ab || b is not MetalGpuBuffer bb || o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ab, 0);
        encoder.SetBuffer(bb, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)sz, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void SimpleGenerate(string lib, System.IntPtr library, string kernel, IGpuBuffer o, int sz)
    {
        ThrowIfDisposed();
        if (o is not MetalGpuBuffer ob)
            throw new System.ArgumentException("Buffer must be MetalGpuBuffer");
        var pipeline = GetPipeline(lib, library, kernel);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(sz);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ob, 0);
        encoder.SetBytes((uint)sz, 1);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
