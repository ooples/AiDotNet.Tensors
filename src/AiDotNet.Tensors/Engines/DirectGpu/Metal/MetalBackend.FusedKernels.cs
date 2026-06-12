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
    public void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_classify_float");
        int __total = size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)A, 0);
        encoder.SetBuffer((MetalGpuBuffer)C, 1);
        encoder.SetBytes(mode, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_take_along_dim_f");
        int __total = outerSize*axisOut*innerSize; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(axisOut, 4);
        encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(axisIn, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_cross3");
        int __total = outerSize*innerSize; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(innerSize, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_ldexp");
        int __total = size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)exponents, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_kron2d");
        int __total = (am*bp)*(an*bq); if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(am, 3);
        encoder.SetBytes(an, 4);
        encoder.SetBytes(bp, 5);
        encoder.SetBytes(bq, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_search_sorted");
        int __total = numValues; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)sortedSeq, 0);
        encoder.SetBuffer((MetalGpuBuffer)values, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(seqLen, 3);
        encoder.SetBytes(numValues, 4);
        encoder.SetBytes(right, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_next_after");
        int __total = size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_index_write");
        int __total = outerSize*idxAxis*innerSize; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)output, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBuffer((MetalGpuBuffer)source, 2);
        encoder.SetBytes(fillValue, 3);
        encoder.SetBytes(mode, 4);
        encoder.SetBytes(outerSize, 5);
        encoder.SetBytes(idxAxis, 6);
        encoder.SetBytes(innerSize, 7);
        encoder.SetBytes(dstAxis, 8);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_cdist");
        int __total = m*n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)x1, 0);
        encoder.SetBuffer((MetalGpuBuffer)x2, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(m, 3);
        encoder.SetBytes(n, 4);
        encoder.SetBytes(d, 5);
        encoder.SetBytes(p, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_pdist");
        int __total = n*n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(d, 3);
        encoder.SetBytes(p, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_histc");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)hist, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(bins, 3);
        encoder.SetBytes(mn, 4);
        encoder.SetBytes(mx, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_bitonic_step");
        int __total = numRows*rowLen; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)values, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBytes(rowLen, 2);
        encoder.SetBytes(k, 3);
        encoder.SetBytes(j, 4);
        encoder.SetBytes(numRows, 5);
        encoder.SetBytes(descending, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_copy_rows");
        int __total = numRows*copyLen; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)src, 0);
        encoder.SetBuffer((MetalGpuBuffer)dst, 1);
        encoder.SetBytes(srcRowLen, 2);
        encoder.SetBytes(dstRowLen, 3);
        encoder.SetBytes(numRows, 4);
        encoder.SetBytes(copyLen, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void IotaPad(IGpuBuffer idx, int l, int p, int numRows)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_iota_pad");
        int __total = numRows*p; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)idx, 0);
        encoder.SetBytes(l, 1);
        encoder.SetBytes(p, 2);
        encoder.SetBytes(numRows, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_rwkv7_forward");
        int __total = batch*numHeads; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)r, 0);
        encoder.SetBuffer((MetalGpuBuffer)k, 1);
        encoder.SetBuffer((MetalGpuBuffer)v, 2);
        encoder.SetBuffer((MetalGpuBuffer)a, 3);
        encoder.SetBuffer((MetalGpuBuffer)b, 4);
        encoder.SetBuffer((MetalGpuBuffer)output, 5);
        encoder.SetBuffer((MetalGpuBuffer)sbuf, 6);
        encoder.SetBytes(batch, 7);
        encoder.SetBytes(seqLen, 8);
        encoder.SetBytes(modelDim, 9);
        encoder.SetBytes(numHeads, 10);
        encoder.SetBytes(headDim, 11);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_hsoftmax_paths");
        int __total = rows*numClasses; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)acts, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(rows, 2);
        encoder.SetBytes(treeDepth, 3);
        encoder.SetBytes(numClasses, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_isin");
        int __total = numElements; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)elements, 0);
        encoder.SetBuffer((MetalGpuBuffer)sortedTest, 1);
        encoder.SetBuffer((MetalGpuBuffer)mask, 2);
        encoder.SetBytes(numElements, 3);
        encoder.SetBytes(testLen, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_copy_block_2d");
        int __total = blockRows*blockCols; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)block, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(blockRows, 2);
        encoder.SetBytes(blockCols, 3);
        encoder.SetBytes(totalCols, 4);
        encoder.SetBytes(rowOff, 5);
        encoder.SetBytes(colOff, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_zeta");
        int __total = size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)x, 0);
        encoder.SetBuffer((MetalGpuBuffer)q, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_polygamma");
        int __total = size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)x, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(size, 3);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_shifted_diff");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)x, 0);
        encoder.SetBuffer((MetalGpuBuffer)mask, 1);
        encoder.SetBytes(n, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void ReflectPad1d(IGpuBuffer input, IGpuBuffer output, int batch, int l, int lp, int pad)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_reflect_pad_1d");
        int __total = batch*lp; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(batch, 2);
        encoder.SetBytes(l, 3);
        encoder.SetBytes(lp, 4);
        encoder.SetBytes(pad, 5);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void StftMagPhase(IGpuBuffer padded, IGpuBuffer window, IGpuBuffer mag, IGpuBuffer phase, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_stft_mag_phase");
        int __total = batch*numFreqs*numFrames; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)padded, 0);
        encoder.SetBuffer((MetalGpuBuffer)window, 1);
        encoder.SetBuffer((MetalGpuBuffer)mag, 2);
        encoder.SetBuffer((MetalGpuBuffer)phase, 3);
        encoder.SetBytes(batch, 4);
        encoder.SetBytes(lp, 5);
        encoder.SetBytes(nFft, 6);
        encoder.SetBytes(hop, 7);
        encoder.SetBytes(numFrames, 8);
        encoder.SetBytes(numFreqs, 9);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void PhaseVocoder(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer newMag, IGpuBuffer newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_phase_vocoder");
        int __total = leading*nFreqV; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)mag, 0);
        encoder.SetBuffer((MetalGpuBuffer)phase, 1);
        encoder.SetBuffer((MetalGpuBuffer)newMag, 2);
        encoder.SetBuffer((MetalGpuBuffer)newPhase, 3);
        encoder.SetBytes(leading, 4);
        encoder.SetBytes(nFramesV, 5);
        encoder.SetBytes(nFreqV, 6);
        encoder.SetBytes(outFrames, 7);
        encoder.SetBytes(rate, 8);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void BuildSpectrum(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer specRe, IGpuBuffer specIm, int batch, int numFreqs, int numFrames, int nFft)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_build_spectrum");
        int __total = batch*numFrames; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)mag, 0);
        encoder.SetBuffer((MetalGpuBuffer)phase, 1);
        encoder.SetBuffer((MetalGpuBuffer)specRe, 2);
        encoder.SetBuffer((MetalGpuBuffer)specIm, 3);
        encoder.SetBytes(batch, 4);
        encoder.SetBytes(numFreqs, 5);
        encoder.SetBytes(numFrames, 6);
        encoder.SetBytes(nFft, 7);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void IstftFromSpectrum(IGpuBuffer specRe, IGpuBuffer specIm, IGpuBuffer window, IGpuBuffer result, IGpuBuffer windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_istft_from_spectrum");
        int __total = batch*numFrames*nFft; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)specRe, 0);
        encoder.SetBuffer((MetalGpuBuffer)specIm, 1);
        encoder.SetBuffer((MetalGpuBuffer)window, 2);
        encoder.SetBuffer((MetalGpuBuffer)result, 3);
        encoder.SetBuffer((MetalGpuBuffer)windowSum, 4);
        encoder.SetBytes(batch, 5);
        encoder.SetBytes(numFrames, 6);
        encoder.SetBytes(nFft, 7);
        encoder.SetBytes(hop, 8);
        encoder.SetBytes(outputLength, 9);
        encoder.SetBytes(center, 10);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void IstftNormalize(IGpuBuffer result, IGpuBuffer windowSum, int total)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_istft_normalize");
        int __total = total; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)result, 0);
        encoder.SetBuffer((MetalGpuBuffer)windowSum, 1);
        encoder.SetBytes(total, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_histogramdd");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)samples, 0);
        encoder.SetBuffer((MetalGpuBuffer)hist, 1);
        encoder.SetBuffer((MetalGpuBuffer)bins, 2);
        encoder.SetBuffer((MetalGpuBuffer)mins, 3);
        encoder.SetBuffer((MetalGpuBuffer)maxs, 4);
        encoder.SetBytes(n, 5);
        encoder.SetBytes(d, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_masks_to_boxes");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)masks, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(n, 2);
        encoder.SetBytes(h, 3);
        encoder.SetBytes(w, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void PairwiseIou(IGpuBuffer boxes, IGpuBuffer iou, int n)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_pairwise_iou");
        int __total = n*n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)boxes, 0);
        encoder.SetBuffer((MetalGpuBuffer)iou, 1);
        encoder.SetBytes(n, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void LogicalOp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int mode, int n)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_logical_op");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(mode, 3);
        encoder.SetBytes(n, 4);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void LogicalNot(IGpuBuffer a, IGpuBuffer output, int n)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_logical_not");
        int __total = n; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(n, 2);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void GridSampleBackwardInputNhwc(IGpuBuffer gradOut, IGpuBuffer grid, IGpuBuffer gradIn, int batch, int h, int w, int c, int outH, int outW)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_gridsample_backward_input");
        int __total = batch*outH*outW*c; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)gradOut, 0);
        encoder.SetBuffer((MetalGpuBuffer)grid, 1);
        encoder.SetBuffer((MetalGpuBuffer)gradIn, 2);
        encoder.SetBytes(batch, 3);
        encoder.SetBytes(h, 4);
        encoder.SetBytes(w, 5);
        encoder.SetBytes(c, 6);
        encoder.SetBytes(outH, 7);
        encoder.SetBytes(outW, 8);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void GridSampleBackwardGridNhwc(IGpuBuffer gradOut, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer gradGrid, int batch, int h, int w, int c, int outH, int outW)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_gridsample_backward_grid");
        int __total = batch*outH*outW; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)gradOut, 0);
        encoder.SetBuffer((MetalGpuBuffer)input, 1);
        encoder.SetBuffer((MetalGpuBuffer)grid, 2);
        encoder.SetBuffer((MetalGpuBuffer)gradGrid, 3);
        encoder.SetBytes(batch, 4);
        encoder.SetBytes(h, 5);
        encoder.SetBytes(w, 6);
        encoder.SetBytes(c, 7);
        encoder.SetBytes(outH, 8);
        encoder.SetBytes(outW, 9);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_scatter_reduce");
        int __total = outerSize*srcDim*innerSize; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)output, 0);
        encoder.SetBuffer((MetalGpuBuffer)source, 1);
        encoder.SetBuffer((MetalGpuBuffer)index, 2);
        encoder.SetBytes(outerSize, 3);
        encoder.SetBytes(srcDim, 4);
        encoder.SetBytes(dstDim, 5);
        encoder.SetBytes(innerSize, 6);
        encoder.SetBytes(mode, 7);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
    public void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step)
    {
        ThrowIfDisposed();
        var pipeline = GetParity210Pipeline("parity210_unfold");
        int __total = outerSize*nWindows*innerSize*size; if (__total <= 0) return;
        var (tgr, tpg) = pipeline.Calculate1DDispatch(__total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)src, 0);
        encoder.SetBuffer((MetalGpuBuffer)dst, 1);
        encoder.SetBytes(outerSize, 2);
        encoder.SetBytes(dimSize, 3);
        encoder.SetBytes(innerSize, 4);
        encoder.SetBytes(nWindows, 5);
        encoder.SetBytes(size, 6);
        encoder.SetBytes(step, 7);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
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
