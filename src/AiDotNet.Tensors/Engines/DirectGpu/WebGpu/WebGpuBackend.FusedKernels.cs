#if NET8_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    // WebGPU fused kernel dispatch via WGSL compute pipelines.
    // Dispatch2BufferAsync(module, source, kernel, inputBuf, outputBuf, uniforms, workSize)

    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz)
    {
        if (sz <= 0) { Fill(o, 0f, 1); return; }
        Dispatch2BufferAsync("ReductionExt", WebGpuKernels.ReductionExtSource, "mean_axis", i, o,
            new float[] { BitConverter.Int32BitsToSingle(1), BitConverter.Int32BitsToSingle(sz), 0, 0 }, 1).GetAwaiter().GetResult();
    }

    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz)
    {
        if (sz <= 0) return;
        Dispatch2BufferAsync("Clamp", WebGpuKernels.ClampSource, "clamp_op", i, o,
            MakeUniform3(sz, mn, mx), sz).GetAwaiter().GetResult();
    }

    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz)
    {
        if (sz <= 0) return;
        Dispatch2BufferAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, "pow_scalar", i, o,
            MakeUniform2(sz, ex), sz).GetAwaiter().GetResult();
    }

    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz)
    {
        if (sz <= 0) return;
        Dispatch2BufferAsync("UnaryMath", WebGpuKernels.UnaryMathSource, "frac_op", i, o,
            MakeUniform2(sz, 0f), sz).GetAwaiter().GetResult();
    }

    public void EyeKernel(IGpuBuffer o, int n)
    {
        if (n <= 0) return;
        Dispatch1BufferAsync("ShapeOps", WebGpuKernels.ShapeOpsSource, "eye_kernel", o,
            new float[] { BitConverter.Int32BitsToSingle(n), 0, 0, 0 }, n * n).GetAwaiter().GetResult();
    }

    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc)
    {
        int total = bs * nc;
        if (total <= 0) return;
        Dispatch2BufferAsync("ShapeOps", WebGpuKernels.ShapeOpsSource, "one_hot_kernel", idx, o,
            new float[] { BitConverter.Int32BitsToSingle(bs), BitConverter.Int32BitsToSingle(nc), 0, 0 }, total).GetAwaiter().GetResult();
    }

    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz)
    {
        if (sz <= 0) return;
        Dispatch3BufferAsync("MaskedFill", WebGpuKernels.MaskedFillSource, "masked_fill", i, m, o,
            new float[] { fv, BitConverter.Int32BitsToSingle(sz) }, sz).GetAwaiter().GetResult();
    }

    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    {
        if (sz <= 0) return;
        Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "equal_to", a, b, o,
            new float[] { BitConverter.Int32BitsToSingle(sz) }, sz).GetAwaiter().GetResult();
    }

    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    {
        if (sz <= 0) return;
        Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "not_equal_to", a, b, o,
            new float[] { BitConverter.Int32BitsToSingle(sz) }, sz).GetAwaiter().GetResult();
    }
    // IEEE classify (isnan/isinf/isfinite): a real WGSL kernel is a HW-validation follow-up; the engine
    // catches this and falls back to the correct CPU path until then.
    public void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size)
    {
        int __total = size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:ClassifyFloat", WebGpuAuditKernels.ClassifyFloat, "main",
            new IGpuBuffer[] { A, C }, new float[] { BitConverter.Int32BitsToSingle(mode), BitConverter.Int32BitsToSingle(size), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn)
    {
        int __total = outerSize*axisOut*innerSize; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:TakeAlongDimF", WebGpuAuditKernels.TakeAlongDimF, "main",
            new IGpuBuffer[] { input, indices, output }, new float[] { BitConverter.Int32BitsToSingle(outerSize), BitConverter.Int32BitsToSingle(axisOut), BitConverter.Int32BitsToSingle(innerSize), BitConverter.Int32BitsToSingle(axisIn) }, __total).GetAwaiter().GetResult();
    }
    public void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        int __total = outerSize*innerSize; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Cross3", WebGpuAuditKernels.Cross3, "main",
            new IGpuBuffer[] { a, b, output }, new float[] { BitConverter.Int32BitsToSingle(outerSize), BitConverter.Int32BitsToSingle(innerSize), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size)
    {
        int __total = size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Ldexp", WebGpuAuditKernels.Ldexp, "main",
            new IGpuBuffer[] { input, exponents, output }, new float[] { BitConverter.Int32BitsToSingle(size), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq)
    {
        int __total = (am*bp)*(an*bq); if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Kron2D", WebGpuAuditKernels.Kron2D, "main",
            new IGpuBuffer[] { a, b, output }, new float[] { BitConverter.Int32BitsToSingle(am), BitConverter.Int32BitsToSingle(an), BitConverter.Int32BitsToSingle(bp), BitConverter.Int32BitsToSingle(bq) }, __total).GetAwaiter().GetResult();
    }
    public void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right)
    {
        int __total = numValues; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:SearchSorted", WebGpuAuditKernels.SearchSorted, "main",
            new IGpuBuffer[] { sortedSeq, values, output }, new float[] { BitConverter.Int32BitsToSingle(seqLen), BitConverter.Int32BitsToSingle(numValues), BitConverter.Int32BitsToSingle(right), 0f }, __total).GetAwaiter().GetResult();
    }
    public void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        int __total = size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:NextAfter", WebGpuAuditKernels.NextAfter, "main",
            new IGpuBuffer[] { a, b, output }, new float[] { BitConverter.Int32BitsToSingle(size), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis)
    {
        int __total = outerSize*dstAxis*innerSize; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:IndexWrite", WebGpuAuditKernels.IndexWrite, "main",
            new IGpuBuffer[] { output, indices, source }, new float[] { fillValue, BitConverter.Int32BitsToSingle(mode), BitConverter.Int32BitsToSingle(outerSize), BitConverter.Int32BitsToSingle(idxAxis), BitConverter.Int32BitsToSingle(innerSize), BitConverter.Int32BitsToSingle(dstAxis), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p)
    {
        int __total = m*n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Cdist", WebGpuAuditKernels.Cdist, "main",
            new IGpuBuffer[] { x1, x2, output }, new float[] { BitConverter.Int32BitsToSingle(m), BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(d), p }, __total).GetAwaiter().GetResult();
    }
    public void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p)
    {
        int __total = n*n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Pdist", WebGpuAuditKernels.Pdist, "main",
            new IGpuBuffer[] { input, output }, new float[] { BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(d), p, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Histc", WebGpuAuditKernels.Histc, "main",
            new IGpuBuffer[] { input, hist }, new float[] { BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(bins), mn, mx }, __total).GetAwaiter().GetResult();
    }
    public void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending)
    {
        int __total = numRows*rowLen; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:BitonicStep", WebGpuAuditKernels.BitonicStep, "main",
            new IGpuBuffer[] { values, indices }, new float[] { BitConverter.Int32BitsToSingle(rowLen), BitConverter.Int32BitsToSingle(k), BitConverter.Int32BitsToSingle(j), BitConverter.Int32BitsToSingle(numRows), BitConverter.Int32BitsToSingle(descending), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen)
    {
        int __total = numRows*copyLen; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:CopyRows", WebGpuAuditKernels.CopyRows, "main",
            new IGpuBuffer[] { src, dst }, new float[] { BitConverter.Int32BitsToSingle(srcRowLen), BitConverter.Int32BitsToSingle(dstRowLen), BitConverter.Int32BitsToSingle(numRows), BitConverter.Int32BitsToSingle(copyLen) }, __total).GetAwaiter().GetResult();
    }
    public void IotaPad(IGpuBuffer idx, int l, int p, int numRows)
    {
        int __total = numRows*p; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:IotaPad", WebGpuAuditKernels.IotaPad, "main",
            new IGpuBuffer[] { idx }, new float[] { BitConverter.Int32BitsToSingle(l), BitConverter.Int32BitsToSingle(p), BitConverter.Int32BitsToSingle(numRows), 0f }, __total).GetAwaiter().GetResult();
    }
    public void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        int __total = batch*numHeads; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Rwkv7Forward", WebGpuAuditKernels.Rwkv7Forward, "main",
            new IGpuBuffer[] { r, k, v, a, b, output, sbuf }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(seqLen), BitConverter.Int32BitsToSingle(modelDim), BitConverter.Int32BitsToSingle(numHeads), BitConverter.Int32BitsToSingle(headDim), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses)
    {
        int __total = rows*numClasses; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:HsoftmaxPaths", WebGpuAuditKernels.HsoftmaxPaths, "main",
            new IGpuBuffer[] { acts, output }, new float[] { BitConverter.Int32BitsToSingle(rows), BitConverter.Int32BitsToSingle(treeDepth), BitConverter.Int32BitsToSingle(numClasses), 0f }, __total).GetAwaiter().GetResult();
    }
    public void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen)
    {
        int __total = numElements; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:IsIn", WebGpuAuditKernels.IsIn, "main",
            new IGpuBuffer[] { elements, sortedTest, mask }, new float[] { BitConverter.Int32BitsToSingle(numElements), BitConverter.Int32BitsToSingle(testLen), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff)
    {
        int __total = blockRows*blockCols; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:CopyBlock2D", WebGpuAuditKernels.CopyBlock2D, "main",
            new IGpuBuffer[] { block, output }, new float[] { BitConverter.Int32BitsToSingle(blockRows), BitConverter.Int32BitsToSingle(blockCols), BitConverter.Int32BitsToSingle(totalCols), BitConverter.Int32BitsToSingle(rowOff), BitConverter.Int32BitsToSingle(colOff), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size)
    {
        int __total = size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Zeta", WebGpuAuditKernels.Zeta, "main",
            new IGpuBuffer[] { x, q, output }, new float[] { BitConverter.Int32BitsToSingle(size), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size)
    {
        int __total = size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Polygamma", WebGpuAuditKernels.Polygamma, "main",
            new IGpuBuffer[] { x, output }, new float[] { BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(size), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:ShiftedDiff", WebGpuAuditKernels.ShiftedDiff, "main",
            new IGpuBuffer[] { x, mask }, new float[] { BitConverter.Int32BitsToSingle(n), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void ReflectPad1d(IGpuBuffer input, IGpuBuffer output, int batch, int l, int lp, int pad)
    {
        int __total = batch*lp; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:ReflectPad1D", WebGpuAuditKernels.ReflectPad1D, "main",
            new IGpuBuffer[] { input, output }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(l), BitConverter.Int32BitsToSingle(lp), BitConverter.Int32BitsToSingle(pad) }, __total).GetAwaiter().GetResult();
    }
    public void StftMagPhase(IGpuBuffer padded, IGpuBuffer window, IGpuBuffer mag, IGpuBuffer phase, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs)
    {
        int __total = batch*numFreqs*numFrames; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:StftMagPhase", WebGpuAuditKernels.StftMagPhase, "main",
            new IGpuBuffer[] { padded, window, mag, phase }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(lp), BitConverter.Int32BitsToSingle(nFft), BitConverter.Int32BitsToSingle(hop), BitConverter.Int32BitsToSingle(numFrames), BitConverter.Int32BitsToSingle(numFreqs), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void PhaseVocoder(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer newMag, IGpuBuffer newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate)
    {
        int __total = leading*nFreqV; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:PhaseVocoder", WebGpuAuditKernels.PhaseVocoder, "main",
            new IGpuBuffer[] { mag, phase, newMag, newPhase }, new float[] { BitConverter.Int32BitsToSingle(leading), BitConverter.Int32BitsToSingle(nFramesV), BitConverter.Int32BitsToSingle(nFreqV), BitConverter.Int32BitsToSingle(outFrames), rate, 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void BuildSpectrum(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer specRe, IGpuBuffer specIm, int batch, int numFreqs, int numFrames, int nFft)
    {
        int __total = batch*numFrames; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:BuildSpectrum", WebGpuAuditKernels.BuildSpectrum, "main",
            new IGpuBuffer[] { mag, phase, specRe, specIm }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(numFreqs), BitConverter.Int32BitsToSingle(numFrames), BitConverter.Int32BitsToSingle(nFft) }, __total).GetAwaiter().GetResult();
    }
    public void IstftFromSpectrum(IGpuBuffer specRe, IGpuBuffer specIm, IGpuBuffer window, IGpuBuffer result, IGpuBuffer windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center)
    {
        int __total = batch*outputLength; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:IstftFromSpectrum", WebGpuAuditKernels.IstftFromSpectrum, "main",
            new IGpuBuffer[] { specRe, specIm, window, result, windowSum }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(numFrames), BitConverter.Int32BitsToSingle(nFft), BitConverter.Int32BitsToSingle(hop), BitConverter.Int32BitsToSingle(outputLength), BitConverter.Int32BitsToSingle(center), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void IstftNormalize(IGpuBuffer result, IGpuBuffer windowSum, int total)
    {
        int __total = total; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:IstftNormalize", WebGpuAuditKernels.IstftNormalize, "main",
            new IGpuBuffer[] { result, windowSum }, new float[] { BitConverter.Int32BitsToSingle(total), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Histogramdd", WebGpuAuditKernels.Histogramdd, "main",
            new IGpuBuffer[] { samples, hist, bins, mins, maxs }, new float[] { BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(d), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:MasksToBoxes", WebGpuAuditKernels.MasksToBoxes, "main",
            new IGpuBuffer[] { masks, output }, new float[] { BitConverter.Int32BitsToSingle(n), BitConverter.Int32BitsToSingle(h), BitConverter.Int32BitsToSingle(w), 0f }, __total).GetAwaiter().GetResult();
    }
    public void PairwiseIou(IGpuBuffer boxes, IGpuBuffer iou, int n)
    {
        int __total = n*n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:PairwiseIou", WebGpuAuditKernels.PairwiseIou, "main",
            new IGpuBuffer[] { boxes, iou }, new float[] { BitConverter.Int32BitsToSingle(n), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void LogicalOp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int mode, int n)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:LogicalOp", WebGpuAuditKernels.LogicalOp, "main",
            new IGpuBuffer[] { a, b, output }, new float[] { BitConverter.Int32BitsToSingle(mode), BitConverter.Int32BitsToSingle(n), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void LogicalNot(IGpuBuffer a, IGpuBuffer output, int n)
    {
        int __total = n; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:LogicalNot", WebGpuAuditKernels.LogicalNot, "main",
            new IGpuBuffer[] { a, output }, new float[] { BitConverter.Int32BitsToSingle(n), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void GridSampleBackwardInputNhwc(IGpuBuffer gradOut, IGpuBuffer grid, IGpuBuffer gradIn, int batch, int h, int w, int c, int outH, int outW)
    {
        int __total = batch*outH*outW*c; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:GridSampleBackwardInput", WebGpuAuditKernels.GridSampleBackwardInput, "main",
            new IGpuBuffer[] { gradOut, grid, gradIn }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(h), BitConverter.Int32BitsToSingle(w), BitConverter.Int32BitsToSingle(c), BitConverter.Int32BitsToSingle(outH), BitConverter.Int32BitsToSingle(outW), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void GridSampleBackwardGridNhwc(IGpuBuffer gradOut, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer gradGrid, int batch, int h, int w, int c, int outH, int outW)
    {
        int __total = batch*outH*outW; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:GridSampleBackwardGrid", WebGpuAuditKernels.GridSampleBackwardGrid, "main",
            new IGpuBuffer[] { gradOut, input, grid, gradGrid }, new float[] { BitConverter.Int32BitsToSingle(batch), BitConverter.Int32BitsToSingle(h), BitConverter.Int32BitsToSingle(w), BitConverter.Int32BitsToSingle(c), BitConverter.Int32BitsToSingle(outH), BitConverter.Int32BitsToSingle(outW), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode)
    {
        int __total = outerSize*srcDim*innerSize; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:ScatterReduce", WebGpuAuditKernels.ScatterReduce, "main",
            new IGpuBuffer[] { output, source, index }, new float[] { BitConverter.Int32BitsToSingle(outerSize), BitConverter.Int32BitsToSingle(srcDim), BitConverter.Int32BitsToSingle(dstDim), BitConverter.Int32BitsToSingle(innerSize), BitConverter.Int32BitsToSingle(mode), 0f, 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step)
    {
        int __total = outerSize*nWindows*innerSize*size; if (__total <= 0) return;
        DispatchNBufferAsync("Audit:Unfold", WebGpuAuditKernels.Unfold, "main",
            new IGpuBuffer[] { src, dst }, new float[] { BitConverter.Int32BitsToSingle(outerSize), BitConverter.Int32BitsToSingle(dimSize), BitConverter.Int32BitsToSingle(innerSize), BitConverter.Int32BitsToSingle(nWindows), BitConverter.Int32BitsToSingle(size), BitConverter.Int32BitsToSingle(step), 0f, 0f }, __total).GetAwaiter().GetResult();
    }
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N)
    {
        int total = M * N;
        if (total <= 0) return;
        Dispatch3BufferAsync("OuterProduct", WebGpuKernels.OuterProductSource, "outer_product", a, b, o,
            MakeUniformInts2(M, N), total).GetAwaiter().GetResult();
    }

    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim)
    {
        if (bs <= 0 || dim <= 0) return;
        Dispatch3BufferAsync("BatchDotProduct", WebGpuKernels.BatchDotProductSource, "batch_dot_product", a, b, o,
            MakeUniformInts2(bs, dim), bs).GetAwaiter().GetResult();
    }

    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
        => DispatchGatedActivation("glu_forward", i, o, os, hd);

    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
        => DispatchGatedActivation("geglu_forward", i, o, os, hd);

    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
        => DispatchGatedActivation("reglu_forward", i, o, os, hd);

    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
        => DispatchGatedActivation("swiglu_forward", i, o, os, hd);

    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz)
    {
        if (sz <= 0) return;
        Dispatch3BufferAsync("Loss", WebGpuKernels.LossSource, "binary_cross_entropy", p, t, l,
            MakeUniform2(sz, 1e-7f), sz).GetAwaiter().GetResult();
    }

    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz)
        => DispatchScalarResident("sub_scalar", i, o, sc, sz);

    public void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz)
        => DispatchScalarResident("add_scalar", i, o, sc, sz);

    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
        => DispatchBroadcastResident("broadcast_add_last", a, b, o, os, isz);

    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
        => DispatchBroadcastResident("broadcast_sub_last", a, b, o, os, isz);

    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
        => DispatchBroadcastResident("broadcast_mul_last", a, b, o, os, isz);

    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
        => DispatchBroadcastResident("broadcast_div_last", a, b, o, os, isz);

    private void DispatchGatedActivation(string kernelName, IGpuBuffer input, IGpuBuffer output, int outerSize, int halfDim)
    {
        int total = outerSize * halfDim;
        if (total <= 0) return;
        Dispatch2BufferAsync("GatedActivation", WebGpuKernels.GatedActivationSource, kernelName, input, output,
            MakeUniformInts2(outerSize, halfDim), total).GetAwaiter().GetResult();
    }

    private void DispatchScalarResident(string kernelName, IGpuBuffer input, IGpuBuffer output, float scalar, int size)
    {
        if (size <= 0) return;
        Dispatch2BufferAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, kernelName, input, output,
            MakeUniform2(size, scalar), size).GetAwaiter().GetResult();
    }

    private void DispatchBroadcastResident(
        string kernelName, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int outerSize, int innerSize)
    {
        int total = outerSize * innerSize;
        if (total <= 0) return;
        Dispatch3BufferAsync("BroadcastExt", WebGpuKernels.BroadcastExtSource, kernelName, left, right, output,
            MakeUniformInts2(outerSize, innerSize), total).GetAwaiter().GetResult();
    }
}
#endif
