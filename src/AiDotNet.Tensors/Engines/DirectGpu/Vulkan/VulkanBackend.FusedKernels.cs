// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend fused kernel implementations for VulkanBackend.
// Dispatches to runtime-compiled GLSL compute shaders via VulkanGlslCompiler (libshaderc).

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend
{
    /// <summary>Reinterpret float bits as uint (net471-safe alternative to BitConverter.SingleToUInt32Bits).</summary>
    private static unsafe uint FloatBits(float value) { return *(uint*)&value; }

    #region Fused Reductions

    // Reduce* methods: dispatch 1 workgroup, push {outerSize=1, reduceSize=sz}
    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.MeanAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceProduct(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.ProductAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceNormL2(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.NormAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceSumOfSquares(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.SumOfSquaresAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceMaxMagnitude(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.MaxMagnitudeAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceMinMagnitude(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.MinMagnitudeAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    public void ReduceLogSumExp(IGpuBuffer i, IGpuBuffer o, float mx, int sz) => GlslUnaryOp(VulkanGlslKernels.LogSumExpAxis, i, o, 1, new uint[] { 1, (uint)sz }, 2 * sizeof(uint));
    // Axis methods: dispatch os workgroups, push {outerSize=os, reduceSize=rs}
    public void VarianceAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.VarianceAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void StdAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.StdAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void ProductAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.ProductAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void NormAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.NormAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void LogSumExpAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.LogSumExpAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void CumSumAxis(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.CumSumAxis, i, o, os, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    // Scalar ops: push {scalar_as_uint_bits, size}
    public void ScalarMinusTensor(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.ScalarMinusTensor, i, o, sz, new uint[] { FloatBits(sc), (uint)sz }, sizeof(float) + sizeof(uint));
    /// <summary>In-place multiply of every element of <paramref name="buffer"/> by the device-resident
    /// scalar at <paramref name="scalar"/>[0]. Vulkan counterpart of the CUDA scale_by_device_scalar
    /// (the in-place buffer is bound as both the read input A and the write output C).</summary>
    public void ScaleByDeviceScalar(IGpuBuffer buffer, IGpuBuffer scalar, int size)
    {
        if (size <= 0) return;
        GlslBinaryOp(VulkanGlslKernels.ScaleByDeviceScalar, buffer, scalar, buffer, size, new uint[] { (uint)size }, sizeof(uint));
    }
    public void NormalizeL2(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.NormalizeL2, i, o, os, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    // Backward: dispatch os*rs workgroups, push {outerSize, reduceSize}
    public void ReduceSumBackward(IGpuBuffer go, IGpuBuffer gi, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.ReduceSumBackwardGlsl, go, gi, os * rs, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));
    public void ReduceMeanBackward(IGpuBuffer go, IGpuBuffer gi, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.ReduceMeanBackwardGlsl, go, gi, os * rs, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));

    public void ReduceMaxBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer mx, IGpuBuffer gi, int os, int rs) => GlslQuadOp(VulkanGlslKernels.ReduceMaxBackwardGlsl, go, inp, mx, gi, os * rs, 2 * sizeof(uint));

    public void ReduceVarianceBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer ms, IGpuBuffer gi, int os, int rs) => GlslQuadOp(VulkanGlslKernels.ReduceVarianceBackwardGlsl, go, inp, ms, gi, os * rs, 2 * sizeof(uint));

    public void ReduceLogVariance(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.LogVarianceAxis, i, o, os, new uint[] { (uint)os, (uint)rs }, 2 * sizeof(uint));

    public void ReduceLogVarianceBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer ms, IGpuBuffer vs, IGpuBuffer gi, int os, int rs) => GlslQuintOp(VulkanGlslKernels.ReduceLogVarianceBackwardGlsl, go, inp, ms, vs, gi, os * rs, 2 * sizeof(uint));

    #endregion

    #region Fused Broadcast / Scalar

    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastAddLast, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastSubLast, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastMulLast, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastDivLast, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastAddFirst, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastMulFirst, a, b, o, os * isz, new uint[] { (uint)os, (uint)isz }, 2 * sizeof(uint));
    public void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.AddScalar, i, o, sz, new uint[] { FloatBits(sc), (uint)sz }, sizeof(float) + sizeof(uint));
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.SubScalar, i, o, sz, new uint[] { FloatBits(sc), (uint)sz }, sizeof(float) + sizeof(uint));
    public void DivScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.DivScalar, i, o, sz, new uint[] { FloatBits(sc), (uint)sz }, sizeof(float) + sizeof(uint));
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) => GlslUnaryOp(VulkanGlslKernels.PowScalar, i, o, sz, new uint[] { FloatBits(ex), (uint)sz }, sizeof(float) + sizeof(uint));
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.FracKernel, i, o, sz, new uint[] { (uint)sz }, sizeof(uint));
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) => GlslUnaryOp(VulkanGlslKernels.ClipKernel, i, o, sz, new uint[] { FloatBits(mn), FloatBits(mx), (uint)sz }, 2 * sizeof(float) + sizeof(uint));
    public void RsqrtKernel(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.RsqrtKernel, i, o, sz, new uint[] { (uint)sz }, sizeof(uint));

    public void SinCosKernel(IGpuBuffer i, IGpuBuffer so, IGpuBuffer co, int sz)
    {
        // SinCos requires 1 input → 2 outputs; use GlslUnaryOp for each separately
        GlslUnaryOp(VulkanGlslKernels.SinKernel, i, so, sz, new uint[] { (uint)sz }, sizeof(uint));
        GlslUnaryOp(VulkanGlslKernels.CosKernel, i, co, sz, new uint[] { (uint)sz }, sizeof(uint));
    }

    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) => GlslBinaryOp(VulkanGlslKernels.EqualsKernel, a, b, o, sz, new uint[] { (uint)sz }, sizeof(uint));
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) => GlslBinaryOp(VulkanGlslKernels.NotEqualsKernel, a, b, o, sz, new uint[] { (uint)sz }, sizeof(uint));
    // IEEE classify (isnan/isinf/isfinite): a real Vulkan GLSL/SPIR-V kernel is a HW-validation
    // follow-up; the engine catches this and falls back to the correct CPU path until then.
    public void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size)
    {
        GlslDispatchN(VulkanAuditKernels.ClassifyFloat, size,
            new IGpuBuffer[] { A, C },
            new uint[] { (uint)mode, (uint)size });
    }
    public void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn)
    {
        GlslDispatchN(VulkanAuditKernels.TakeAlongDimF, outerSize*axisOut*innerSize,
            new IGpuBuffer[] { input, indices, output },
            new uint[] { (uint)outerSize, (uint)axisOut, (uint)innerSize, (uint)axisIn });
    }
    public void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize)
    {
        GlslDispatchN(VulkanAuditKernels.Cross3, outerSize*innerSize,
            new IGpuBuffer[] { a, b, output },
            new uint[] { (uint)outerSize, (uint)innerSize });
    }
    public void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size)
    {
        GlslDispatchN(VulkanAuditKernels.Ldexp, size,
            new IGpuBuffer[] { input, exponents, output },
            new uint[] { (uint)size });
    }
    public void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq)
    {
        GlslDispatchN(VulkanAuditKernels.Kron2d, (am*bp)*(an*bq),
            new IGpuBuffer[] { a, b, output },
            new uint[] { (uint)am, (uint)an, (uint)bp, (uint)bq });
    }
    public void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right)
    {
        GlslDispatchN(VulkanAuditKernels.SearchSorted, numValues,
            new IGpuBuffer[] { sortedSeq, values, output },
            new uint[] { (uint)seqLen, (uint)numValues, (uint)right });
    }
    public void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        GlslDispatchN(VulkanAuditKernels.NextAfter, size,
            new IGpuBuffer[] { a, b, output },
            new uint[] { (uint)size });
    }
    public void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis)
    {
        GlslDispatchN(VulkanAuditKernels.IndexWrite, outerSize*idxAxis*innerSize,
            new IGpuBuffer[] { output, indices, source },
            new uint[] { FloatBits(fillValue), (uint)mode, (uint)outerSize, (uint)idxAxis, (uint)innerSize, (uint)dstAxis });
    }
    public void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p)
    {
        GlslDispatchN(VulkanAuditKernels.Cdist, m*n,
            new IGpuBuffer[] { x1, x2, output },
            new uint[] { (uint)m, (uint)n, (uint)d, FloatBits(p) });
    }
    public void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p)
    {
        GlslDispatchN(VulkanAuditKernels.Pdist, n*n,
            new IGpuBuffer[] { input, output },
            new uint[] { (uint)n, (uint)d, FloatBits(p) });
    }
    public void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx)
    {
        GlslDispatchN(VulkanAuditKernels.Histc, n,
            new IGpuBuffer[] { input, hist },
            new uint[] { (uint)n, (uint)bins, FloatBits(mn), FloatBits(mx) });
    }
    public void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending)
    {
        GlslDispatchN(VulkanAuditKernels.BitonicStep, numRows*rowLen,
            new IGpuBuffer[] { values, indices },
            new uint[] { (uint)rowLen, (uint)k, (uint)j, (uint)numRows, (uint)descending });
    }
    public void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen)
    {
        GlslDispatchN(VulkanAuditKernels.CopyRows, numRows*copyLen,
            new IGpuBuffer[] { src, dst },
            new uint[] { (uint)srcRowLen, (uint)dstRowLen, (uint)numRows, (uint)copyLen });
    }
    public void IotaPad(IGpuBuffer idx, int l, int p, int numRows)
    {
        GlslDispatchN(VulkanAuditKernels.IotaPad, numRows*p,
            new IGpuBuffer[] { idx },
            new uint[] { (uint)l, (uint)p, (uint)numRows });
    }
    public void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim)
    {
        GlslDispatchN(VulkanAuditKernels.Rwkv7Forward, batch*numHeads,
            new IGpuBuffer[] { r, k, v, a, b, output, sbuf },
            new uint[] { (uint)batch, (uint)seqLen, (uint)modelDim, (uint)numHeads, (uint)headDim });
    }
    public void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses)
    {
        GlslDispatchN(VulkanAuditKernels.HsoftmaxPaths, rows*numClasses,
            new IGpuBuffer[] { acts, output },
            new uint[] { (uint)rows, (uint)treeDepth, (uint)numClasses });
    }
    public void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen)
    {
        GlslDispatchN(VulkanAuditKernels.Isin, numElements,
            new IGpuBuffer[] { elements, sortedTest, mask },
            new uint[] { (uint)numElements, (uint)testLen });
    }
    public void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff)
    {
        GlslDispatchN(VulkanAuditKernels.CopyBlock2d, blockRows*blockCols,
            new IGpuBuffer[] { block, output },
            new uint[] { (uint)blockRows, (uint)blockCols, (uint)totalCols, (uint)rowOff, (uint)colOff });
    }
    public void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size)
    {
        GlslDispatchN(VulkanAuditKernels.Zeta, size,
            new IGpuBuffer[] { x, q, output },
            new uint[] { (uint)size });
    }
    public void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size)
    {
        GlslDispatchN(VulkanAuditKernels.Polygamma, size,
            new IGpuBuffer[] { x, output },
            new uint[] { (uint)n, (uint)size });
    }
    public void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n)
    {
        GlslDispatchN(VulkanAuditKernels.ShiftedDiff, n,
            new IGpuBuffer[] { x, mask },
            new uint[] { (uint)n });
    }
    public void ReflectPad1d(IGpuBuffer input, IGpuBuffer output, int batch, int l, int lp, int pad)
    {
        GlslDispatchN(VulkanAuditKernels.ReflectPad1d, batch*lp,
            new IGpuBuffer[] { input, output },
            new uint[] { (uint)batch, (uint)l, (uint)lp, (uint)pad });
    }
    public void StftMagPhase(IGpuBuffer padded, IGpuBuffer window, IGpuBuffer mag, IGpuBuffer phase, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs)
    {
        GlslDispatchN(VulkanAuditKernels.StftMagPhase, batch*numFreqs*numFrames,
            new IGpuBuffer[] { padded, window, mag, phase },
            new uint[] { (uint)batch, (uint)lp, (uint)nFft, (uint)hop, (uint)numFrames, (uint)numFreqs });
    }
    public void PhaseVocoder(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer newMag, IGpuBuffer newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate)
    {
        GlslDispatchN(VulkanAuditKernels.PhaseVocoder, leading*nFreqV,
            new IGpuBuffer[] { mag, phase, newMag, newPhase },
            new uint[] { (uint)leading, (uint)nFramesV, (uint)nFreqV, (uint)outFrames, FloatBits(rate) });
    }
    public void BuildSpectrum(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer specRe, IGpuBuffer specIm, int batch, int numFreqs, int numFrames, int nFft)
    {
        GlslDispatchN(VulkanAuditKernels.BuildSpectrum, batch*numFrames,
            new IGpuBuffer[] { mag, phase, specRe, specIm },
            new uint[] { (uint)batch, (uint)numFreqs, (uint)numFrames, (uint)nFft });
    }
    public void IstftFromSpectrum(IGpuBuffer specRe, IGpuBuffer specIm, IGpuBuffer window, IGpuBuffer result, IGpuBuffer windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center)
    {
        GlslDispatchN(VulkanAuditKernels.IstftFromSpectrum, batch*numFrames*nFft,
            new IGpuBuffer[] { specRe, specIm, window, result, windowSum },
            new uint[] { (uint)batch, (uint)numFrames, (uint)nFft, (uint)hop, (uint)outputLength, (uint)center });
    }
    public void IstftNormalize(IGpuBuffer result, IGpuBuffer windowSum, int total)
    {
        GlslDispatchN(VulkanAuditKernels.IstftNormalize, total,
            new IGpuBuffer[] { result, windowSum },
            new uint[] { (uint)total });
    }
    public void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d)
    {
        GlslDispatchN(VulkanAuditKernels.Histogramdd, n,
            new IGpuBuffer[] { samples, hist, bins, mins, maxs },
            new uint[] { (uint)n, (uint)d });
    }
    public void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w)
    {
        GlslDispatchN(VulkanAuditKernels.MasksToBoxes, n,
            new IGpuBuffer[] { masks, output },
            new uint[] { (uint)n, (uint)h, (uint)w });
    }
    public void PairwiseIou(IGpuBuffer boxes, IGpuBuffer iou, int n)
    {
        GlslDispatchN(VulkanAuditKernels.PairwiseIou, n*n,
            new IGpuBuffer[] { boxes, iou },
            new uint[] { (uint)n });
    }
    public void LogicalOp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int mode, int n)
    {
        GlslDispatchN(VulkanAuditKernels.LogicalOp, n,
            new IGpuBuffer[] { a, b, output },
            new uint[] { (uint)mode, (uint)n });
    }
    public void LogicalNot(IGpuBuffer a, IGpuBuffer output, int n)
    {
        GlslDispatchN(VulkanAuditKernels.LogicalNot, n,
            new IGpuBuffer[] { a, output },
            new uint[] { (uint)n });
    }
    public void GridSampleBackwardInputNhwc(IGpuBuffer gradOut, IGpuBuffer grid, IGpuBuffer gradIn, int batch, int h, int w, int c, int outH, int outW)
    {
        GlslDispatchN(VulkanAuditKernels.GridsampleBackwardInput, batch*outH*outW*c,
            new IGpuBuffer[] { gradOut, grid, gradIn },
            new uint[] { (uint)batch, (uint)h, (uint)w, (uint)c, (uint)outH, (uint)outW });
    }
    public void GridSampleBackwardGridNhwc(IGpuBuffer gradOut, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer gradGrid, int batch, int h, int w, int c, int outH, int outW)
    {
        GlslDispatchN(VulkanAuditKernels.GridsampleBackwardGrid, batch*outH*outW,
            new IGpuBuffer[] { gradOut, input, grid, gradGrid },
            new uint[] { (uint)batch, (uint)h, (uint)w, (uint)c, (uint)outH, (uint)outW });
    }
    public void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode)
    {
        GlslDispatchN(VulkanAuditKernels.ScatterReduce, outerSize*srcDim*innerSize,
            new IGpuBuffer[] { output, source, index },
            new uint[] { (uint)outerSize, (uint)srcDim, (uint)dstDim, (uint)innerSize, (uint)mode });
    }
    public void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step)
    {
        GlslDispatchN(VulkanAuditKernels.Unfold, outerSize*nWindows*innerSize*size,
            new IGpuBuffer[] { src, dst },
            new uint[] { (uint)outerSize, (uint)dimSize, (uint)innerSize, (uint)nWindows, (uint)size, (uint)step });
    }

    #endregion

    #region Fused Gated Activations

    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.GluForward, i, o, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void GluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.GluBackward, go, inp, gi, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.GeGluForward, i, o, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void GeGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.GeGluBackward, go, inp, gi, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.ReGluForward, i, o, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void ReGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.ReGluBackward, go, inp, gi, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.SwiGluForward, i, o, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void SwiGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.SwiGluBackward, go, inp, gi, os * hd, new uint[] { (uint)os, (uint)hd }, 2 * sizeof(uint));
    public void ReluDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.ReluDerivative, i, o, sz, new uint[] { (uint)sz }, sizeof(uint));
    public void SigmoidDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.SigmoidDerivative, i, o, sz, new uint[] { (uint)sz }, sizeof(uint));
    public void TanhDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.TanhDerivative, i, o, sz, new uint[] { (uint)sz }, sizeof(uint));

    #endregion

    #region Fused Shape / Layout

    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int ais, int bis) => GlslBinaryOp(VulkanGlslKernels.ConcatAxisGlsl, a, b, o, os * (ais + bis), new uint[] { (uint)os, (uint)ais, (uint)bis }, 3 * sizeof(uint));
    public void SliceLastAxis(IGpuBuffer i, IGpuBuffer o, int os, int iis, int st, int ss) => GlslUnaryOp(VulkanGlslKernels.SliceLastAxisGlsl, i, o, os * ss, new uint[] { (uint)os, (uint)iis, (uint)st, (uint)ss }, 4 * sizeof(uint));
    public void SetSliceLastAxis(IGpuBuffer o, IGpuBuffer v, int os, int ois, int st, int ss) => GlslUnaryOp(VulkanGlslKernels.SetSliceLastAxisGlsl, v, o, os * ss, new uint[] { (uint)os, (uint)ois, (uint)st, (uint)ss }, 4 * sizeof(uint));
    public void SliceAxis(IGpuBuffer i, IGpuBuffer o, int os, int axSz, int stride, int idx)
    {
        if (stride == 1) { SliceLastAxis(i, o, os, axSz, idx, 1); return; }
        GlslUnaryOp(VulkanGlslKernels.SliceAxisGlsl, i, o, os * stride, new uint[] { (uint)os, (uint)axSz, (uint)stride, (uint)idx }, 4 * sizeof(uint));
    }
    public void SetSliceAxis(IGpuBuffer o, IGpuBuffer v, int os, int axSz, int stride, int idx)
    {
        if (stride == 1) { SetSliceLastAxis(o, v, os, axSz, idx, 1); return; }
        GlslUnaryOp(VulkanGlslKernels.SetSliceAxisGlsl, v, o, os * stride, new uint[] { (uint)os, (uint)axSz, (uint)stride, (uint)idx }, 4 * sizeof(uint));
    }
    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) => GlslBinaryOp(VulkanGlslKernels.Stack2Glsl, a, b, o, sz * 2, sizeof(uint));
    public void Pad2D(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int oh, int ow, int pt, int pl, float pv) => GlslUnaryOp(VulkanGlslKernels.Pad2DGlsl, i, o, ba * ch * oh * ow, 8 * sizeof(uint));
    public void Pad2DBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int oh, int ow, int pt, int pl) => GlslUnaryOp(VulkanGlslKernels.Pad2DBackwardGlsl, go, gi, ba * ch * ih * iw, 8 * sizeof(uint));
    public void TileLastAxis(IGpuBuffer i, IGpuBuffer o, int os, int isz, int rp) => GlslUnaryOp(VulkanGlslKernels.TileLastAxisGlsl, i, o, os * isz * rp, 3 * sizeof(uint));
    public void RepeatElements(IGpuBuffer i, IGpuBuffer o, int os, int isz, int rp) => GlslUnaryOp(VulkanGlslKernels.RepeatElementsGlsl, i, o, os * isz * rp, 3 * sizeof(uint));
    public void PixelShuffle(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int sc) => GlslUnaryOp(VulkanGlslKernels.PixelShuffleGlsl, i, o, ba * ch * ih * sc * iw * sc, 5 * sizeof(uint));
    public void PixelShuffleBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int sc) => GlslUnaryOp(VulkanGlslKernels.PixelShuffleBackwardGlsl, go, gi, ba * ch * sc * sc * ih * iw, 5 * sizeof(uint));
    public void Crop2D(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int oh, int ow, int ofh, int ofw) => GlslUnaryOp(VulkanGlslKernels.Crop2DGlsl, i, o, ba * ch * oh * ow, 8 * sizeof(uint));
    public void Crop2DBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int oh, int ow, int ofh, int ofw)
    {
        // Zero gradInput first — backward only writes to the cropped window, positions outside must be zero
        Scale(gi, gi, 0f, ba * ch * ih * iw);
        GlslUnaryOp(VulkanGlslKernels.Crop2DBackwardGlsl, go, gi, ba * ch * oh * ow, 8 * sizeof(uint));
    }
    public void EyeKernel(IGpuBuffer o, int n) => GlslGenerateOp(VulkanGlslKernels.EyeKernelGlsl, o, n * n, sizeof(uint));
    public void LinspaceKernel(IGpuBuffer o, float st, float sp, int sz) => GlslGenerateOp(VulkanGlslKernels.LinspaceKernelGlsl, o, sz, 2 * sizeof(float) + sizeof(uint));
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) => GlslUnaryOp(VulkanGlslKernels.OneHotKernel, idx, o, bs * nc, 2 * sizeof(uint));
    public void DiagKernel(IGpuBuffer i, IGpuBuffer o, int n) => GlslUnaryOp(VulkanGlslKernels.DiagKernelGlsl, i, o, n * n, sizeof(uint));
    public void ExtractDiagKernel(IGpuBuffer i, IGpuBuffer o, int n, int cols) => GlslUnaryOp(VulkanGlslKernels.ExtractDiagKernelGlsl, i, o, n, 2 * sizeof(uint));
    public void TriangularMask(IGpuBuffer o, int rows, int cols, int diag, float mv) => GlslGenerateOp(VulkanGlslKernels.TriangularMaskGlsl, o, rows * cols, 2 * sizeof(uint) + sizeof(int) + sizeof(float));
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) => GlslBinaryOp(VulkanGlslKernels.MaskedFillKernel, i, m, o, sz, sizeof(float) + sizeof(uint));
    public void IndexSelect(IGpuBuffer i, IGpuBuffer idx, IGpuBuffer o, int ni, int isz) => GlslBinaryOp(VulkanGlslKernels.IndexSelectGlsl, i, idx, o, ni * isz, 2 * sizeof(uint));

    #endregion

    #region Fused Loss + Noise

    public void CrossEntropyLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int bs, int nc) => GlslBinaryOp(VulkanGlslKernels.CrossEntropyLossGlsl, p, t, l, bs, 2 * sizeof(uint));
    public void MseLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int bs, int nf) => GlslBinaryOp(VulkanGlslKernels.MseLossGlsl, p, t, l, bs, 2 * sizeof(uint));
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) => GlslBinaryOp(VulkanGlslKernels.BceLossGlsl, p, t, l, sz, sizeof(uint));
    public void DropoutMask(IGpuBuffer m, int sz, float kp, ulong seed) => GlslGenerateOp(
        VulkanGlslKernels.DropoutMaskGlsl, m, sz,
        new uint[] { (uint)sz, FloatBits(kp), (uint)(seed & 0xFFFFFFFF), (uint)(seed >> 32) },
        4 * sizeof(uint));
    public void GaussianNoise(IGpuBuffer o, int sz, float mn, float sd, ulong seed) => GlslGenerateOp(
        VulkanGlslKernels.GaussianNoiseGlsl, o, sz,
        new uint[] { (uint)sz, FloatBits(mn), FloatBits(sd), (uint)(seed & 0xFFFFFFFF), (uint)(seed >> 32) },
        5 * sizeof(uint));

    #endregion

    #region Fused Softmax Variants + Distance

    public void LogSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.LogSoftmax, i, o, os, 2 * sizeof(uint));
    public void GumbelSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz, float temp, ulong seed) => GlslUnaryOp(
        VulkanGlslKernels.GumbelSoftmaxGlsl, i, o, os,
        new uint[] { (uint)os, (uint)isz, FloatBits(temp), (uint)(seed & 0xFFFFFFFF), (uint)(seed >> 32) },
        5 * sizeof(uint));
    public void Sparsemax(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.SparsemaxGlsl, i, o, os, 2 * sizeof(uint));
    public void TaylorSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.TaylorSoftmaxGlsl, i, o, os, 2 * sizeof(uint));
    public void SphericalSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.SphericalSoftmaxGlsl, i, o, os, 2 * sizeof(uint));
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) => GlslBinaryOp(VulkanGlslKernels.BatchDotProductGlsl, a, b, o, bs, 2 * sizeof(uint));
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) => GlslBinaryOp(VulkanGlslKernels.OuterProduct, a, b, o, M * N, 2 * sizeof(uint));
    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int M, int N) => GlslBinaryOp(VulkanGlslKernels.BatchOuterProductGlsl, a, b, o, bs * M * N, 3 * sizeof(uint));
    public void CosineSimilarity(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) => GlslBinaryOp(VulkanGlslKernels.CosineSimilarity, a, b, o, bs, 2 * sizeof(uint));
    public void PairwiseDistance(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N, int dim) => GlslBinaryOp(VulkanGlslKernels.PairwiseDistanceGlsl, a, b, o, M * N, 3 * sizeof(uint));
    public void PairwiseDistanceSquared(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N, int dim) => GlslBinaryOp(VulkanGlslKernels.PairwiseDistanceSquaredGlsl, a, b, o, M * N, 3 * sizeof(uint));

    #endregion

    // UploadToBuffer is defined in VulkanBackend.GpuBackend.cs (shared across partials)
}
