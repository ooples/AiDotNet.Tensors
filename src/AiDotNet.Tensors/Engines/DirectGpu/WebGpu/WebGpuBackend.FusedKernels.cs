#if NET8_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    // WebGPU fused kernel dispatch via WGSL compute pipelines.
    // Dispatch2BufferAsync(module, source, kernel, inputBuf, outputBuf, uniforms, workSize)

    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz)
    { if (sz <= 0) { ((WebGpuBuffer)o).CopyFrom(new[]{ 0f }); return; } try { Dispatch2BufferAsync("ReductionExt", WebGpuKernels.ReductionExtSource, "mean_axis", i, o, new float[]{BitConverter.Int32BitsToSingle(1),BitConverter.Int32BitsToSingle(sz),0,0}, 1).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++)s+=d[j]; ((WebGpuBuffer)o).CopyFrom(new[]{s/sz}); } }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz)
    { try { Dispatch2BufferAsync("UnaryMath", WebGpuKernels.UnaryMathSource, "clip_op", i, o, new float[]{mn,mx,BitConverter.Int32BitsToSingle(sz),0}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=Math.Min(Math.Max(d[j],mn),mx);((WebGpuBuffer)o).CopyFrom(r); } }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz)
    { try { Dispatch2BufferAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, "pow_op", i, o, new float[]{ex,0,BitConverter.Int32BitsToSingle(sz),0}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=MathF.Pow(d[j],ex);((WebGpuBuffer)o).CopyFrom(r); } }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz)
    { try { Dispatch2BufferAsync("UnaryMath", WebGpuKernels.UnaryMathSource, "frac_op", i, o, new float[]{BitConverter.Int32BitsToSingle(sz),0,0,0}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]-MathF.Floor(d[j]);((WebGpuBuffer)o).CopyFrom(r); } }
    public void EyeKernel(IGpuBuffer o, int n)
    { try { Dispatch1BufferAsync("ShapeOps", WebGpuKernels.ShapeOpsSource, "eye_kernel", o, new float[]{BitConverter.Int32BitsToSingle(n),0,0,0}, n*n).GetAwaiter().GetResult(); } catch { float[] r=new float[n*n]; for(int j=0;j<n;j++) r[j*n+j]=1f; ((WebGpuBuffer)o).CopyFrom(r); } }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc)
    { try { Dispatch2BufferAsync("ShapeOps", WebGpuKernels.ShapeOpsSource, "one_hot_kernel", idx, o, new float[]{BitConverter.Int32BitsToSingle(bs),BitConverter.Int32BitsToSingle(nc),0,0}, bs*nc).GetAwaiter().GetResult(); } catch { float[] id=DownloadBuffer(idx); float[] r=new float[bs*nc]; for(int b=0;b<bs;b++) r[b*nc+(int)id[b]]=1f; ((WebGpuBuffer)o).CopyFrom(r); } }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz)
    { try { Dispatch3BufferAsync("MaskedFill", WebGpuKernels.MaskedFillSource, "masked_fill", i, m, o, new float[]{fv,BitConverter.Int32BitsToSingle(sz)}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i); float[] md=DownloadBuffer(m); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=md[j]!=0?fv:d[j]; ((WebGpuBuffer)o).CopyFrom(r); } }
    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    { try { Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "equal_to", a, b, o, new float[]{BitConverter.Int32BitsToSingle(sz)}, sz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=ad[j]==bd[j]?1f:0f;((WebGpuBuffer)o).CopyFrom(r); } }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    { try { Dispatch3BufferAsync("Comparison", WebGpuKernels.ComparisonSource, "not_equal_to", a, b, o, new float[]{BitConverter.Int32BitsToSingle(sz)}, sz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=ad[j]!=bd[j]?1f:0f;((WebGpuBuffer)o).CopyFrom(r); } }
    // IEEE classify (isnan/isinf/isfinite): a real WGSL kernel is a HW-validation follow-up; the engine
    // catches this and falls back to the correct CPU path until then.
    public void ClassifyFloat(IGpuBuffer A, IGpuBuffer C, int mode, int size) => throw new NotSupportedException("ClassifyFloat not yet implemented on the WebGPU backend.");
    public void TakeAlongDim(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outerSize, int axisOut, int innerSize, int axisIn) => throw new NotSupportedException("TakeAlongDim not yet implemented on the WebGPU backend.");
    public void Cross3(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int outerSize, int innerSize) => throw new NotSupportedException("Cross3 not yet implemented on the WebGPU backend.");
    public void Ldexp(IGpuBuffer input, IGpuBuffer exponents, IGpuBuffer output, int size) => throw new NotSupportedException("Ldexp not yet implemented on the WebGPU backend.");
    public void Kron2D(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int am, int an, int bp, int bq) => throw new NotSupportedException("Kron2D not yet implemented on the WebGPU backend.");
    public void SearchSorted(IGpuBuffer sortedSeq, IGpuBuffer values, IGpuBuffer output, int seqLen, int numValues, int right) => throw new NotSupportedException("SearchSorted not yet implemented on the WebGPU backend.");
    public void NextAfter(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size) => throw new NotSupportedException("NextAfter not yet implemented on the WebGPU backend.");
    public void IndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis) => throw new NotSupportedException("IndexWrite not yet implemented on the WebGPU backend.");
    public void CDist(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int m, int n, int d, float p) => throw new NotSupportedException("CDist not yet implemented on the WebGPU backend.");
    public void PDist(IGpuBuffer input, IGpuBuffer output, int n, int d, float p) => throw new NotSupportedException("PDist not yet implemented on the WebGPU backend.");
    public void Histc(IGpuBuffer input, IGpuBuffer hist, int n, int bins, float mn, float mx) => throw new NotSupportedException("Histc not yet implemented on the WebGPU backend.");
    public void BitonicStep(IGpuBuffer values, IGpuBuffer indices, int rowLen, int k, int j, int numRows, int descending) => throw new NotSupportedException("BitonicStep not yet implemented on the WebGPU backend.");
    public void CopyRows(IGpuBuffer src, IGpuBuffer dst, int srcRowLen, int dstRowLen, int numRows, int copyLen) => throw new NotSupportedException("CopyRows not yet implemented on the WebGPU backend.");
    public void IotaPad(IGpuBuffer idx, int l, int p, int numRows) => throw new NotSupportedException("IotaPad not yet implemented on the WebGPU backend.");
    public void Rwkv7Forward(IGpuBuffer r, IGpuBuffer k, IGpuBuffer v, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, IGpuBuffer sbuf, int batch, int seqLen, int modelDim, int numHeads, int headDim) => throw new NotSupportedException("Rwkv7Forward not yet implemented on the WebGPU backend.");
    public void HierarchicalSoftmaxPaths(IGpuBuffer acts, IGpuBuffer output, int rows, int treeDepth, int numClasses) => throw new NotSupportedException("HierarchicalSoftmaxPaths not yet implemented on the WebGPU backend.");
    public void IsIn(IGpuBuffer elements, IGpuBuffer sortedTest, IGpuBuffer mask, int numElements, int testLen) => throw new NotSupportedException("IsIn not yet implemented on the WebGPU backend.");
    public void CopyBlock2D(IGpuBuffer block, IGpuBuffer output, int blockRows, int blockCols, int totalCols, int rowOff, int colOff) => throw new NotSupportedException("CopyBlock2D not yet implemented on the WebGPU backend.");
    public void Zeta(IGpuBuffer x, IGpuBuffer q, IGpuBuffer output, int size) => throw new NotSupportedException("Zeta not yet implemented on the WebGPU backend.");
    public void Polygamma(IGpuBuffer x, IGpuBuffer output, int n, int size) => throw new NotSupportedException("Polygamma not yet implemented on the WebGPU backend.");
    public void ShiftedDiff(IGpuBuffer x, IGpuBuffer mask, int n) => throw new NotSupportedException("ShiftedDiff not yet implemented on the WebGPU backend.");
    public void ReflectPad1d(IGpuBuffer input, IGpuBuffer output, int batch, int l, int lp, int pad) => throw new NotSupportedException("ReflectPad1d not yet implemented on the WebGPU backend.");
    public void StftMagPhase(IGpuBuffer padded, IGpuBuffer window, IGpuBuffer mag, IGpuBuffer phase, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs) => throw new NotSupportedException("StftMagPhase not yet implemented on the WebGPU backend.");
    public void PhaseVocoder(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer newMag, IGpuBuffer newPhase, int leading, int nFramesV, int nFreqV, int outFrames, float rate) => throw new NotSupportedException("PhaseVocoder not yet implemented on the WebGPU backend.");
    public void BuildSpectrum(IGpuBuffer mag, IGpuBuffer phase, IGpuBuffer specRe, IGpuBuffer specIm, int batch, int numFreqs, int numFrames, int nFft) => throw new NotSupportedException("BuildSpectrum not yet implemented on the WebGPU backend.");
    public void IstftFromSpectrum(IGpuBuffer specRe, IGpuBuffer specIm, IGpuBuffer window, IGpuBuffer result, IGpuBuffer windowSum, int batch, int numFrames, int nFft, int hop, int outputLength, int center) => throw new NotSupportedException("IstftFromSpectrum not yet implemented on the WebGPU backend.");
    public void IstftNormalize(IGpuBuffer result, IGpuBuffer windowSum, int total) => throw new NotSupportedException("IstftNormalize not yet implemented on the WebGPU backend.");
    public void HistogramDD(IGpuBuffer samples, IGpuBuffer hist, IGpuBuffer bins, IGpuBuffer mins, IGpuBuffer maxs, int n, int d) => throw new NotSupportedException("HistogramDD not yet implemented on the WebGPU backend.");
    public void MasksToBoxes(IGpuBuffer masks, IGpuBuffer output, int n, int h, int w) => throw new NotSupportedException("MasksToBoxes not yet implemented on the WebGPU backend.");
    public void PairwiseIou(IGpuBuffer boxes, IGpuBuffer iou, int n) => throw new NotSupportedException("PairwiseIou not yet implemented on the WebGPU backend.");
    public void LogicalOp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int mode, int n) => throw new NotSupportedException("LogicalOp not yet implemented on the WebGPU backend.");
    public void LogicalNot(IGpuBuffer a, IGpuBuffer output, int n) => throw new NotSupportedException("LogicalNot not yet implemented on the WebGPU backend.");
    public void GridSampleBackwardInputNhwc(IGpuBuffer gradOut, IGpuBuffer grid, IGpuBuffer gradIn, int batch, int h, int w, int c, int outH, int outW) => throw new NotSupportedException("GridSampleBackwardInputNhwc not yet implemented on the WebGPU backend.");
    public void GridSampleBackwardGridNhwc(IGpuBuffer gradOut, IGpuBuffer input, IGpuBuffer grid, IGpuBuffer gradGrid, int batch, int h, int w, int c, int outH, int outW) => throw new NotSupportedException("GridSampleBackwardGridNhwc not yet implemented on the WebGPU backend.");
    public void ScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outerSize, int srcDim, int dstDim, int innerSize, int mode) => throw new NotSupportedException("ScatterReduce not yet implemented on the WebGPU backend.");
    public void Unfold(IGpuBuffer src, IGpuBuffer dst, int outerSize, int dimSize, int innerSize, int nWindows, int size, int step) => throw new NotSupportedException("Unfold not yet implemented on the WebGPU backend.");
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N)
    { try { Dispatch3BufferAsync("OuterProduct", WebGpuKernels.OuterProductSource, "outer_product", a, b, o, new float[]{BitConverter.Int32BitsToSingle(M),BitConverter.Int32BitsToSingle(N)}, M*N).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[M*N]; for(int i2=0;i2<M;i2++) for(int j=0;j<N;j++) r[i2*N+j]=ad[i2]*bd[j]; ((WebGpuBuffer)o).CopyFrom(r); } }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim)
    { try { Dispatch3BufferAsync("BatchDotProduct", WebGpuKernels.BatchDotProductSource, "batch_dot_product", a, b, o, new float[]{BitConverter.Int32BitsToSingle(bs),BitConverter.Int32BitsToSingle(dim)}, bs).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[bs]; for(int i2=0;i2<bs;i2++){float s=0;for(int j=0;j<dim;j++)s+=ad[i2*dim+j]*bd[i2*dim+j];r[i2]=s;} ((WebGpuBuffer)o).CopyFrom(r); } }
    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
    { try { Dispatch2BufferAsync("GatedActivation", WebGpuKernels.GatedActivationSource, "glu_forward", i, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(hd),0,0}, os*hd).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int k=0;k<hd;k++){float v=d[j*fd+k];float g=d[j*fd+hd+k];r[j*hd+k]=v*(1f/(1f+MathF.Exp(-g)));}((WebGpuBuffer)o).CopyFrom(r); } }
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
    { try { Dispatch2BufferAsync("GatedActivation", WebGpuKernels.GatedActivationSource, "geglu_forward", i, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(hd),0,0}, os*hd).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int k=0;k<hd;k++){float v=d[j*fd+k];float g=d[j*fd+hd+k];float x3=v*v*v;r[j*hd+k]=0.5f*v*(1f+MathF.Tanh(0.7978845608f*(v+0.044715f*x3)))*g;}((WebGpuBuffer)o).CopyFrom(r); } }
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
    { try { Dispatch2BufferAsync("GatedActivation", WebGpuKernels.GatedActivationSource, "reglu_forward", i, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(hd),0,0}, os*hd).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int k=0;k<hd;k++)r[j*hd+k]=Math.Max(d[j*fd+k],0f)*d[j*fd+hd+k];((WebGpuBuffer)o).CopyFrom(r); } }
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd)
    { try { Dispatch2BufferAsync("GatedActivation", WebGpuKernels.GatedActivationSource, "swiglu_forward", i, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(hd),0,0}, os*hd).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[os*hd];int fd=hd*2;for(int j=0;j<os;j++)for(int k=0;k<hd;k++){float v=d[j*fd+k];float sig=1f/(1f+MathF.Exp(-v));r[j*hd+k]=v*sig*d[j*fd+hd+k];}((WebGpuBuffer)o).CopyFrom(r); } }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz)
    { try { Dispatch3BufferAsync("Loss", WebGpuKernels.LossSource, "bce_op", p, t, l, new float[]{BitConverter.Int32BitsToSingle(sz),0,0,0}, sz).GetAwaiter().GetResult(); } catch { float[] pd=DownloadBuffer(p);float[] td=DownloadBuffer(t);float[] r=new float[sz];for(int j=0;j<sz;j++){float pc=Math.Min(Math.Max(pd[j],1e-7f),1f-1e-7f);r[j]=-(td[j]*MathF.Log(pc)+(1f-td[j])*MathF.Log(1f-pc));}((WebGpuBuffer)l).CopyFrom(r); } }
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz)
    { try { Dispatch2BufferAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, "sub_op", i, o, new float[]{sc,0,BitConverter.Int32BitsToSingle(sz),0}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]-sc;((WebGpuBuffer)o).CopyFrom(r); } }
    public void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz)
    { try { Dispatch2BufferAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, "add_op", i, o, new float[]{sc,0,BitConverter.Int32BitsToSingle(sz),0}, sz).GetAwaiter().GetResult(); } catch { float[] d=DownloadBuffer(i);float[] r=new float[sz];for(int j=0;j<sz;j++)r[j]=d[j]+sc;((WebGpuBuffer)o).CopyFrom(r); } }
    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
    { try { Dispatch3BufferAsync("BroadcastExt", WebGpuKernels.BroadcastExtSource, "broadcast_add_last", a, b, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(isz)}, os*isz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]+bd[j%isz];((WebGpuBuffer)o).CopyFrom(r); } }
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
    { try { Dispatch3BufferAsync("BroadcastExt", WebGpuKernels.BroadcastExtSource, "broadcast_sub_last", a, b, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(isz)}, os*isz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]-bd[j%isz];((WebGpuBuffer)o).CopyFrom(r); } }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
    { try { Dispatch3BufferAsync("BroadcastExt", WebGpuKernels.BroadcastExtSource, "broadcast_mul_last", a, b, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(isz)}, os*isz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]*bd[j%isz];((WebGpuBuffer)o).CopyFrom(r); } }
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz)
    { try { Dispatch3BufferAsync("BroadcastExt", WebGpuKernels.BroadcastExtSource, "broadcast_div_last", a, b, o, new float[]{BitConverter.Int32BitsToSingle(os),BitConverter.Int32BitsToSingle(isz)}, os*isz).GetAwaiter().GetResult(); } catch { float[] ad=DownloadBuffer(a);float[] bd=DownloadBuffer(b);float[] r=new float[os*isz];for(int j=0;j<os*isz;j++)r[j]=ad[j]/(bd[j%isz]+1e-12f);((WebGpuBuffer)o).CopyFrom(r); } }
}
#endif
