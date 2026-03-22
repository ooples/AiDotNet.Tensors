// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend fused kernel implementations for VulkanBackend.
// Dispatches to runtime-compiled GLSL compute shaders via VulkanGlslCompiler (libshaderc).

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed partial class VulkanBackend
{
    #region Fused Reductions

    public void ReduceMean(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++) s+=d[j]; float[] r={s/sz}; UploadToBuffer(r,o); }
    public void ReduceProduct(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float p=1; for(int j=0;j<sz;j++) p*=d[j]; float[] r={p}; UploadToBuffer(r,o); }
    public void ReduceNormL2(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++) s+=d[j]*d[j]; float[] r={s}; UploadToBuffer(r,o); }
    public void ReduceSumOfSquares(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++) s+=d[j]*d[j]; float[] r={s}; UploadToBuffer(r,o); }
    public void ReduceMaxMagnitude(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float m=0; for(int j=0;j<sz;j++) m=Math.Max(m,Math.Abs(d[j])); float[] r={m}; UploadToBuffer(r,o); }
    public void ReduceMinMagnitude(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float m=float.MaxValue; for(int j=0;j<sz;j++) m=Math.Min(m,Math.Abs(d[j])); float[] r={m}; UploadToBuffer(r,o); }
    public void ReduceLogSumExp(IGpuBuffer i, IGpuBuffer o, float mx, int sz) { float[] d=DownloadBuffer(i); float s=0; for(int j=0;j<sz;j++) s+=MathF.Exp(d[j]-mx); float[] r={s}; UploadToBuffer(r,o); }
    public void VarianceAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.VarianceAxis, i, o, os, 2 * sizeof(uint));
    public void StdAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.StdAxis, i, o, os, 2 * sizeof(uint));
    public void ProductAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.ProductAxis, i, o, os, 2 * sizeof(uint));
    public void NormAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.NormAxis, i, o, os, 2 * sizeof(uint));
    public void LogSumExpAxis(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.LogSumExpAxis, i, o, os, 2 * sizeof(uint));
    public void CumSumAxis(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.CumSumAxis, i, o, os, 2 * sizeof(uint));
    public void ScalarMinusTensor(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.ScalarMinusTensor, i, o, sz, sizeof(float) + sizeof(uint));
    public void NormalizeL2(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.NormalizeL2, i, o, os, 2 * sizeof(uint));
    public void ReduceSumBackward(IGpuBuffer go, IGpuBuffer gi, int os, int rs) { float[] goD=DownloadBuffer(go); float[] giD=new float[os*rs]; for(int j=0;j<os*rs;j++) giD[j]=goD[j/rs]; UploadToBuffer(giD,gi); }
    public void ReduceMeanBackward(IGpuBuffer go, IGpuBuffer gi, int os, int rs) { float[] goD=DownloadBuffer(go); float[] giD=new float[os*rs]; for(int j=0;j<os*rs;j++) giD[j]=goD[j/rs]/rs; UploadToBuffer(giD,gi); }

    public void ReduceMaxBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer mx, IGpuBuffer gi, int os, int rs)
    {
        // Download, compute on CPU, upload — complex multi-buffer reduction backward
        float[] goD = DownloadBuffer(go), inD = DownloadBuffer(inp), mxD = DownloadBuffer(mx);
        float[] giD = new float[os * rs];
        for (int i = 0; i < os * rs; i++) giD[i] = inD[i] == mxD[i / rs] ? goD[i / rs] : 0;
        UploadToBuffer(giD, gi);
    }

    public void ReduceVarianceBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer ms, IGpuBuffer gi, int os, int rs)
    {
        float[] goD = DownloadBuffer(go), inD = DownloadBuffer(inp), mD = DownloadBuffer(ms);
        float[] giD = new float[os * rs];
        for (int i = 0; i < os * rs; i++) giD[i] = goD[i / rs] * 2f * (inD[i] - mD[i / rs]) / rs;
        UploadToBuffer(giD, gi);
    }

    public void ReduceLogVariance(IGpuBuffer i, IGpuBuffer o, int os, int rs) => GlslUnaryOp(VulkanGlslKernels.VarianceAxis, i, o, os, 2 * sizeof(uint));

    public void ReduceLogVarianceBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer ms, IGpuBuffer vs, IGpuBuffer gi, int os, int rs)
    {
        float[] goD = DownloadBuffer(go), inD = DownloadBuffer(inp), mD = DownloadBuffer(ms), vD = DownloadBuffer(vs);
        float[] giD = new float[os * rs];
        for (int i = 0; i < os * rs; i++) { int o2 = i / rs; giD[i] = goD[o2] * (1f / (vD[o2] + 1e-8f)) * 2f * (inD[i] - mD[o2]) / rs; }
        UploadToBuffer(giD, gi);
    }

    #endregion

    #region Fused Broadcast / Scalar

    public void BroadcastAddLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastAddLast, a, b, o, os * isz, 2 * sizeof(uint));
    public void BroadcastSubLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]-bd[j%isz]; UploadToBuffer(r,o); }
    public void BroadcastMulLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) => GlslBinaryOp(VulkanGlslKernels.BroadcastMulLast, a, b, o, os * isz, 2 * sizeof(uint));
    public void BroadcastDivLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j]/(bd[j%isz]+1e-12f); UploadToBuffer(r,o); }
    public void BroadcastAddFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j/isz]+bd[j]; UploadToBuffer(r,o); }
    public void BroadcastMulFirst(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int isz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[os*isz]; for(int j=0;j<os*isz;j++) r[j]=ad[j/isz]*bd[j]; UploadToBuffer(r,o); }
    public void AddScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) => GlslUnaryOp(VulkanGlslKernels.AddScalar, i, o, sz, sizeof(float) + sizeof(uint));
    public void SubScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]-sc; UploadToBuffer(r,o); }
    public void DivScalar(IGpuBuffer i, IGpuBuffer o, float sc, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]/sc; UploadToBuffer(r,o); }
    public void PowScalar(IGpuBuffer i, IGpuBuffer o, float ex, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=MathF.Pow(d[j],ex); UploadToBuffer(r,o); }
    public void FracKernel(IGpuBuffer i, IGpuBuffer o, int sz) { float[] d=DownloadBuffer(i); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=d[j]-MathF.Floor(d[j]); UploadToBuffer(r,o); }
    public void ClipKernel(IGpuBuffer i, IGpuBuffer o, float mn, float mx, int sz) => GlslUnaryOp(VulkanGlslKernels.ClipKernel, i, o, sz, 2 * sizeof(float) + sizeof(uint));
    public void RsqrtKernel(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.RsqrtKernel, i, o, sz);

    public void SinCosKernel(IGpuBuffer i, IGpuBuffer so, IGpuBuffer co, int sz)
    {
        float[] d = DownloadBuffer(i); float[] s = new float[sz]; float[] c = new float[sz];
        for (int j = 0; j < sz; j++) { s[j] = MathF.Sin(d[j]); c[j] = MathF.Cos(d[j]); }
        UploadToBuffer(s, so); UploadToBuffer(c, co);
    }

    public void EqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=ad[j]==bd[j]?1f:0f; UploadToBuffer(r,o); }
    public void NotEqualsKernel(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz) { float[] ad=DownloadBuffer(a); float[] bd=DownloadBuffer(b); float[] r=new float[sz]; for(int j=0;j<sz;j++) r[j]=ad[j]!=bd[j]?1f:0f; UploadToBuffer(r,o); }

    #endregion

    #region Fused Gated Activations

    public void GluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.GluForward, i, o, os * hd, 2 * sizeof(uint));
    public void GluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.GluForward, go, inp, gi, os * hd, 2 * sizeof(uint));
    public void GeGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.GeGluForward, i, o, os * hd, 2 * sizeof(uint));
    public void GeGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.GeGluForward, go, inp, gi, os * hd, 2 * sizeof(uint));
    public void ReGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.ReGluForward, i, o, os * hd, 2 * sizeof(uint));
    public void ReGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.ReGluForward, go, inp, gi, os * hd, 2 * sizeof(uint));
    public void SwiGluForward(IGpuBuffer i, IGpuBuffer o, int os, int hd) => GlslUnaryOp(VulkanGlslKernels.SwiGluForward, i, o, os * hd, 2 * sizeof(uint));
    public void SwiGluBackward(IGpuBuffer go, IGpuBuffer inp, IGpuBuffer gi, int os, int hd) => GlslBinaryOp(VulkanGlslKernels.SwiGluForward, go, inp, gi, os * hd, 2 * sizeof(uint));
    public void ReluDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.ReluDerivative, i, o, sz);
    public void SigmoidDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.SigmoidDerivative, i, o, sz);
    public void TanhDerivative(IGpuBuffer i, IGpuBuffer o, int sz) => GlslUnaryOp(VulkanGlslKernels.TanhDerivative, i, o, sz);

    #endregion

    #region Fused Shape / Layout

    public void ConcatAxis(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int os, int ais, int bis)
    {
        float[] ad = DownloadBuffer(a), bd = DownloadBuffer(b);
        int ti = ais + bis; float[] r = new float[os * ti];
        for (int i = 0; i < os; i++) { Array.Copy(ad, i * ais, r, i * ti, ais); Array.Copy(bd, i * bis, r, i * ti + ais, bis); }
        UploadToBuffer(r, o);
    }

    public void SliceLastAxis(IGpuBuffer i, IGpuBuffer o, int os, int iis, int st, int ss)
    {
        float[] d = DownloadBuffer(i); float[] r = new float[os * ss];
        for (int j = 0; j < os; j++) Array.Copy(d, j * iis + st, r, j * ss, ss);
        UploadToBuffer(r, o);
    }

    public void SetSliceLastAxis(IGpuBuffer o, IGpuBuffer v, int os, int ois, int st, int ss)
    {
        float[] od = DownloadBuffer(o), vd = DownloadBuffer(v);
        for (int j = 0; j < os; j++) Array.Copy(vd, j * ss, od, j * ois + st, ss);
        UploadToBuffer(od, o);
    }

    public void Stack2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int sz)
    {
        float[] ad = DownloadBuffer(a), bd = DownloadBuffer(b); float[] r = new float[sz * 2];
        for (int i = 0; i < sz; i++) { r[2 * i] = ad[i]; r[2 * i + 1] = bd[i]; }
        UploadToBuffer(r, o);
    }

    public void Pad2D(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int oh, int ow, int pt, int pl, float pv)
    {
        float[] d = DownloadBuffer(i); float[] r = new float[ba * ch * oh * ow]; for (int fi = 0; fi < r.Length; fi++) r[fi] = pv;
        for (int b2 = 0; b2 < ba; b2++) for (int c = 0; c < ch; c++) for (int h = 0; h < ih; h++)
            Array.Copy(d, ((b2 * ch + c) * ih + h) * iw, r, ((b2 * ch + c) * oh + h + pt) * ow + pl, iw);
        UploadToBuffer(r, o);
    }

    public void Pad2DBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int oh, int ow, int pt, int pl)
    {
        float[] god = DownloadBuffer(go); float[] gid = new float[ba * ch * ih * iw];
        for (int b2 = 0; b2 < ba; b2++) for (int c = 0; c < ch; c++) for (int h = 0; h < ih; h++)
            Array.Copy(god, ((b2 * ch + c) * oh + h + pt) * ow + pl, gid, ((b2 * ch + c) * ih + h) * iw, iw);
        UploadToBuffer(gid, gi);
    }

    public void TileLastAxis(IGpuBuffer i, IGpuBuffer o, int os, int isz, int rp)
    {
        float[] d = DownloadBuffer(i); float[] r = new float[os * isz * rp];
        for (int j = 0; j < os; j++) for (int rep = 0; rep < rp; rep++) Array.Copy(d, j * isz, r, j * isz * rp + rep * isz, isz);
        UploadToBuffer(r, o);
    }

    public void RepeatElements(IGpuBuffer i, IGpuBuffer o, int os, int isz, int rp)
    {
        float[] d = DownloadBuffer(i); float[] r = new float[os * isz * rp];
        for (int j = 0; j < os * isz; j++) for (int rep = 0; rep < rp; rep++) r[j * rp + rep] = d[j];
        UploadToBuffer(r, o);
    }

    public void PixelShuffle(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int sc)
    {
        float[] d = DownloadBuffer(i); int oh2 = ih * sc, ow2 = iw * sc; float[] r = new float[ba * ch * oh2 * ow2];
        for (int b2 = 0; b2 < ba; b2++) for (int oc = 0; oc < ch; oc++) for (int ohh = 0; ohh < oh2; ohh++) for (int oww = 0; oww < ow2; oww++) {
            int srcC = oc * sc * sc + (ohh % sc) * sc + (oww % sc);
            r[((b2 * ch + oc) * oh2 + ohh) * ow2 + oww] = d[((b2 * ch * sc * sc + srcC) * ih + ohh / sc) * iw + oww / sc];
        }
        UploadToBuffer(r, o);
    }

    public void PixelShuffleBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int sc)
    {
        float[] god = DownloadBuffer(go); int oh2 = ih * sc, ow2 = iw * sc;
        float[] gid = new float[ba * ch * sc * sc * ih * iw];
        for (int b2 = 0; b2 < ba; b2++) for (int ic = 0; ic < ch * sc * sc; ic++) {
            int oc = ic / (sc * sc); int sub = ic % (sc * sc); int subH = sub / sc, subW = sub % sc;
            for (int ihh = 0; ihh < ih; ihh++) for (int iww = 0; iww < iw; iww++)
                gid[((b2 * ch * sc * sc + ic) * ih + ihh) * iw + iww] = god[((b2 * ch + oc) * oh2 + ihh * sc + subH) * ow2 + iww * sc + subW];
        }
        UploadToBuffer(gid, gi);
    }

    public void Crop2D(IGpuBuffer i, IGpuBuffer o, int ba, int ch, int ih, int iw, int oh, int ow, int ofh, int ofw)
    {
        float[] d = DownloadBuffer(i); float[] r = new float[ba * ch * oh * ow];
        for (int b2 = 0; b2 < ba; b2++) for (int c = 0; c < ch; c++) for (int h = 0; h < oh; h++)
            Array.Copy(d, ((b2 * ch + c) * ih + h + ofh) * iw + ofw, r, ((b2 * ch + c) * oh + h) * ow, ow);
        UploadToBuffer(r, o);
    }

    public void Crop2DBackward(IGpuBuffer go, IGpuBuffer gi, int ba, int ch, int ih, int iw, int oh, int ow, int ofh, int ofw)
    {
        float[] god = DownloadBuffer(go); float[] gid = new float[ba * ch * ih * iw];
        for (int b2 = 0; b2 < ba; b2++) for (int c = 0; c < ch; c++) for (int h = 0; h < oh; h++)
            Array.Copy(god, ((b2 * ch + c) * oh + h) * ow, gid, ((b2 * ch + c) * ih + h + ofh) * iw + ofw, ow);
        UploadToBuffer(gid, gi);
    }

    public void EyeKernel(IGpuBuffer o, int n) { float[] r = new float[n * n]; for (int i = 0; i < n; i++) r[i * n + i] = 1f; UploadToBuffer(r, o); }
    public void LinspaceKernel(IGpuBuffer o, float st, float sp, int sz) { float[] r = new float[sz]; for (int i = 0; i < sz; i++) r[i] = st + sp * i; UploadToBuffer(r, o); }
    public void OneHotKernel(IGpuBuffer idx, IGpuBuffer o, int bs, int nc) { float[] id = DownloadBuffer(idx); float[] r = new float[bs * nc]; for (int b = 0; b < bs; b++) r[b * nc + (int)id[b]] = 1f; UploadToBuffer(r, o); }
    public void DiagKernel(IGpuBuffer i, IGpuBuffer o, int n) { float[] d = DownloadBuffer(i); float[] r = new float[n * n]; for (int j = 0; j < n; j++) r[j * n + j] = d[j]; UploadToBuffer(r, o); }
    public void ExtractDiagKernel(IGpuBuffer i, IGpuBuffer o, int n, int cols) { float[] d = DownloadBuffer(i); float[] r = new float[n]; for (int j = 0; j < n; j++) r[j] = d[j * cols + j]; UploadToBuffer(r, o); }
    public void TriangularMask(IGpuBuffer o, int rows, int cols, int diag, float mv) { float[] r = new float[rows * cols]; for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) if (j > i + diag) r[i * cols + j] = mv; UploadToBuffer(r, o); }
    public void MaskedFillKernel(IGpuBuffer i, IGpuBuffer m, IGpuBuffer o, float fv, int sz) { float[] d = DownloadBuffer(i); float[] md = DownloadBuffer(m); float[] r = new float[sz]; for (int j = 0; j < sz; j++) r[j] = md[j] != 0 ? fv : d[j]; UploadToBuffer(r, o); }
    public void IndexSelect(IGpuBuffer i, IGpuBuffer idx, IGpuBuffer o, int ni, int isz) { float[] d = DownloadBuffer(i); float[] id = DownloadBuffer(idx); float[] r = new float[ni * isz]; for (int j = 0; j < ni; j++) Array.Copy(d, (int)id[j] * isz, r, j * isz, isz); UploadToBuffer(r, o); }

    #endregion

    #region Fused Loss + Noise

    public void CrossEntropyLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int bs, int nc) { float[] pd = DownloadBuffer(p); float[] td = DownloadBuffer(t); float[] ld = new float[bs]; for (int b = 0; b < bs; b++) { float s = 0; for (int c = 0; c < nc; c++) if (td[b * nc + c] > 0) s -= td[b * nc + c] * MathF.Log(Math.Max(pd[b * nc + c], 1e-7f)); ld[b] = s; } UploadToBuffer(ld, l); }
    public void MseLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int bs, int nf) { float[] pd = DownloadBuffer(p); float[] td = DownloadBuffer(t); float[] ld = new float[bs]; for (int b = 0; b < bs; b++) { float s = 0; for (int f = 0; f < nf; f++) { float d = pd[b * nf + f] - td[b * nf + f]; s += d * d; } ld[b] = s / nf; } UploadToBuffer(ld, l); }
    public void BceLoss(IGpuBuffer p, IGpuBuffer t, IGpuBuffer l, int sz) { float[] pd = DownloadBuffer(p); float[] td = DownloadBuffer(t); float[] r = new float[sz]; for (int i = 0; i < sz; i++) { float pc = Math.Min(Math.Max(pd[i], 1e-7f), 1f - 1e-7f); r[i] = -(td[i] * MathF.Log(pc) + (1f - td[i]) * MathF.Log(1f - pc)); } UploadToBuffer(r, l); }
    public void DropoutMask(IGpuBuffer m, int sz, float kp, ulong seed) { var rng = new Random((int)(seed & 0x7FFFFFFF)); float[] r = new float[sz]; float inv = 1f / kp; for (int i = 0; i < sz; i++) r[i] = rng.NextDouble() < kp ? inv : 0f; UploadToBuffer(r, m); }
    public void GaussianNoise(IGpuBuffer o, int sz, float mn, float sd, ulong seed) { var rng = new Random((int)(seed & 0x7FFFFFFF)); float[] r = new float[sz]; for (int i = 0; i < sz; i++) { double u1 = 1.0 - rng.NextDouble(); double u2 = rng.NextDouble(); r[i] = mn + sd * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)); } UploadToBuffer(r, o); }

    #endregion

    #region Fused Softmax Variants + Distance

    public void LogSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) => GlslUnaryOp(VulkanGlslKernels.LogSoftmax, i, o, os, 2 * sizeof(uint));
    public void GumbelSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz, float temp, ulong seed) { float[] d = DownloadBuffer(i); var rng = new Random((int)(seed & 0x7FFFFFFF)); float[] r = new float[os * isz]; for (int j = 0; j < os; j++) { int b = j * isz; float mx = float.MinValue; for (int k = 0; k < isz; k++) { float u = (float)(1.0 - rng.NextDouble()); r[b + k] = (d[b + k] + (-MathF.Log(-MathF.Log(u)))) / temp; mx = Math.Max(mx, r[b + k]); } float se = 0; for (int k = 0; k < isz; k++) { r[b + k] = MathF.Exp(r[b + k] - mx); se += r[b + k]; } for (int k = 0; k < isz; k++) r[b + k] /= se + 1e-10f; } UploadToBuffer(r, o); }
    public void Sparsemax(IGpuBuffer i, IGpuBuffer o, int os, int isz) { float[] d = DownloadBuffer(i); float[] r = new float[os * isz]; for (int j = 0; j < os; j++) { int b = j * isz; float[] sorted = new float[isz]; Array.Copy(d, b, sorted, 0, isz); Array.Sort(sorted); Array.Reverse(sorted); float cs = 0, tau = 0; for (int k = 0; k < isz; k++) { cs += sorted[k]; float cand = (cs - 1f) / (k + 1); if (sorted[k] > cand) tau = cand; } for (int k = 0; k < isz; k++) r[b + k] = Math.Max(d[b + k] - tau, 0f); } UploadToBuffer(r, o); }
    public void TaylorSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) { float[] d = DownloadBuffer(i); float[] r = new float[os * isz]; for (int j = 0; j < os; j++) { int b = j * isz; float s = 0; for (int k = 0; k < isz; k++) { float x = d[b + k]; float t = Math.Max(1f + x + 0.5f * x * x, 1e-10f); r[b + k] = t; s += t; } for (int k = 0; k < isz; k++) r[b + k] /= s; } UploadToBuffer(r, o); }
    public void SphericalSoftmax(IGpuBuffer i, IGpuBuffer o, int os, int isz) { float[] d = DownloadBuffer(i); float[] r = new float[os * isz]; for (int j = 0; j < os; j++) { int b = j * isz; float mx = float.MinValue; for (int k = 0; k < isz; k++) { r[b + k] = d[b + k]; mx = Math.Max(mx, r[b + k]); } float se = 0; for (int k = 0; k < isz; k++) { r[b + k] = MathF.Exp(r[b + k] - mx); se += r[b + k]; } for (int k = 0; k < isz; k++) r[b + k] /= se + 1e-10f; } UploadToBuffer(r, o); }
    public void BatchDotProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) { float[] ad = DownloadBuffer(a); float[] bd = DownloadBuffer(b); float[] r = new float[bs]; for (int i = 0; i < bs; i++) { float s = 0; for (int j = 0; j < dim; j++) s += ad[i * dim + j] * bd[i * dim + j]; r[i] = s; } UploadToBuffer(r, o); }
    public void OuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N) => GlslBinaryOp(VulkanGlslKernels.OuterProduct, a, b, o, M * N, 2 * sizeof(uint));
    public void BatchOuterProduct(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int M, int N) { float[] ad = DownloadBuffer(a); float[] bd = DownloadBuffer(b); float[] r = new float[bs * M * N]; for (int ba = 0; ba < bs; ba++) for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) r[ba * M * N + i * N + j] = ad[ba * M + i] * bd[ba * N + j]; UploadToBuffer(r, o); }
    public void CosineSimilarity(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int bs, int dim) => GlslBinaryOp(VulkanGlslKernels.CosineSimilarity, a, b, o, bs, 2 * sizeof(uint));
    public void PairwiseDistance(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N, int dim) { float[] ad = DownloadBuffer(a); float[] bd = DownloadBuffer(b); float[] r = new float[M * N]; for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float s = 0; for (int k = 0; k < dim; k++) { float d = ad[i * dim + k] - bd[j * dim + k]; s += d * d; } r[i * N + j] = MathF.Sqrt(s); } UploadToBuffer(r, o); }
    public void PairwiseDistanceSquared(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int M, int N, int dim) { float[] ad = DownloadBuffer(a); float[] bd = DownloadBuffer(b); float[] r = new float[M * N]; for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float s = 0; for (int k = 0; k < dim; k++) { float d = ad[i * dim + k] - bd[j * dim + k]; s += d * d; } r[i * N + j] = s; } UploadToBuffer(r, o); }

    #endregion

    // UploadToBuffer is defined in VulkanBackend.GpuBackend.cs (shared across partials)
}
