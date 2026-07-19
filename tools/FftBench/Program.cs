using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Fft = AiDotNet.Tensors.LinearAlgebra.Fft.Fft;

// Minimal repro of the HRE dense-path AccessViolation: CPU engine, tape-recorded
// [rows,64]@[64,64] matmul + backward, the exact op the dense SpectralFilter uses.
if (args.Length > 0 && args[0] == "matmul")
{
    AiDotNetEngine.Current = new CpuEngine();
    var mEng = AiDotNetEngine.Current;
    int mrows = args.Length > 1 ? int.Parse(args[1]) : 12288;
    int W = 64;
    var rngM = new Random(1);
    var xd = new float[mrows * W]; var wd = new float[W * W];
    for (int i = 0; i < xd.Length; i++) xd[i] = (float)(rngM.NextDouble() * 2 - 1);
    for (int i = 0; i < wd.Length; i++) wd[i] = (float)(rngM.NextDouble() * 2 - 1);
    Console.WriteLine($"matmul-tape repro: mrows={mrows} W={W}, engine={mEng.GetType().Name}");
    for (int it = 0; it < 30; it++)
    {
        var x = new Tensor<float>((float[])xd.Clone(), new[] { mrows, W });
        var w = new Tensor<float>((float[])wd.Clone(), new[] { W, W });
        using var tape = new GradientTape<float>();
        var y = mEng.TensorMatMul<float>(x, w);
        var loss = mEng.ReduceSum<float>(mEng.TensorMultiply<float>(y, y), new[] { 0, 1 }, false);
        var grads = tape.ComputeGradients(loss, new[] { x, w });
        var gx = grads[x].GetCpuData(); var gw = grads[w].GetCpuData();
        if (it == 0) Console.WriteLine($"  iter0 ok: gx[0]={gx[0]:F3} gw[0]={gw[0]:F3}");
    }
    Console.WriteLine("matmul-tape repro: NO CRASH after 30 iters");
    return;
}

// Repro of the HRE standard-block AccessViolation: the trainable-MHA ops the spectral
// path never touches — TensorPermute(3D) + BatchMatMul(3D) + Softmax(axis=2) + their
// backward. Bisect with STDATTN_STOP=proj|perm|scores|softmax|ctx|out to find the faulting op.
if (args.Length > 0 && args[0] == "stdattn")
{
    AiDotNetEngine.Current = new CpuEngine();
    var e = AiDotNetEngine.Current;
    int B = 256, S = 6, W = 64;
    string stop = Environment.GetEnvironmentVariable("STDATTN_STOP") ?? "out";
    var arng = new Random(1);
    float[] Rnd(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(arng.NextDouble() * 2 - 1); return a; }
    var hd = Rnd(B * S * W); var wqd = Rnd(W * W); var wkd = Rnd(W * W); var wvd = Rnd(W * W); var wod = Rnd(W * W);
    Console.WriteLine($"stdattn-tape repro: B={B} S={S} E={W} engine={e.GetType().Name} STOP={stop}");
    for (int it = 0; it < 20; it++)
    {
        var h = new Tensor<float>((float[])hd.Clone(), new[] { B, S, W });
        var wq = new Tensor<float>((float[])wqd.Clone(), new[] { W, W });
        var wk = new Tensor<float>((float[])wkd.Clone(), new[] { W, W });
        var wv = new Tensor<float>((float[])wvd.Clone(), new[] { W, W });
        var wo = new Tensor<float>((float[])wod.Clone(), new[] { W, W });
        using var tape = new GradientTape<float>();
        var h2 = e.Reshape<float>(h, new[] { B * S, W });
        Tensor<float> Proj(Tensor<float> w) => e.Reshape<float>(e.TensorMatMul<float>(h2, w), new[] { B, S, W });
        var q = Proj(wq); var k = Proj(wk); var v = Proj(wv);
        Tensor<float> outT;
        if (stop == "proj") outT = e.Reshape<float>(q, new[] { B, S, W });
        else
        {
            var kT = e.TensorPermute<float>(k, new[] { 0, 2, 1 });          // [B,E,S]
            if (stop == "perm") outT = e.Reshape<float>(e.BatchMatMul<float>(q, kT), new[] { B, S, S });
            else
            {
                var scores = e.TensorMultiplyScalar<float>(e.BatchMatMul<float>(q, kT), 1f / (float)Math.Sqrt(W)); // [B,S,S]
                if (stop == "scores") outT = scores;
                else
                {
                    var sm = e.Softmax<float>(scores, 2);
                    if (stop == "softmax") outT = e.BatchMatMul<float>(sm, v);   // still exercises ctx matmul
                    else
                    {
                        var ctx = e.BatchMatMul<float>(sm, v);                    // [B,S,E]
                        if (stop == "ctx") outT = ctx;
                        else outT = e.Reshape<float>(e.TensorMatMul<float>(e.Reshape<float>(ctx, new[] { B * S, W }), wo), new[] { B, S, W });
                    }
                }
            }
        }
        var loss = e.ReduceSum<float>(e.TensorMultiply<float>(outT, outT), null, false);
        var srcs = new[] { h, wq, wk, wv, wo };
        var grads = tape.ComputeGradients(loss, srcs);
        float g0 = grads[wq].GetCpuData()[0];
        if (it == 0) Console.WriteLine($"  iter0 ok: loss={loss.GetCpuData()[0]:E3} g_wq[0]={g0:F4}");
    }
    Console.WriteLine($"stdattn-tape repro STOP={stop}: NO CRASH after 20 iters");
    return;
}

// #226 repro: the HRE spectral-FFT filter backward on GPU, looped under VRAM pressure to
// trigger the activation-cache eviction / deferred-download race. Mirrors SpectralFilterFFT.
if (args.Length > 0 && args[0] == "fftgrad")
{
    var fe = AiDotNetEngine.Current;   // GPU by default (unless AIDOTNET_DISABLE_GPU)
    int nr = args.Length > 1 ? int.Parse(args[1]) : 4096;
    int Ef = args.Length > 2 ? int.Parse(args[2]) : 256;
    int ni = args.Length > 3 ? int.Parse(args[3]) : 60;
    Console.WriteLine($"fftgrad #226 repro: rows={nr} E={Ef} iters={ni} engine={fe.GetType().Name}");
    var frng = new Random(5);
    float[] FRnd(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(frng.NextDouble() * 2 - 1); return a; }
    var fxd = FRnd(nr * Ef); var hred = FRnd(Ef); var himd = FRnd(Ef);
    var em = new float[2 * Ef]; var om = new float[2 * Ef];
    for (int k = 0; k < Ef; k++) { em[2 * k] = 1f; om[2 * k + 1] = 1f; }
    var evenMask = new Tensor<float>(em, new[] { 1, 2 * Ef });
    var oddMask = new Tensor<float>(om, new[] { 1, 2 * Ef });
    var realSelD = new float[] { 1f, 0f };
    var onesD = new float[nr]; for (int i = 0; i < nr; i++) onesD[i] = 1f;
    for (int it = 0; it < ni; it++)
    {
        var x2d = new Tensor<float>((float[])fxd.Clone(), new[] { nr, Ef });
        var Hre = new Tensor<float>((float[])hred.Clone(), new[] { Ef });
        var Him = new Tensor<float>((float[])himd.Clone(), new[] { Ef });
        using var tape = new GradientTape<float>();
        var ones = new Tensor<float>((float[])onesD.Clone(), new[] { nr, 1 });
        var xci = fe.TensorMultiply<float>(fe.TensorRepeatInterleave<float>(x2d, 2, 1), fe.TensorMatMul<float>(ones, evenMask));
        var F = Fft.Fft1<float>(xci);
        var Hil = fe.TensorAdd<float>(
            fe.TensorMultiply<float>(fe.TensorRepeatInterleave<float>(fe.Reshape<float>(Hre, new[] { 1, Ef }), 2, 1), evenMask),
            fe.TensorMultiply<float>(fe.TensorRepeatInterleave<float>(fe.Reshape<float>(Him, new[] { 1, Ef }), 2, 1), oddMask));
        var G = fe.TensorComplexMultiply<float>(fe.TensorMatMul<float>(ones, Hil), F);
        var yc = Fft.IFft1<float>(G);
        var realSel = new Tensor<float>((float[])realSelD.Clone(), new[] { 2, 1 });
        var yR = fe.TensorMatMul<float>(fe.Reshape<float>(yc, new[] { nr * Ef, 2 }), realSel);
        var loss = fe.ReduceSum<float>(fe.TensorMultiply<float>(yR, yR), null, false);
        var grads = tape.ComputeGradients(loss, new[] { x2d, Hre, Him });
        float g0 = grads[Hre].GetCpuData()[0];
        if (it % 10 == 0) Console.WriteLine($"  iter{it}: loss={loss.GetCpuData()[0]:E3} g_Hre[0]={g0:F4}");
    }
    Console.WriteLine($"fftgrad #226 repro: NO CRASH after {ni} iters");
    return;
}

// Micro-bench for the batched 1D complex FFT vs a dense E×E DFT BLAS matmul.
// Goal (A1): make our FFT kernel beat OpenBLAS on this batched workload.
// Reports ms/call AND bytes allocated/call (the per-row `new double[]` smoking gun).
int E = args.Length > 0 ? int.Parse(args[0]) : 512;
int iters = args.Length > 1 ? int.Parse(args[1]) : 300;
const int rows = 2048;
if (Environment.GetEnvironmentVariable("HE_FFT_ENGINE") == "cpu")
    AiDotNetEngine.Current = new CpuEngine();
var eng = AiDotNetEngine.Current;
Console.WriteLine($"  ENGINE = {eng.GetType().Name}");
var rng = new Random(3);

var xcData = new float[rows * 2 * E];   // interleaved complex, length-E per row (im=0)
for (int i = 0; i < rows; i++)
    for (int k = 0; k < E; k++) xcData[i * 2 * E + 2 * k] = (float)(rng.NextDouble() * 2 - 1);
var xc = new Tensor<float>(xcData, new[] { rows, 2 * E });

var xrData = new float[rows * E];
for (int i = 0; i < rows * E; i++) xrData[i] = (float)(rng.NextDouble() * 2 - 1);
var xr = new Tensor<float>(xrData, new[] { rows, E });

var cosData = new float[E * E]; var sinData = new float[E * E];
for (int n = 0; n < E; n++)
    for (int k = 0; k < E; k++) { double a = 2 * Math.PI * k * n / E; cosData[n * E + k] = (float)Math.Cos(a); sinData[n * E + k] = (float)Math.Sin(a); }
var cos = new Tensor<float>(cosData, new[] { E, E });
var sin = new Tensor<float>(sinData, new[] { E, E });

for (int w = 0; w < 5; w++) { var _ = Fft.Fft1<float>(xc); var __ = eng.TensorMatMul<float>(xr, cos); }

// --- correctness vs the trusted double reference path (unchanged) ---
var xcDdata = new double[rows * 2 * E];
for (int i = 0; i < xcData.Length; i++) xcDdata[i] = xcData[i];
var xcD = new Tensor<double>(xcDdata, new[] { rows, 2 * E });
var refD = Fft.Fft1<double>(xcD).GetCpuData();
var myF = Fft.Fft1<float>(xc).GetCpuData();
double maxAbs = 0, maxRel = 0;
for (int i = 0; i < myF.Length; i++)
{
    double e = Math.Abs(myF[i] - refD[i]);
    if (e > maxAbs) maxAbs = e;
    double d = Math.Abs(refD[i]);
    if (d > 1e-3) { double r = e / d; if (r > maxRel) maxRel = r; }
}
Console.WriteLine($"  correctness  : maxAbs={maxAbs:E3}  maxRel={maxRel:E3}  ({(maxAbs < 5e-2 ? "PASS" : "FAIL")})");

// inverse round-trip: ifft(fft(x)) ~= x  (validates inverse + norm scaling)
var rt = Fft.IFft1<float>(Fft.Fft1<float>(xc)).GetCpuData();
double rtErr = 0;
for (int i = 0; i < xcData.Length; i++) { double e = Math.Abs(rt[i] - xcData[i]); if (e > rtErr) rtErr = e; }
Console.WriteLine($"  ifft(fft(x)) : maxAbs={rtErr:E3}  ({(rtErr < 5e-2 ? "PASS" : "FAIL")})");

// ---- Spectral-filter drop-in check: HRE's y=(1/E)IDFT(H⊙DFT(x)) dense-matmul vs FFT ----
// Proves the fast FFT is a numerically exact replacement for the dense cos/sin DFT the
// HRE SpectralFilter uses. (Differentiability already covered by FftGradcheckTests.)
{
    var Hre = new float[E]; var Him = new float[E];
    for (int k = 0; k < E; k++) { Hre[k] = (float)(rng.NextDouble() * 2 - 1); Him[k] = (float)(rng.NextDouble() * 2 - 1); }

    // DENSE (exactly the HRE math): Fre=x@cos, Fim=-(x@sin); G=H⊙F; y=(Gre@cos-Gim@sin)/E
    var xrArr = xr.GetCpuData();
    var FreT = eng.TensorMatMul<float>(xr, cos).GetCpuData();
    var FimRaw = eng.TensorMatMul<float>(xr, sin).GetCpuData();
    var yDense = new float[rows * E];
    var GreA = new float[rows * E]; var GimA = new float[rows * E];
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < E; k++)
        {
            float Fre = FreT[i * E + k], Fim = -FimRaw[i * E + k];
            GreA[i * E + k] = Fre * Hre[k] - Fim * Him[k];
            GimA[i * E + k] = Fre * Him[k] + Fim * Hre[k];
        }
    var yD1 = eng.TensorMatMul<float>(new Tensor<float>(GreA, new[] { rows, E }), cos).GetCpuData();
    var yD2 = eng.TensorMatMul<float>(new Tensor<float>(GimA, new[] { rows, E }), sin).GetCpuData();
    for (int i = 0; i < rows * E; i++) yDense[i] = (yD1[i] - yD2[i]) / E;

    // FFT: interleave x (imag=0) -> Fft1 -> H⊙ -> IFft1 -> real part
    var xcS = new float[rows * 2 * E];
    for (int i = 0; i < rows; i++) for (int k = 0; k < E; k++) xcS[i * 2 * E + 2 * k] = xrArr[i * E + k];
    var Fc = Fft.Fft1<float>(new Tensor<float>(xcS, new[] { rows, 2 * E })).GetCpuData();
    var GcS = new float[rows * 2 * E];
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < E; k++)
        {
            float Fre = Fc[i * 2 * E + 2 * k], Fim = Fc[i * 2 * E + 2 * k + 1];
            GcS[i * 2 * E + 2 * k] = Fre * Hre[k] - Fim * Him[k];
            GcS[i * 2 * E + 2 * k + 1] = Fre * Him[k] + Fim * Hre[k];
        }
    var yc = Fft.IFft1<float>(new Tensor<float>(GcS, new[] { rows, 2 * E })).GetCpuData();
    double sMax = 0, sDen = 0;
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < E; k++)
        {
            double e = Math.Abs(yc[i * 2 * E + 2 * k] - yDense[i * E + k]);
            if (e > sMax) sMax = e;
            double d = Math.Abs(yDense[i * E + k]); if (d > sDen) sDen = d;
        }
    Console.WriteLine($"  spectral A/B : dense-vs-FFT maxAbs={sMax:E3} (scale {sDen:F2})  ({(sMax < 1e-3 * (1 + sDen) ? "MATCH" : "MISMATCH")})");

    // --- EXACT HRE tensor-op formulation: RepeatInterleave + masks + TensorComplexMultiply + FFT ---
    // (this is the differentiable pipeline that will replace the dense cos/sin matmuls in SpectralFilter)
    var evenMaskA = new float[2 * E]; var oddMaskA = new float[2 * E];
    for (int k = 0; k < E; k++) { evenMaskA[2 * k] = 1f; oddMaskA[2 * k + 1] = 1f; }
    var evenMask = new Tensor<float>(evenMaskA, new[] { 1, 2 * E });
    var oddMask = new Tensor<float>(oddMaskA, new[] { 1, 2 * E });
    var realSel = new Tensor<float>(new[] { 1f, 0f }, new[] { 2, 1 });
    var onesRA = new float[rows]; for (int i = 0; i < rows; i++) onesRA[i] = 1f;
    var onesR = new Tensor<float>(onesRA, new[] { rows, 1 });
    var HreT = new Tensor<float>((float[])Hre.Clone(), new[] { 1, E });
    var HimT = new Tensor<float>((float[])Him.Clone(), new[] { 1, E });

    var xRep = eng.TensorRepeatInterleave<float>(xr, 2, 1);
    var evenB = eng.TensorMatMul<float>(onesR, evenMask);
    var xcOps = eng.TensorMultiply<float>(xRep, evenB);
    var Fops = Fft.Fft1<float>(xcOps);
    var HreRep = eng.TensorRepeatInterleave<float>(HreT, 2, 1);
    var HimRep = eng.TensorRepeatInterleave<float>(HimT, 2, 1);
    var Hil = eng.TensorAdd<float>(eng.TensorMultiply<float>(HreRep, evenMask), eng.TensorMultiply<float>(HimRep, oddMask));
    var HilB = eng.TensorMatMul<float>(onesR, Hil);
    var Gops = eng.TensorComplexMultiply<float>(HilB, Fops);
    var ycOps = Fft.IFft1<float>(Gops);
    var ycR = eng.Reshape<float>(ycOps, new[] { rows * E, 2 });
    var yOps = eng.Reshape<float>(eng.TensorMatMul<float>(ycR, realSel), new[] { rows, E }).GetCpuData();
    double oMax = 0;
    for (int i = 0; i < rows * E; i++) { double e = Math.Abs(yOps[i] - yDense[i]); if (e > oMax) oMax = e; }
    Console.WriteLine($"  spectral ops : tensor-op-FFT vs dense maxAbs={oMax:E3}  ({(oMax < 1e-3 * (1 + sDen) ? "MATCH" : "MISMATCH")})");
}

if (E <= 8)
{
    Trace(xcData, E);
    // naive DFT of row 0 (independent ground truth): X[k] = sum_n x[n] exp(-2pi i kn/N)
    Console.WriteLine("  row0  k :   mine(re,im)      ref(re,im)       naive(re,im)");
    for (int k = 0; k < E; k++)
    {
        double nr = 0, ni = 0;
        for (int nn = 0; nn < E; nn++)
        {
            double xr0 = xcData[2 * nn], xi0 = xcData[2 * nn + 1];
            double a = -2 * Math.PI * k * nn / E;
            double c = Math.Cos(a), s = Math.Sin(a);
            nr += xr0 * c - xi0 * s;
            ni += xr0 * s + xi0 * c;
        }
        Console.WriteLine($"       {k} : ({myF[2 * k],7:F3},{myF[2 * k + 1],7:F3})  ({refD[2 * k],7:F3},{refD[2 * k + 1],7:F3})  ({nr,7:F3},{ni,7:F3})");
    }
}

long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
var sw = Stopwatch.StartNew();
for (int it = 0; it < iters; it++) { var _ = Fft.Fft1<float>(xc); }
sw.Stop();
long allocAfter = GC.GetTotalAllocatedBytes(precise: true);
double fftMs = sw.Elapsed.TotalMilliseconds / iters;
double fftAllocMB = (allocAfter - allocBefore) / (double)iters / (1024 * 1024);

var sw2 = Stopwatch.StartNew();
for (int it = 0; it < iters; it++) { var a = eng.TensorMatMul<float>(xr, cos); var b = eng.TensorMatMul<float>(xr, sin); }
sw2.Stop();
double denseMs = sw2.Elapsed.TotalMilliseconds / iters;

Console.WriteLine($"[rows={rows} E={E} iters={iters}]");
Console.WriteLine($"  Fft.Fft1 : {fftMs,8:F3} ms/call   {fftAllocMB,8:F3} MB alloc/call");
Console.WriteLine($"  denseDFT : {denseMs,8:F3} ms/call   (2 BLAS matmuls)");
Console.WriteLine($"  FFT is {denseMs / fftMs:F2}x {(fftMs < denseMs ? "FASTER" : "SLOWER")} than dense-BLAS  (target: FASTER)");

// Single-row replica of the batch-float kernel (lb=1) to isolate algorithm vs batching.
static void Trace(float[] xrow, int n)
{
    int logn = System.Numerics.BitOperations.Log2((uint)n);
    var brev = new int[n];
    for (int i = 0; i < n; i++) { int r = 0, x = i; for (int b = 0; b < logn; b++) { r = (r << 1) | (x & 1); x >>= 1; } brev[i] = r; }
    var twR = new float[n]; var twI = new float[n];
    int o = 0;
    for (int len = 2; len <= n; len <<= 1) { int half = len >> 1; for (int k = 0; k < half; k++) { double a = -2 * Math.PI * k / len; twR[o + k] = (float)Math.Cos(a); twI[o + k] = (float)Math.Sin(a); } o += half; }
    var re = new float[n]; var im = new float[n];
    for (int i = 0; i < n; i++) { int j = brev[i]; re[j] = xrow[2 * i]; im[j] = xrow[2 * i + 1]; }
    int off = 0;
    for (int len = 2; len <= n; len <<= 1)
    {
        int half = len >> 1;
        for (int k = 0; k < half; k++)
        {
            float wr = twR[off + k], wi = twI[off + k];
            for (int p0 = 0; p0 < n; p0 += len)
            {
                int A = p0 + k, B = p0 + k + half;
                float ar = re[A], ai = im[A], br = re[B], bi = im[B];
                float tr = wr * br - wi * bi, ti = wr * bi + wi * br;
                re[A] = ar + tr; im[A] = ai + ti; re[B] = ar - tr; im[B] = ai - ti;
            }
        }
        off += half;
    }
    Console.Write("  TRACE(single-row) : ");
    for (int k = 0; k < n; k++) Console.Write($"({re[k]:F3},{im[k]:F3}) ");
    Console.WriteLine();
}
