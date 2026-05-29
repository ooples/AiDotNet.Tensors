using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Issue #471 repro: CPU engine hard-crashes (exit 255, no managed exception) on a large TensorMatMul
/// contracting over a big dimension (V≈50257), on AVX2-no-AVX512 hardware with OpenBLAS NOT loaded (managed
/// AVX2 fallback). Env-gated TENSORS_RUN_REPRO=1. A clean run prints "ALL OK".
/// </summary>
public class Issue471CpuMatmulCrashRepro
{
    private readonly ITestOutputHelper _o;
    public Issue471CpuMatmulCrashRepro(ITestOutputHelper o) { _o = o; }

    [SkippableFact]
    public void Cpu_large_matmul_contracting_over_V()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("TENSORS_RUN_REPRO") == "1",
            "Env-gated #471 repro: set TENSORS_RUN_REPRO=1 to run.");

        // ResetToCpu() mutates AiDotNetEngine.Current — a shared process-wide
        // singleton. xUnit test order is nondeterministic, so without restoring
        // the previous selection the GPU repros below could silently run on CPU
        // and miss the hardware path they exist to cover. Capture-and-restore
        // around the CPU-specific body keeps this test isolated.
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.ResetToCpu();
        try
        {
            var eng = AiDotNetEngine.Current;
            _o.WriteLine($"engine={eng.Name}");
            const int d = 128;
            foreach (int V in new[] { 1000, 10000, 30000, 50257 })
            {
                var aData = new double[d * V]; for (int i = 0; i < aData.Length; i++) aData[i] = 0.01;
                var bData = new double[V * d]; for (int i = 0; i < bData.Length; i++) bData[i] = 0.01;
                var a = new Tensor<double>(aData, new[] { d, V });
                var b = new Tensor<double>(bData, new[] { V, d });
                _o.WriteLine($"matmul [d={d} x V={V}] x [V={V} x d={d}] ...");
                var c = eng.TensorMatMul(a, b);
                _o.WriteLine($"  OK V={V}: c[0,0]={c[0, 0]:F4} (expect {V * 0.01 * 0.01:F4})");
            }
            // FLOAT path (matches HE's training-readout dtype; float is the oneDNN/SimdGemm-Sgemm dispatch)
            foreach (int V in new[] { 10000, 50257 })
            {
                var aData = new float[d * V]; for (int i = 0; i < aData.Length; i++) aData[i] = 0.01f;
                var bData = new float[V * d]; for (int i = 0; i < bData.Length; i++) bData[i] = 0.01f;
                var a = new Tensor<float>(aData, new[] { d, V });
                var b = new Tensor<float>(bData, new[] { V, d });
                _o.WriteLine($"FLOAT matmul [d={d} x V={V}] x [V={V} x d={d}] ...");
                var c = eng.TensorMatMul(a, b);
                _o.WriteLine($"  OK float V={V}: c[0,0]={c[0, 0]:F4}");
            }
            _o.WriteLine("ALL OK — no crash");
            Assert.True(true);
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    /// <summary>
    /// GPU training-loop stress: the HE facade-Transformer training crashed (exit 255) on gfx1012/RDNA1 during the
    /// high-frequency minibatch loop. Repro: default (auto-detected GPU) engine, many sequential matmuls of
    /// training-batch shapes + large [B,V] readout matmuls, mimicking the per-batch kernel-launch cadence.
    /// Env-gated TENSORS_RUN_REPRO=1. A clean run prints "GPU ALL OK".
    /// </summary>
    [SkippableFact]
    public void Gpu_training_loop_stress()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("TENSORS_RUN_REPRO") == "1",
            "Env-gated #471 repro: set TENSORS_RUN_REPRO=1 to run.");
        var eng = AiDotNetEngine.Current;   // default = auto-detected GPU on this box
        _o.WriteLine($"engine={eng.Name}");
        const int B = 128, d = 128, ff = 512, V = 15001;
        // per-batch matmuls a training loop issues: x·W1 [B,d]x[d,ff], h·W2 [B,ff]x[ff,d], readout [B,d]x[d,V]
        float Fill(int i) => 0.001f * ((i % 17) - 8);
        Tensor<float> Mk(int r, int c) { var v = new float[r * c]; for (int i = 0; i < v.Length; i++) v[i] = Fill(i); return new Tensor<float>(v, new[] { r, c }); }
        var x = Mk(B, d); var w1 = Mk(d, ff); var w2 = Mk(ff, d); var wOut = Mk(d, V);
        int iters = 3000;
        for (int it = 0; it < iters; it++)
        {
            var h = eng.TensorMatMul(x, w1);     // [B,ff]
            var o = eng.TensorMatMul(h, w2);     // [B,d]
            var logits = eng.TensorMatMul(o, wOut); // [B,V] — the big readout
            if (it % 500 == 0) _o.WriteLine($"  iter {it}: logits[0,0]={logits[0, 0]:F4}");
        }
        _o.WriteLine($"GPU ALL OK — {iters} iters, no crash");
        Assert.True(true);
    }

    /// <summary>
    /// GPU AUTODIFF training stress — the faithful TrainWithTape workload: per-iter GradientTape, forward
    /// matmul-chain + ReduceSum loss, then ComputeGradients (the BACKWARD pass = the prime crash suspect), with a
    /// CPU-side weight update to keep values bounded. Many iters on the default (GPU) engine. Env TENSORS_RUN_REPRO=1.
    /// </summary>
    [SkippableFact]
    public void Gpu_autodiff_training_stress()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("TENSORS_RUN_REPRO") == "1",
            "Env-gated #471 repro: set TENSORS_RUN_REPRO=1 to run.");
        var eng = AiDotNetEngine.Current;
        _o.WriteLine($"engine={eng.Name}");
        const int B = 64, d = 128, ff = 512, V = 15001;
        float Fill(int i) => 0.001f * ((i % 17) - 8);
        Tensor<float> Mk(int r, int c) { var v = new float[r * c]; for (int i = 0; i < v.Length; i++) v[i] = Fill(i); return new Tensor<float>(v, new[] { r, c }); }
        var x = Mk(B, d); var w1 = Mk(d, ff); var w2 = Mk(ff, d); var wOut = Mk(d, V);
        const float lr = 1e-4f;
        int iters = 1000;
        for (int it = 0; it < iters; it++)
        {
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
            var h = eng.TensorMatMul(x, w1);
            var o = eng.TensorMatMul(h, w2);
            var logits = eng.TensorMatMul(o, wOut);
            var loss = eng.ReduceSum(logits, null);
            var grads = tape.ComputeGradients(loss, new[] { w1, w2, wOut });   // BACKWARD on GPU — suspect
            // CPU-side SGD to keep weights bounded (update mechanics irrelevant to the crash)
            void Upd(Tensor<float> w, Tensor<float> g) { var sw = w.AsWritableSpan(); var sg = g.AsWritableSpan(); for (int i = 0; i < sw.Length; i++) sw[i] -= lr * sg[i]; }
            Upd(w1, grads[w1]); Upd(w2, grads[w2]); Upd(wOut, grads[wOut]);
            if (it % 200 == 0) { try { double l = Convert.ToDouble(loss.AsWritableSpan()[0]); _o.WriteLine($"  iter {it}: loss={l:E3}"); } catch { _o.WriteLine($"  iter {it}: ok"); } }
        }
        _o.WriteLine($"GPU AUTODIFF ALL OK — {iters} backward iters, no crash");
        Assert.True(true);
    }
}
