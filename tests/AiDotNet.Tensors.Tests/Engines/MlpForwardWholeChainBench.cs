using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// In-process validation of the compiled-inference Phase 1+3 wins WITHOUT a
/// package release: times the whole AIsEval MLP forward (784→512→128→10) via the
/// new <see cref="CpuEngine.MlpForward{T}"/> (managed cached-B routing + ping-pong
/// scratch, last layer writes the result) against a reconstruction of the OLD
/// path (native BLAS GEMM + pinned→managed copy + separate bias/act pass + a
/// fresh output buffer per layer). The AiDotNet↔torch.compile comparison needs
/// the released package; this measures the kernel-path delta we control directly.
/// Env-gated (AIDOTNET_RUN_JIT_PERF=1); CI-safe no-op otherwise.
/// </summary>
// #513: the Phase-F probe scopes the PROCESS-GLOBAL OpenBLAS thread count
// (ScopeOpenBlasThreads). Even though it's env-gated, classify it with the other
// global-BLAS-state tests so it can never run in parallel with the bit-match GEMM
// tests (which would see a transient thread count → reduction-order bit mismatch).
[Collection("BlasManaged-Stats-Serial")]
public class MlpForwardWholeChainBench
{
    private readonly ITestOutputHelper _output;
    public MlpForwardWholeChainBench(ITestOutputHelper output) => _output = output;

    [Fact]
    public void WholeMlp_NewPath_vs_OldNativePerLayer()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        var engine = new CpuEngine();
        int[] dims = { 784, 512, 128, 10 };   // AIsEval MLP
        _output.WriteLine($"HasRawSgemm={BlasProvider.HasRawSgemm}, IsMklVerified={BlasProvider.IsMklVerified}, cores={Environment.ProcessorCount}");

        foreach (int m in new[] { 1, 8, 32, 128 })
        {
            var input = MakeTensor(m, dims[0]);
            var weights = new List<Tensor<float>>();
            var biases = new List<Tensor<float>?>();
            for (int l = 0; l < dims.Length - 1; l++)
            {
                weights.Add(MakeTensor(dims[l], dims[l + 1]));
                biases.Add(MakeTensor(1, dims[l + 1]));
            }

            // NEW path — single MlpForward call (routing gate + ping-pong).
            _ = engine.MlpForward(input, weights, biases, FusedActivationType.ReLU);  // warm prepack cache
            double newMs = Bench(() => { var _ = engine.MlpForward(input, weights, biases, FusedActivationType.ReLU); });

            // OLD path — native BLAS + copy + separate pass + per-layer fresh buffer.
            double oldMs = Bench(() => OldNativePerLayer(input, weights, biases, m, dims));

            // Parity cross-check (final layer output).
            var newOut = engine.MlpForward(input, weights, biases, FusedActivationType.ReLU).ToArray();
            var oldOut = OldNativePerLayerResult(input, weights, biases, m, dims);
            double maxDiff = 0;
            for (int i = 0; i < newOut.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(newOut[i] - oldOut[i]));

            string verdict = newMs < oldMs ? "NEW wins" : "old wins";
            _output.WriteLine(
                $"[m={m,4}]  new {newMs * 1000:F1}us  old(native+copy+sep+alloc) {oldMs * 1000:F1}us  " +
                $"ratio {oldMs / newMs:F2}x  {verdict}  (maxDiff {maxDiff:E2})");
        }
    }

    [Fact]
    public void TinyModel_CompiledMlp_SerialVsParallel_Probe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        // Phase 4 probe: does forcing single-thread help a cache-resident tiny
        // model, or do its GEMMs already run serial (below ParallelWorkThreshold)?
        int[] dims = { 64, 48, 16, 5 };
        int layers = dims.Length - 1;
        var w = new List<float[]>(); var b = new List<float[]?>(); var inF = new List<int>(); var outF = new List<int>();
        for (int l = 0; l < layers; l++)
        {
            w.Add(MakeArr(dims[l] * dims[l + 1], 100 + l));
            b.Add(MakeArr(dims[l + 1], 200 + l));
            inF.Add(dims[l]); outF.Add(dims[l + 1]);
        }
        var plan = CompiledMlp.Create(w, b, inF, outF, FusedActivationType.ReLU, FusedActivationType.None, maxBatch: 64);
        long modelBytes = 0; foreach (var ww in w) modelBytes += (long)ww.Length * 4;
        _output.WriteLine($"ParallelWorkThreshold={SimdGemm.ParallelWorkThreshold} FMAs; tiny-model weights={modelBytes} B");

        bool savedPar = SimdGemm.UseParallelGemm;
        try
        {
            foreach (int m in new[] { 1, 8, 32 })
            {
                var input = MakeArr(m * dims[0], 7);
                var output = new float[m * plan.OutputFeatures];
                SimdGemm.UseParallelGemm = true;
                double par = Bench(() => plan.Run(input, m, output));
                SimdGemm.UseParallelGemm = false;
                double ser = Bench(() => plan.Run(input, m, output));
                _output.WriteLine($"[m={m,3}]  parallel-flag {par * 1000:F1}us  serial-flag {ser * 1000:F1}us  ratio {par / ser:F2}x");
            }
        }
        finally { SimdGemm.UseParallelGemm = savedPar; }
    }

    [Fact]
    public void Phase5_CompiledMlp_SelfTuningHeadroom_Probe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        // Phase 5 headroom probe (measure-before-build): the static CompiledMlp
        // ALWAYS routes every layer through managed cached-B SgemmWithCachedB.
        // The FusedLinear lesson is that native BLAS overtakes managed cached-B
        // at large M (batch). If a per-batch best-of {managed, native} chain
        // beats static-managed at some batch, a self-tuning plan that observes
        // the steady-state batch and locks the winning per-layer kernel has real
        // headroom to capture. If managed wins at every batch, there is nothing
        // to self-tune here (cf. the Phase-4 finding that tiny models already
        // run serial) — and we ship no speculative loop.
        int[] dims = { 784, 512, 128, 10 };   // AIsEval MLP
        int layers = dims.Length - 1;
        _output.WriteLine($"HasRawSgemm={BlasProvider.HasRawSgemm}, IsMklVerified={BlasProvider.IsMklVerified}, cores={Environment.ProcessorCount}");

        const int maxB = 512;
        var w = new List<float[]>(); var b = new List<float[]?>(); var inF = new List<int>(); var outF = new List<int>();
        var wT = new List<Tensor<float>>(); var bT = new List<Tensor<float>?>();
        for (int l = 0; l < layers; l++)
        {
            var wa = MakeArr(dims[l] * dims[l + 1], 100 + l);
            var ba = MakeArr(dims[l + 1], 200 + l);
            w.Add(wa); b.Add(ba); inF.Add(dims[l]); outF.Add(dims[l + 1]);
            var wt = new Tensor<float>(new[] { dims[l], dims[l + 1] }); wa.AsSpan().CopyTo(wt.AsWritableSpan()); wT.Add(wt);
            var bt = new Tensor<float>(new[] { 1, dims[l + 1] }); ba.AsSpan().CopyTo(bt.AsWritableSpan()); bT.Add(bt);
        }
        var plan = CompiledMlp.Create(w, b, inF, outF, FusedActivationType.ReLU, FusedActivationType.None, maxBatch: maxB);

        foreach (int m in new[] { 1, 8, 32, 128, 512 })
        {
            var inputArr = MakeArr(m * dims[0], 7);
            var output = new float[m * plan.OutputFeatures];
            var inputT = new Tensor<float>(new[] { m, dims[0] });
            inputArr.AsSpan().CopyTo(inputT.AsWritableSpan());

            // Gated: CompiledMlp.Run (per-layer managed/native self-tuner).
            double gated = Bench(() => plan.Run(inputArr, m, output));

            // Native-always: per-layer native-BLAS GEMM + fused epilogue, reusing buffers.
            double native = Bench(() => NativePerLayer(inputArr, m, dims, w, b, KernelForce.Native));

            // Managed-always: per-layer managed cached-B (the OLD static CompiledMlp).
            double managed = Bench(() => NativePerLayer(inputArr, m, dims, w, b, KernelForce.Managed));

            double best = Math.Min(gated, Math.Min(native, managed));
            string verdict = gated <= best + 1e-9 ? "GATED ≤ best" : $"gated {gated / best:F2}× of best";
            _output.WriteLine($"[m={m,4}]  gated {gated * 1000:F1}us  native-always {native * 1000:F1}us  managed-always {managed * 1000:F1}us  → {verdict}");
        }
    }

    [Fact]
    public void Phase7_EndToEndEntryPath_Probe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        // Phase 7 (option E) headroom probe: compares the END-TO-END entry-path cost
        // of the Tensor-based engine MlpForward versus the array-based CompiledMlp.Run.
        // The printed delta is NOT pure per-call overhead: CompiledMlp.Run uses per-batch
        // self-tuned kernel selection while MlpForward follows the static preferManaged
        // gate, so the gap also folds in routing-policy differences alongside the entry
        // costs MlpForward pays that Run does not (AutoTensorCache.RentOrAllocate for the
        // result tensor, two ArrayPool rents for ping-pong scratch, the BLAS thread-cap
        // scope, shape checks). Treat the number as "which inference entry is faster
        // end-to-end on this shape", a bound on what a leaner entry+routing could save —
        // not an isolated overhead measurement.
        var engine = new CpuEngine();
        int[] dims = { 784, 512, 128, 10 };
        int layers = dims.Length - 1;
        var w = new List<float[]>(); var b = new List<float[]?>(); var inF = new List<int>(); var outF = new List<int>();
        var wT = new List<Tensor<float>>(); var bT = new List<Tensor<float>?>();
        for (int l = 0; l < layers; l++)
        {
            var wa = MakeArr(dims[l] * dims[l + 1], 100 + l);
            var ba = MakeArr(dims[l + 1], 200 + l);
            w.Add(wa); b.Add(ba); inF.Add(dims[l]); outF.Add(dims[l + 1]);
            var wt = new Tensor<float>(new[] { dims[l], dims[l + 1] }); wa.AsSpan().CopyTo(wt.AsWritableSpan()); wT.Add(wt);
            var bt = new Tensor<float>(new[] { 1, dims[l + 1] }); ba.AsSpan().CopyTo(bt.AsWritableSpan()); bT.Add(bt);
        }
        var plan = CompiledMlp.Create(w, b, inF, outF, FusedActivationType.ReLU, FusedActivationType.None, maxBatch: 512);

        foreach (int m in new[] { 1, 8, 32, 128 })
        {
            var inputArr = MakeArr(m * dims[0], 7);
            var output = new float[m * plan.OutputFeatures];
            var inputT = new Tensor<float>(new[] { m, dims[0] });
            inputArr.AsSpan().CopyTo(inputT.AsWritableSpan());

            double compiled = Bench(() => plan.Run(inputArr, m, output));
            double mlpFwd = Bench(() => { var _ = engine.MlpForward(inputT, wT, bT, FusedActivationType.ReLU); });
            _output.WriteLine($"[m={m,4}]  CompiledMlp.Run {compiled * 1000:F1}us  engine.MlpForward {mlpFwd * 1000:F1}us  delta {(mlpFwd - compiled) * 1000:F1}us ({mlpFwd / compiled:F2}×, end-to-end incl. routing policy)");
        }
    }

    [Fact]
    public void Phase7F_GemmAutotuneHeadroom_Probe()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        // Option F headroom: the self-tuner currently A/Bs managed cached-B vs
        // native BLAS at DEFAULT (all-core) threads. #475 found wide-layer
        // parallel scaling plateaus ~2× — i.e. all-core native may oversubscribe
        // the small-M wide GEMM. Sweep native thread counts {1,2,4,8,16} per
        // (shape, M) and compare best-of-all-threads against the self-tuner's
        // current best-of-{managed, native@all}. If a capped thread count beats
        // native@all, F has real headroom → add thread-count to the tuner's
        // candidate set; if native@all is already best, F is a no-op here.
        if (!BlasProvider.HasRawSgemm) { _output.WriteLine("no native sgemm; skipping"); return; }
        _output.WriteLine($"cores={Environment.ProcessorCount}");

        // CONCLUSION (measured): native@cores/2 beats native@all-cores ~1.3× here
        // when the thread count is held FIXED across calls. But capturing it in
        // the self-tuner via a per-call ScopeOpenBlasThreads BACKFIRES: changing
        // the OpenBLAS/OMP thread count rebuilds the thread pool (~700 µs), so
        // flipping it per layer per Run made CompiledMlp ~6× SLOWER (110→836 µs at
        // M=1). The win is only realizable by pinning the process-global BLAS
        // thread count ONCE at the deployment/serving layer — NOT a per-call
        // primitive concern. So this stays a probe; the self-tuner keeps
        // native@ambient. (This is the measured reason option F was not wired.)
        // The wide layers of the AIsEval MLP: 784→512 and 512→128.
        (int k, int n)[] shapes = { (784, 512), (512, 128) };
        int[] threadCounts = { 1, 2, 4, 8, 16 };

        foreach (var (k, n) in shapes)
        {
            var wArr = MakeArr(k * n, 11);
            foreach (int m in new[] { 1, 8, 32, 128 })
            {
                var src = MakeArr(m * k, 7);
                var dst = new float[m * n];

                double managed = Bench(() => SimdGemm.SgemmWithCachedB(src.AsSpan(0, m * k), wArr, dst.AsSpan(0, m * n), m, k, n));

                // Ambient native baseline (NO scope): this is exactly what the self-tuner's
                // native candidate runs per call — process-global thread count, no per-call
                // retune. Use THIS as the tuner baseline, not perThread[last], which would
                // (a) assume the process ambient count equals the last swept value and
                // (b) include the pool-rebuild cost (see below).
                double nativeAmbient = Bench(() =>
                {
                    unsafe { fixed (float* ps = src, pw = wArr, pd = dst) BlasProvider.SgemmRaw(m, n, k, ps, k, pw, n, pd, n); }
                });

                // Fixed-thread sweep: set the OpenBLAS thread count ONCE *outside* the timed
                // region so perThread[ti] measures steady-state throughput held fixed at th
                // threads — NOT the per-call ScopeOpenBlasThreads pool-rebuild (~700µs) that
                // the CONCLUSION above documents as why option F can't be a per-call primitive.
                var perThread = new double[threadCounts.Length];
                for (int ti = 0; ti < threadCounts.Length; ti++)
                {
                    int th = threadCounts[ti];
                    using (BlasProvider.ScopeOpenBlasThreads(th))
                        perThread[ti] = Bench(() =>
                        {
                            unsafe { fixed (float* ps = src, pw = wArr, pd = dst) BlasProvider.SgemmRaw(m, n, k, ps, k, pw, n, pd, n); }
                        });
                }
                int bestTi = 0; for (int ti = 1; ti < perThread.Length; ti++) if (perThread[ti] < perThread[bestTi]) bestTi = ti;
                double tunerPick = Math.Min(managed, nativeAmbient);     // true self-tuner baseline (managed vs native@ambient)
                double fBest = Math.Min(managed, perThread[bestTi]);     // best achievable if a fixed thread count were pinned
                string verdict = fBest < tunerPick - 0.0005 ? $"F WINS via native@{threadCounts[bestTi]}t-fixed ({tunerPick / fBest:F2}×)" : "no F headroom";
                _output.WriteLine(
                    $"[{k}->{n} m={m,4}] managed {managed*1000:F1}  native@ambient {nativeAmbient*1000:F1}  fixed@[" +
                    string.Join(",", System.Linq.Enumerable.Range(0, threadCounts.Length).Select(i => $"{threadCounts[i]}t:{perThread[i]*1000:F0}")) +
                    $"]us  tuner {tunerPick*1000:F1}  Fbest {fBest*1000:F1}  → {verdict}");
            }
        }
    }

    private enum KernelForce { Native, Managed }
    private static readonly float[][] _nativeScratch = new float[2][];
    private static void NativePerLayer(float[] input, int m, int[] dims, List<float[]> weights, List<float[]?> biases, KernelForce force)
    {
        int layers = weights.Count;
        float[] src = input;
        int k = dims[0];
        for (int l = 0; l < layers; l++)
        {
            int n = dims[l + 1];
            if (_nativeScratch[l % 2] is null || _nativeScratch[l % 2]!.Length < m * n)
                _nativeScratch[l % 2] = new float[m * n];
            var dst = _nativeScratch[l % 2]!;
            if (force == KernelForce.Native && BlasProvider.HasRawSgemm)
            {
                unsafe
                {
                    fixed (float* ps = src, pw = weights[l], pd = dst)
                        BlasProvider.SgemmRaw(m, n, k, ps, k, pw, n, pd, n);
                }
            }
            else
            {
                SimdGemm.SgemmWithCachedB(src.AsSpan(0, m * k), weights[l], dst.AsSpan(0, m * n), m, k, n);
            }
            var act = l == layers - 1 ? FusedActivationType.None : FusedActivationType.ReLU;
            CpuFusedOperations.ApplyBiasActivationInPlace(dst, biases[l], m, n, act);
            src = dst;
            k = n;
        }
    }

    private static float[] MakeArr(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }

    private static void OldNativePerLayer(Tensor<float> input, List<Tensor<float>> weights, List<Tensor<float>?> biases, int m, int[] dims)
    {
        _ = OldNativePerLayerResult(input, weights, biases, m, dims);
    }

    private static float[] OldNativePerLayerResult(Tensor<float> input, List<Tensor<float>> weights, List<Tensor<float>?> biases, int m, int[] dims)
    {
        float[] src = input.ToArray();
        int k = dims[0];
        int last = weights.Count - 1;
        for (int l = 0; l < weights.Count; l++)
        {
            int n = dims[l + 1];
            var wArr = weights[l].ToArray();
            var scratch = new float[m * n];           // fresh per-layer alloc (old behaviour)
            var outArr = new float[m * n];
            if (BlasProvider.HasRawSgemm)
            {
                unsafe
                {
                    fixed (float* ps = src, pw = wArr, pSc = scratch)
                        BlasProvider.SgemmRaw(m, n, k, ps, k, pw, n, pSc, n);
                }
                Array.Copy(scratch, outArr, m * n);   // pinned→managed copy
            }
            else
            {
                SimdGemm.SgemmWithCachedB(src.AsSpan(0, m * k), wArr, outArr.AsSpan(0, m * n), m, k, n);
            }
            var bArr = biases[l]!.ToArray();
            var act = l == last ? FusedActivationType.None : FusedActivationType.ReLU;
            CpuFusedOperations.ApplyBiasActivationInPlace(outArr, bArr, m, n, act);
            src = outArr;
            k = n;
        }
        return src;
    }

    private static double Bench(Action f)
    {
        for (int w = 0; w < 20; w++) f();
        double best = double.MaxValue;
        for (int r = 0; r < 100; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static Tensor<float> MakeTensor(int r, int c)
    {
        var rng = new Random(2026);
        var t = new Tensor<float>(new[] { r, c });
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
