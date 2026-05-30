using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
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
