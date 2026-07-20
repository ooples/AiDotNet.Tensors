#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests;

/// <summary>
/// Localizes the Phase-2 transparent-fusion gap: does a SINGLE op materialize its result when run inside a
/// deferred scope (capture -> optimize -> replay)? If a trivial matmul comes back all-zero, the deferred
/// path is broken at the basic output-materialization level (not decoder-specific). Skips without a GPU.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeferredScopeReproTests
{
    [SkippableFact]
    public void DeferredScope_SingleMatMul_MaterializesSameAsEager()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

            // Eager reference on the same GPU engine.
            var eager = gpu.TensorMatMul(a, b).Contiguous().AsSpan().ToArray();

            var scope = gpu.BeginDeferredScope();
            if (scope is null) { Skip.If(true, "Backend has no deferred execution"); return; }

            float[] deferred;
            using (scope)
            {
                var r = gpu.TensorMatMul(a, b); // records into the graph
                scope.Execute();                // optimize + replay
                deferred = r.Contiguous().AsSpan().ToArray();
            }

            bool allZero = true;
            for (int i = 0; i < deferred.Length; i++) if (Math.Abs(deferred[i]) > 1e-9) { allZero = false; break; }
            Assert.False(allZero, "Deferred replay produced all-zero output — output buffer not materialized.");

            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - deferred[i]) < 1e-3f,
                    $"[{i}] eager {eager[i]} vs deferred {deferred[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    /// <summary>
    /// The decoder token-0 repro at op level: RMSNorm reading the output of a DEFERRED matmul. If RmsNorm is
    /// not recorded it runs eagerly during recording on the matmul's not-yet-filled buffer, producing garbage
    /// (all-zero / wrong) — which is exactly what poisoned the whole decoder forward. With RmsNorm recorded,
    /// both ops replay in order and the result matches eager.
    /// </summary>
    [SkippableFact]
    public void DeferredScope_MatMulThenRmsNorm_MaterializesSameAsEager()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
            var gamma = new Tensor<float>(new float[] { 1.0f, 1.0f }, new[] { 2 });

            // Eager reference: RMSNorm normalizes the matmul output over the last dim.
            var eagerMm = gpu.TensorMatMul(a, b);
            var eager = gpu.RMSNorm(eagerMm, gamma, 1e-5, out _).Contiguous().AsSpan().ToArray();

            var scope = gpu.BeginDeferredScope();
            if (scope is null) { Skip.If(true, "Backend has no deferred execution"); return; }

            float[] deferred;
            using (scope)
            {
                var mm = gpu.TensorMatMul(a, b);             // deferred
                var r = gpu.RMSNorm(mm, gamma, 1e-5, out _); // reads the DEFERRED matmul output
                scope.Execute();                            // optimize + replay
                deferred = r.Contiguous().AsSpan().ToArray();
            }

            bool allZero = true;
            for (int i = 0; i < deferred.Length; i++) if (Math.Abs(deferred[i]) > 1e-9) { allZero = false; break; }
            Assert.False(allZero, "Deferred RMSNorm produced all-zero output — RmsNorm was not recorded (ran eager on an unfilled buffer).");

            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - deferred[i]) < 1e-3f,
                    $"[{i}] eager {eager[i]} vs deferred {deferred[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    private static (Tensor<float> cos, Tensor<float> sin) BuildRopeCache(int maxSeq, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[maxSeq * half];
        var sin = new float[maxSeq * half];
        for (int pos = 0; pos < maxSeq; pos++)
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(theta, (2.0 * i) / headDim);
                double angle = pos * freq;
                cos[pos * half + i] = (float)Math.Cos(angle);
                sin[pos * half + i] = (float)Math.Sin(angle);
            }
        return (new Tensor<float>(cos, new[] { maxSeq, half }), new Tensor<float>(sin, new[] { maxSeq, half }));
    }

    /// <summary>
    /// The decoder's actual attention op chain — interleaved RoPE on Q and K, then grouped-query attention with
    /// UNEXPANDED K/V — must replay from a deferred graph identically to eager. This is the end-to-end proof
    /// that the device-agnostic ApplyRoPEInterleaved + ScaledDotProductAttentionGqa are graph-recordable, so a
    /// decoder attention rewritten onto them (dropping managed RoPE + ExpandKVHeads) becomes fully recordable.
    /// </summary>
    [SkippableFact]
    public void DeferredScope_RoPEThenGqaAttention_MatchesEager()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            const int qHeads = 4, kvHeads = 2, seq = 3, headDim = 4, maxSeq = 8;
            float scale = 1f / (float)Math.Sqrt(headDim);
            var rng = new Random(31);
            float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
            var q = new Tensor<float>(Rand(qHeads * seq * headDim), new[] { 1, qHeads, seq, headDim });
            var k = new Tensor<float>(Rand(kvHeads * seq * headDim), new[] { 1, kvHeads, seq, headDim });
            var v = new Tensor<float>(Rand(kvHeads * seq * headDim), new[] { 1, kvHeads, seq, headDim });
            var (cos, sin) = BuildRopeCache(maxSeq, headDim, 10000f);

            var eng = (IEngine)gpu;

            // Eager reference on the GPU engine.
            var eRq = eng.ApplyRoPEInterleaved(q, cos, sin, 0);
            var eRk = eng.ApplyRoPEInterleaved(k, cos, sin, 0);
            var eager = eng.ScaledDotProductAttentionGqa(eRq, eRk, v, scale, isCausal: true)
                .Contiguous().AsSpan().ToArray();

            var scope = gpu.BeginDeferredScope();
            if (scope is null) { Skip.If(true, "Backend has no deferred execution"); return; }

            float[] deferred;
            using (scope)
            {
                var rq = eng.ApplyRoPEInterleaved(q, cos, sin, 0);
                var rk = eng.ApplyRoPEInterleaved(k, cos, sin, 0);
                var attn = eng.ScaledDotProductAttentionGqa(rq, rk, v, scale, isCausal: true);
                scope.Execute();
                deferred = attn.Contiguous().AsSpan().ToArray();
            }

            bool allZero = true;
            for (int i = 0; i < deferred.Length; i++) if (Math.Abs(deferred[i]) > 1e-9) { allZero = false; break; }
            Assert.False(allZero, "Deferred RoPE+GQA replay produced all-zero output — an op was not recorded.");

            Assert.Equal(eager.Length, deferred.Length);
            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - deferred[i]) < 1e-3f,
                    $"[{i}] eager {eager[i]} vs deferred {deferred[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}

#endif
