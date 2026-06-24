using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tests for <see cref="CpuEngine.MultiHeadAttentionForward{T}"/> — the fused
/// multi-head attention primitive added for the AIsEval Transformer
/// inference gap (issue #436). Verifies the wrapper produces the same output
/// as the explicit Q/K/V + reshape + SDPA + output-projection chain.
/// </summary>
public class MultiHeadAttentionForwardTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine;

    public MultiHeadAttentionForwardTests(ITestOutputHelper output)
    {
        _output = output;
        _engine = new CpuEngine();
    }

    [Fact]
    public void MultiHeadAttentionForward_AisevalShape_PerfMeasurement()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        // AIsEval Transformer per-layer attention shape:
        // [B=128, seq=32, dModel=64], numHeads=4, dHead=16.
        // PyTorch nn.TransformerEncoderLayer @ bs=128 was 13.85 ms steady-state
        // for the whole encoder layer (attention + FFN). We're benchmarking
        // just MHA here, which should be ~half of that.
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var rng = new Random(2026);
        var input = MakeRandom(rng, batch, seq, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        for (int w = 0; w < 8; w++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        double best = double.MaxValue;
        for (int r = 0; r < 15; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
            sw.Stop();
            best = System.Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        _output.WriteLine($"MHA AIsEval [128,32,64] h=4 best-of-15: {best:F2} ms (PyTorch ~2.74 ms)");
    }

    [Fact]
    public void MultiHeadAttentionForward_ZeroAllocSdpa_AllocationBelowThreshold()
    {
        // #476: the zero-alloc SDPA path must not allocate the per-call [B,H,Sq,Sk]
        // attention-weights tensor or a separate SDPA output tensor. At the AIsEval
        // shape the old path allocated ~7.3 MB/call (dominated by the SDPA
        // scores/softmax + output tensors); the fix does in-place softmax in pooled
        // scratch and writes P·V straight into the pooled MHA buffer.
        const int batch = 128, seq = 32, dModel = 64, numHeads = 4;
        var rng = new Random(2026);
        var input = MakeRandom(rng, batch, seq, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        // Warm up: JIT, fill ArrayPool buckets, prime AutoTensorCache.
        for (int w = 0; w < 12; w++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

#if NET5_0_OR_GREATER
        const int iters = 20;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < iters; i++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        long after = GC.GetAllocatedBytesForCurrentThread();
        double perCallKb = (after - before) / (double)iters / 1024.0;
        _output.WriteLine($"MHA per-call allocation (steady state): {perCallKb:F1} KB " +
                          $"(old SDPA scores+softmax+output path was ~7.3 MB/call)");

        // Old path ≈ 7.3 MB/call. After the fix the only sizeable per-call allocation
        // is the returned [B,seq,dModel] result tensor (~1 MB here); the ~3 MB SDPA
        // weights+output scratch is gone. 2 MB is a comfortable bar — far below the
        // old profile, above the inherent result allocation, robust to pool/GC noise.
        Assert.True(perCallKb < 2048.0,
            $"MHA allocated {perCallKb:F1} KB/call — expected < 2048 KB (zero-alloc SDPA regressed?).");
#else
        // GC.GetAllocatedBytesForCurrentThread is net5+; the allocation assertion runs
        // on net10.0. On net471 just exercise the path (correctness is covered by the
        // decomposed-chain tests).
        _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
#endif
    }

    [Fact]
    public void MultiHeadAttentionForward_MatchesDecomposedChain()
    {
        // Small shape that runs the decomposed reference quickly.
        const int batch = 2, seq = 4, dModel = 8, numHeads = 2;
        int dHead = dModel / numHeads;
        var rng = new Random(42);

        var input    = MakeRandom(rng, batch, seq, dModel);
        var qW       = MakeRandom(rng, dModel, dModel);
        var kW       = MakeRandom(rng, dModel, dModel);
        var vW       = MakeRandom(rng, dModel, dModel);
        var oW       = MakeRandom(rng, dModel, dModel);

        var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        var reference = DecomposedChain(input, qW, kW, vW, oW, numHeads);

        Assert.Equal(3, fused.Shape.Length);
        Assert.Equal(batch, fused.Shape[0]);
        Assert.Equal(seq, fused.Shape[1]);
        Assert.Equal(dModel, fused.Shape[2]);

        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void MultiHeadAttentionForward_WithCausalMask_MatchesDecomposedChain()
    {
        const int batch = 2, seq = 4, dModel = 8, numHeads = 2;
        int dHead = dModel / numHeads;
        var rng = new Random(7);

        var input = MakeRandom(rng, batch, seq, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        // Causal mask [B, H, seq, seq]: query i may attend to key j <= i.
        var mask = new Tensor<bool>(new[] { batch, numHeads, seq, seq });
        var maskSpan = mask.AsWritableSpan();
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int i = 0; i < seq; i++)
                    for (int j = 0; j < seq; j++)
                        maskSpan[((b * numHeads + h) * seq + i) * seq + j] = j <= i;

        var fusedMasked = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, mask);
        var referenceMasked = DecomposedChain(input, qW, kW, vW, oW, numHeads, mask);

        Assert.Equal(new[] { batch, seq, dModel }, fusedMasked.Shape.ToArray());
        AssertClose(fusedMasked, referenceMasked, atol: 1e-4f);

        // A causal mask must change the result vs unmasked (full) attention.
        var fusedUnmasked = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        bool differs = false;
        var a = fusedMasked.AsSpan();
        var u = fusedUnmasked.AsSpan();
        for (int i = 0; i < a.Length; i++)
            if (Math.Abs(a[i] - u[i]) > 1e-4f) { differs = true; break; }
        Assert.True(differs, "causal mask should change the attention output vs unmasked");
    }

    [Fact]
    public void MultiHeadAttentionForward_BroadcastMask_MatchesFullyMaterializedMask()
    {
        // #674 (CodeRabbit): the IEngine contract allows masks BROADCASTABLE to [batch, heads, seq, seq]
        // — e.g. a shared causal mask [1, 1, seq, seq]. Direct mask[b, h, i, j] indexing threw
        // IndexOutOfRangeException (or read wrong data) for b>0/h>0. A [1, 1, seq, seq] mask must be
        // applied to EVERY (batch, head) and produce the same result as the fully-materialized mask.
        // seq=5 (not a multiple of 8) forces the scalar softmax path that consults the mask.
        const int batch = 3, seq = 5, dModel = 8, numHeads = 2;
        var rng = new Random(11);

        var input = MakeRandom(rng, batch, seq, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        // Shared causal mask broadcast over batch + heads: [1, 1, seq, seq].
        var shared = new Tensor<bool>(new[] { 1, 1, seq, seq });
        var sharedSpan = shared.AsWritableSpan();
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                sharedSpan[i * seq + j] = j <= i;

        // The same pattern, fully materialized to [batch, heads, seq, seq] — the known-good reference.
        var full = new Tensor<bool>(new[] { batch, numHeads, seq, seq });
        var fullSpan = full.AsWritableSpan();
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int i = 0; i < seq; i++)
                    for (int j = 0; j < seq; j++)
                        fullSpan[((b * numHeads + h) * seq + i) * seq + j] = j <= i;

        // (a) must not throw on b>0/h>0, and (b) must equal the fully-materialized mask's result.
        var broadcastResult = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, shared);
        var fullResult = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, full);

        Assert.Equal(new[] { batch, seq, dModel }, broadcastResult.Shape.ToArray());
        AssertClose(broadcastResult, fullResult, atol: 1e-4f);
    }

    [Fact]
    public void MultiHeadAttentionForward_RejectsBadShapes()
    {
        var input = Tensor<float>.CreateZeros(2, 4, 8);
        var goodW = Tensor<float>.CreateZeros(8, 8);
        var badW = Tensor<float>.CreateZeros(7, 8);

        // numHeads must divide dModel.
        Assert.Throws<ArgumentException>(() =>
            _engine.MultiHeadAttentionForward(input, goodW, goodW, goodW, goodW, numHeads: 3));

        // Wrong weight shape.
        Assert.Throws<ArgumentException>(() =>
            _engine.MultiHeadAttentionForward(input, badW, goodW, goodW, goodW, numHeads: 2));

        // numHeads must be positive.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _engine.MultiHeadAttentionForward(input, goodW, goodW, goodW, goodW, numHeads: 0));
    }

    // ===================== Issue #468 regression =====================
    // Before the fix, MultiHeadAttentionForwardFloat wrapped its ArrayPool-rented
    // scratch (qHead/kHead/vHead) with the exact logical shape. ArrayPool returns a
    // BUCKET-sized array (rounded up to a power of 2), and the Tensor ctor validates
    // data.Length == product(shape) exactly — so it threw "The number of values does
    // not match the specified shape" whenever batch*heads*seq*dHead was not a power
    // of 2, i.e. every non-power-of-2 sequence length. The fix wraps EXACTLY
    // product(shape) elements as a zero-copy Memory view. These tests prove the fused
    // primitive is numerically correct (vs the decomposed chain) across power-of-2
    // AND non-power-of-2 lengths, multiple batch/head/dModel shapes, single-token
    // and masked edge cases — not merely that it stops throwing.

    [Theory]
    [InlineData(1)]    // degenerate single token
    [InlineData(2)]    // power of 2  (worked pre-fix)
    [InlineData(3)]    // NOT pow2    (threw pre-fix)
    [InlineData(5)]    // NOT pow2    (exact issue #468 repro length)
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]    // power of 2
    [InlineData(12)]
    [InlineData(16)]   // power of 2
    [InlineData(31)]
    [InlineData(32)]   // power of 2
    [InlineData(100)]  // common non-pow2 transformer length
    [InlineData(128)]  // power of 2
    public void MultiHeadAttentionForward_AnySeqLen_MatchesDecomposedChain(int seqLen)
    {
        const int batch = 2, dModel = 16, numHeads = 4;
        var rng = new Random(1000 + seqLen);
        var input = MakeRandom(rng, batch, seqLen, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        var reference = DecomposedChain(input, qW, kW, vW, oW, numHeads);

        Assert.Equal(new[] { batch, seqLen, dModel }, fused.Shape.ToArray());
        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void MultiHeadAttentionForward_IssueRepro_SeqLen5_DoesNotThrowAndIsCorrect()
    {
        // The exact shape from issue #468: batch=1, seqLen=5 (not pow2), embed=32, heads=4.
        const int batch = 1, seqLen = 5, embed = 32, numHeads = 4;
        var rng = new Random(468);
        var input = MakeRandom(rng, batch, seqLen, embed);
        var qW = MakeRandom(rng, embed, embed);
        var kW = MakeRandom(rng, embed, embed);
        var vW = MakeRandom(rng, embed, embed);
        var oW = MakeRandom(rng, embed, embed);

        // Pre-fix: ArgumentException "The number of values does not match the specified shape".
        var ex = Record.Exception(() => _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads));
        Assert.Null(ex);

        var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        AssertClose(fused, DecomposedChain(input, qW, kW, vW, oW, numHeads), atol: 1e-4f);
    }

    [Theory]
    // (batch, seqLen, dModel, numHeads) — each chosen so batch*heads*seqLen*dHead is
    // NOT a power of 2, and spanning batch>1, several head counts and dModel sizes.
    [InlineData(3, 5, 12, 3)]    // qkvElems = 3·3·5·4   = 180
    [InlineData(2, 7, 24, 4)]    //           2·4·7·6   = 336
    [InlineData(1, 100, 32, 4)]  //           1·4·100·8 = 3200
    [InlineData(4, 13, 48, 6)]   //           4·6·13·8  = 2496
    [InlineData(5, 3, 8, 2)]     //           5·2·3·4   = 120
    [InlineData(2, 6, 6, 3)]     //           2·3·6·2   = 72
    [InlineData(3, 9, 30, 5)]    //           3·5·9·6   = 810
    public void MultiHeadAttentionForward_VariousNonPow2Shapes_MatchesDecomposedChain(
        int batch, int seqLen, int dModel, int numHeads)
    {
        var rng = new Random(batch * 1000 + seqLen * 31 + dModel * 7 + numHeads);
        var input = MakeRandom(rng, batch, seqLen, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        var reference = DecomposedChain(input, qW, kW, vW, oW, numHeads);

        Assert.Equal(new[] { batch, seqLen, dModel }, fused.Shape.ToArray());
        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Theory]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(12)]
    [InlineData(31)]
    public void MultiHeadAttentionForward_NonPow2SeqLen_WithCausalMask_MatchesDecomposedChain(int seqLen)
    {
        const int batch = 2, dModel = 16, numHeads = 4;
        var rng = new Random(2000 + seqLen);
        var input = MakeRandom(rng, batch, seqLen, dModel);
        var qW = MakeRandom(rng, dModel, dModel);
        var kW = MakeRandom(rng, dModel, dModel);
        var vW = MakeRandom(rng, dModel, dModel);
        var oW = MakeRandom(rng, dModel, dModel);

        // Causal mask [B, H, seq, seq]: query i attends to key j <= i.
        var mask = new Tensor<bool>(new[] { batch, numHeads, seqLen, seqLen });
        var maskSpan = mask.AsWritableSpan();
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        maskSpan[((b * numHeads + h) * seqLen + i) * seqLen + j] = j <= i;

        var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, mask);
        var reference = DecomposedChain(input, qW, kW, vW, oW, numHeads, mask);

        Assert.Equal(new[] { batch, seqLen, dModel }, fused.Shape.ToArray());
        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void MultiHeadAttentionForward_NonPow2SeqLen_AcrossManyLengths_NeverThrows()
    {
        // Sweep every length 1..40 (most are non-pow2) to lock the pattern: the
        // pre-fix failure was "OK iff seqLen is a power of 2", so a contiguous sweep
        // is the strongest guard against the bucket-size regression reappearing.
        const int batch = 2, dModel = 16, numHeads = 4;
        for (int seqLen = 1; seqLen <= 40; seqLen++)
        {
            var rng = new Random(7000 + seqLen);
            var input = MakeRandom(rng, batch, seqLen, dModel);
            var qW = MakeRandom(rng, dModel, dModel);
            var kW = MakeRandom(rng, dModel, dModel);
            var vW = MakeRandom(rng, dModel, dModel);
            var oW = MakeRandom(rng, dModel, dModel);

            var ex = Record.Exception(() =>
            {
                var fused = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
                Assert.Equal(new[] { batch, seqLen, dModel }, fused.Shape.ToArray());
            });
            Assert.True(ex is null, $"seqLen={seqLen} threw: {ex?.GetType().Name}: {ex?.Message}");
        }
    }

    // ----------------- Helpers -----------------

    private static Tensor<float> MakeRandom(Random rng, params int[] shape)
    {
        var t = Tensor<float>.CreateZeros(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 0.4 - 0.2); // [-0.2, 0.2] to keep softmax well-conditioned
        return t;
    }

    private static void AssertClose(Tensor<float> a, Tensor<float> b, float atol)
    {
        Assert.Equal(a.Shape.Length, b.Shape.Length);
        for (int d = 0; d < a.Shape.Length; d++)
            Assert.Equal(a.Shape[d], b.Shape[d]);

        var sa = a.AsSpan();
        var sb = b.AsSpan();
        Assert.Equal(sa.Length, sb.Length);
        for (int i = 0; i < sa.Length; i++)
        {
            float diff = MathF.Abs(sa[i] - sb[i]);
            Assert.True(diff < atol,
                $"Mismatch at index {i}: fused={sa[i]:G6}, ref={sb[i]:G6}, diff={diff:G3} (atol={atol:G3}).");
        }
    }

    /// <summary>
    /// Decomposed reference implementation. Mirrors the chain documented in
    /// MultiHeadAttentionForward's XML doc so the test is self-evidently the
    /// same computation.
    /// </summary>
    private Tensor<float> DecomposedChain(
        Tensor<float> input, Tensor<float> qW, Tensor<float> kW, Tensor<float> vW, Tensor<float> oW, int numHeads)
        => DecomposedChain(input, qW, kW, vW, oW, numHeads, mask: null);

    private Tensor<float> DecomposedChain(
        Tensor<float> input, Tensor<float> qW, Tensor<float> kW, Tensor<float> vW, Tensor<float> oW, int numHeads,
        Tensor<bool>? mask)
    {
        int batch = input.Shape[0], seq = input.Shape[1], dModel = input.Shape[2];
        int dHead = dModel / numHeads;

        var flat = input.Reshape(new[] { batch * seq, dModel });
        var q = _engine.TensorMatMul(flat, qW);
        var k = _engine.TensorMatMul(flat, kW);
        var v = _engine.TensorMatMul(flat, vW);

        var qH = q.Reshape(new[] { batch, seq, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var kH = k.Reshape(new[] { batch, seq, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });
        var vH = v.Reshape(new[] { batch, seq, numHeads, dHead }).Transpose(new[] { 0, 2, 1, 3 });

        var attn = _engine.ScaledDotProductAttention<float>(qH, kH, vH, mask: mask, scale: null, out _);
        var concat = attn.Transpose(new[] { 0, 2, 1, 3 }).Reshape(new[] { batch, seq, dModel });
        var concatFlat = concat.Reshape(new[] { batch * seq, dModel });
        var outFlat = _engine.TensorMatMul(concatFlat, oW);
        return outFlat.Reshape(new[] { batch, seq, dModel });
    }
}
