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

    [Fact(Skip = "Performance measurement — run manually with --filter MultiHeadAttentionForward_AisevalShape")]
    public void MultiHeadAttentionForward_AisevalShape_PerfMeasurement()
    {
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

        for (int w = 0; w < 3; w++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        const int iters = 10;
        for (int i = 0; i < iters; i++)
            _ = _engine.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"MultiHeadAttentionForward AIsEval shape [128,32,64] h=4: {ms:F2} ms/iter");
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

        Assert.Equal(new[] { batch, seq, dModel }, fusedMasked.Shape);
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
