using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The fused float LSTM (tape node) runs its cell and BPTT rows with System.Numerics.Vector over the hidden axis
/// and a scalar tail. The finite-difference tests use hidden sizes below one vector, so they only reach the tail;
/// these use hidden sizes that need both the vector body and a tail (and an exact multiple), and check the fused
/// output and every gradient against the decomposed differentiable graph (the state-returning overload records
/// ordinary primitives), which shares no code with the fused rows.
/// </summary>
public class LstmFusedVectorizedCellTests
{
    private static Tensor<float> Rand(int[] shape, System.Random rng, float scale = 0.5f)
    {
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    [Theory]
    [InlineData(19, true)]    // 2 vectors of 8 + a 3-wide tail (AVX2), or 4 of 4 + 3 (SSE)
    [InlineData(19, false)]
    [InlineData(32, true)]    // exact multiple: vector body only
    [InlineData(64, false)]   // the parity model's hidden size
    public void FusedRows_MatchDecomposedGraph(int hidden, bool returnSequences)
    {
        var eng = new CpuEngine();
        var rng = new System.Random(hidden * 31 + (returnSequences ? 1 : 0));
        int batch = 3, seq = 4, inF = 5, G = 4 * hidden;
        var input = Rand(new[] { batch, seq, inF }, rng);
        var wIh = Rand(new[] { G, inF }, rng);
        var wHh = Rand(new[] { G, hidden }, rng);
        var bIh = Rand(new[] { G }, rng);
        var bHh = Rand(new[] { G }, rng);
        var h0 = Rand(new[] { batch, hidden }, rng);
        var c0 = Rand(new[] { batch, hidden }, rng);
        var coeff = Rand(returnSequences ? new[] { batch, seq, hidden } : new[] { batch, hidden }, rng);
        var sources = new[] { input, wIh, wHh, bIh, bHh, h0, c0 };

        Tensor<float> fusedOut;
        Dictionary<Tensor<float>, Tensor<float>> fused;
        using (var tape = new GradientTape<float>())
        {
            fusedOut = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
            fused = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(fusedOut, coeff), null), sources);
        }

        Tensor<float> refOut;
        Dictionary<Tensor<float>, Tensor<float>> reference;
        using (var tape = new GradientTape<float>())
        {
            refOut = eng.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, out _, out _, returnSequences);
            reference = tape.ComputeGradients(eng.ReduceSum(eng.TensorMultiply(refOut, coeff), null), sources);
        }

        AssertClose(refOut.ToArray(), fusedOut.ToArray(), "output");
        string[] names = { "input", "wIh", "wHh", "bIh", "bHh", "h0", "c0" };
        for (int i = 0; i < sources.Length; i++)
            AssertClose(reference[sources[i]].ToArray(), fused[sources[i]].ToArray(), names[i]);
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (System.Math.Abs(e - a) > 2e-5f * (1f + System.Math.Abs(e)))
                Assert.Fail($"{what}: element {i} decomposed={e:R} fused={a:R}");
        }
    }
}