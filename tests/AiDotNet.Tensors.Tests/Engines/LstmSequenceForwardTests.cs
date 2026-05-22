using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness + perf tests for <see cref="CpuEngine.LstmSequenceForward{T}"/> —
/// the fused-LSTM-sequence primitive added for the AIsEval P0 inference gap
/// (issue #436). Correctness is checked against a decomposed reference impl
/// (matmul + sigmoid + tanh per timestep) on small shapes; the perf test
/// exercises the AIsEval workload shape [B=128, seq=32, in=32, hidden=64].
/// </summary>
public class LstmSequenceForwardTests
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine;

    public LstmSequenceForwardTests(ITestOutputHelper output)
    {
        _output = output;
        _engine = new CpuEngine();
    }

    [Fact]
    public void LstmSequenceForward_MatchesDecomposedReference_NoSequencesReturn()
    {
        // Small shape so the decomposed reference runs fast.
        const int batch = 2, seq = 4, inFeatures = 3, hidden = 5;
        var rng = new Random(42);

        var input = MakeRandom(rng, batch, seq, inFeatures);
        var wIh   = MakeRandom(rng, 4 * hidden, inFeatures);
        var wHh   = MakeRandom(rng, 4 * hidden, hidden);
        var bIh   = MakeRandom1D(rng, 4 * hidden);
        var bHh   = MakeRandom1D(rng, 4 * hidden);

        var fused = _engine.LstmSequenceForward(input, h0: null, c0: null, wIh, wHh, bIh, bHh, returnSequences: false);
        var reference = DecomposedReference(input, wIh, wHh, bIh, bHh, returnSequences: false);

        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void LstmSequenceForward_MatchesDecomposedReference_ReturnSequences()
    {
        const int batch = 2, seq = 4, inFeatures = 3, hidden = 5;
        var rng = new Random(123);

        var input = MakeRandom(rng, batch, seq, inFeatures);
        var wIh   = MakeRandom(rng, 4 * hidden, inFeatures);
        var wHh   = MakeRandom(rng, 4 * hidden, hidden);

        var fused = _engine.LstmSequenceForward(input, h0: null, c0: null, wIh, wHh, bIh: null, bHh: null, returnSequences: true);
        var reference = DecomposedReference(input, wIh, wHh, null, null, returnSequences: true);

        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void LstmSequenceForward_AcceptsInitialHiddenAndCellStates()
    {
        // h0/c0 path: feed non-zero initial states and verify the reference
        // implementation also picks them up identically.
        const int batch = 3, seq = 2, inFeatures = 4, hidden = 6;
        var rng = new Random(7);

        var input = MakeRandom(rng, batch, seq, inFeatures);
        var wIh   = MakeRandom(rng, 4 * hidden, inFeatures);
        var wHh   = MakeRandom(rng, 4 * hidden, hidden);
        var h0    = MakeRandom(rng, batch, hidden);
        var c0    = MakeRandom(rng, batch, hidden);

        var fused = _engine.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh: null, bHh: null, returnSequences: false);
        var reference = DecomposedReference(input, wIh, wHh, null, null, returnSequences: false, h0: h0, c0: c0);

        AssertClose(fused, reference, atol: 1e-4f);
    }

    [Fact]
    public void LstmSequenceForward_RejectsBadShapes()
    {
        var input = Tensor<float>.CreateZeros(2, 4, 3);
        // wIh first dim must be divisible by 4.
        var badWih = Tensor<float>.CreateZeros(5, 3);
        var goodWhh = Tensor<float>.CreateZeros(4, 1);
        Assert.Throws<ArgumentException>(() =>
            _engine.LstmSequenceForward(input, h0: null, c0: null, badWih, goodWhh, bIh: null, bHh: null));

        // wHh hidden dim mismatch.
        var goodWih = Tensor<float>.CreateZeros(8, 3);
        var badWhh = Tensor<float>.CreateZeros(8, 5);
        Assert.Throws<ArgumentException>(() =>
            _engine.LstmSequenceForward(input, h0: null, c0: null, goodWih, badWhh, bIh: null, bHh: null));
    }

    [Fact]
    public void LstmSequenceForward_AisevalShape_FinishesWithinBudget()
    {
        // AIsEval LSTM inference shape: [B=128, seq=32, in=32, hidden=64].
        // PyTorch nn.LSTM does this in ~12 ms at bs=128 steady-state latency.
        // The decomposed-per-step implementation in AiDotNet (LSTMLayer.Forward)
        // did not finish a single inference iteration in 3+ minutes on the
        // reference rig. Budget here is 100 ms (within ~10x PyTorch) — generous
        // for CI noise, tight enough to lock in that the fused primitive
        // actually finishes, which the per-step path didn't.
        const int batch = 128, seq = 32, inFeatures = 32, hidden = 64;
        var rng = new Random(2026);

        var input = MakeRandom(rng, batch, seq, inFeatures);
        var wIh   = MakeRandom(rng, 4 * hidden, inFeatures);
        var wHh   = MakeRandom(rng, 4 * hidden, hidden);

        // Warmup.
        _ = _engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null, returnSequences: false);

        var sw = Stopwatch.StartNew();
        const int iters = 5;
        for (int i = 0; i < iters; i++)
            _ = _engine.LstmSequenceForward(input, null, null, wIh, wHh, null, null, returnSequences: false);
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;

        _output.WriteLine($"LstmSequenceForward AIsEval shape [128,32,32->64]: {ms:F2} ms/iter (budget: 100 ms)");
        Assert.True(ms < 100.0,
            $"LstmSequenceForward took {ms:F2} ms — exceeds 100 ms budget. This indicates the " +
            "fused primitive has regressed enough that we are back to the AIsEval LSTM-doesn't-finish state.");
    }

    // ----------------- Helpers -----------------

    private static Tensor<float> MakeRandom(Random rng, params int[] shape)
    {
        var t = Tensor<float>.CreateZeros(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2 - 1); // [-1, 1]
        return t;
    }

    private static Tensor<float> MakeRandom1D(Random rng, int n) => MakeRandom(rng, n);

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
    /// Decomposed reference impl of an LSTM sequence — matmul + sigmoid + tanh
    /// per timestep with no fusion. Used only as the ground truth for the
    /// correctness tests. Not a perf target.
    /// </summary>
    private Tensor<float> DecomposedReference(
        Tensor<float> input,
        Tensor<float> wIh, Tensor<float> wHh,
        Tensor<float>? bIh, Tensor<float>? bHh,
        bool returnSequences,
        Tensor<float>? h0 = null, Tensor<float>? c0 = null)
    {
        int batch = input.Shape[0], seq = input.Shape[1], inFeatures = input.Shape[2];
        int gateRows = wIh.Shape[0], hidden = gateRows / 4;

        // Materialize the inputs as raw arrays for direct scalar arithmetic.
        var inputArr = input.AsSpan().ToArray();
        var wIhArr = wIh.AsSpan().ToArray();
        var wHhArr = wHh.AsSpan().ToArray();
        var bIhArr = bIh?.AsSpan().ToArray();
        var bHhArr = bHh?.AsSpan().ToArray();

        var h = h0 is null ? new float[batch * hidden] : h0.AsSpan().ToArray();
        var c = c0 is null ? new float[batch * hidden] : c0.AsSpan().ToArray();
        var hNew = new float[batch * hidden];
        var cNew = new float[batch * hidden];

        var outShape = returnSequences ? new[] { batch, seq, hidden } : new[] { batch, hidden };
        var output = Tensor<float>.CreateZeros(outShape);
        var outSpan = output.AsWritableSpan();

        for (int t = 0; t < seq; t++)
        {
            for (int b = 0; b < batch; b++)
            {
                // Compute the 4 gates for sample b at timestep t.
                for (int gi = 0; gi < 4; gi++)
                {
                    for (int j = 0; j < hidden; j++)
                    {
                        int gateRow = gi * hidden + j;
                        // Wx contribution: dot(wIh[gateRow, :], input[b, t, :]).
                        float gate = 0f;
                        for (int k = 0; k < inFeatures; k++)
                            gate += wIhArr[gateRow * inFeatures + k] * inputArr[(b * seq + t) * inFeatures + k];
                        if (bIhArr is not null) gate += bIhArr[gateRow];
                        // Wh contribution: dot(wHh[gateRow, :], h[b, :]).
                        for (int k = 0; k < hidden; k++)
                            gate += wHhArr[gateRow * hidden + k] * h[b * hidden + k];
                        if (bHhArr is not null) gate += bHhArr[gateRow];

                        // Stash into the gates buffer for this sample (use cNew/hNew as scratch indexed by gate).
                        // We'll dispatch after computing all 4 gates for this (b, j).
                        switch (gi)
                        {
                            case 0: cNew[b * hidden + j] = gate; break; // i
                            case 1: hNew[b * hidden + j] = gate; break; // f
                            case 2: c[b * hidden + j]    = c[b * hidden + j]; cNew[b * hidden + j] = gate; break; // g overwrites scratch path differently
                            case 3: break; // o handled below
                        }
                    }
                }
                // Re-walk: easier to compute gates separately than reuse scratch — recompute cleanly.
                for (int j = 0; j < hidden; j++)
                {
                    float gi_ = 0f, gf_ = 0f, gg_ = 0f, go_ = 0f;
                    for (int k = 0; k < inFeatures; k++)
                    {
                        float x = inputArr[(b * seq + t) * inFeatures + k];
                        gi_ += wIhArr[(0 * hidden + j) * inFeatures + k] * x;
                        gf_ += wIhArr[(1 * hidden + j) * inFeatures + k] * x;
                        gg_ += wIhArr[(2 * hidden + j) * inFeatures + k] * x;
                        go_ += wIhArr[(3 * hidden + j) * inFeatures + k] * x;
                    }
                    if (bIhArr is not null)
                    {
                        gi_ += bIhArr[0 * hidden + j];
                        gf_ += bIhArr[1 * hidden + j];
                        gg_ += bIhArr[2 * hidden + j];
                        go_ += bIhArr[3 * hidden + j];
                    }
                    for (int k = 0; k < hidden; k++)
                    {
                        float hk = h[b * hidden + k];
                        gi_ += wHhArr[(0 * hidden + j) * hidden + k] * hk;
                        gf_ += wHhArr[(1 * hidden + j) * hidden + k] * hk;
                        gg_ += wHhArr[(2 * hidden + j) * hidden + k] * hk;
                        go_ += wHhArr[(3 * hidden + j) * hidden + k] * hk;
                    }
                    if (bHhArr is not null)
                    {
                        gi_ += bHhArr[0 * hidden + j];
                        gf_ += bHhArr[1 * hidden + j];
                        gg_ += bHhArr[2 * hidden + j];
                        go_ += bHhArr[3 * hidden + j];
                    }
                    float i_ = Sigmoid(gi_);
                    float f_ = Sigmoid(gf_);
                    float g_ = MathF.Tanh(gg_);
                    float o_ = Sigmoid(go_);
                    float cNewVal = f_ * c[b * hidden + j] + i_ * g_;
                    float hNewVal = o_ * MathF.Tanh(cNewVal);
                    cNew[b * hidden + j] = cNewVal;
                    hNew[b * hidden + j] = hNewVal;
                }
            }
            // Commit.
            Array.Copy(hNew, h, h.Length);
            Array.Copy(cNew, c, c.Length);

            if (returnSequences)
            {
                for (int b = 0; b < batch; b++)
                {
                    int srcOff = b * hidden;
                    int dstOff = (b * seq + t) * hidden;
                    for (int j = 0; j < hidden; j++)
                        outSpan[dstOff + j] = h[srcOff + j];
                }
            }
        }

        if (!returnSequences)
            for (int i = 0; i < h.Length; i++) outSpan[i] = h[i];

        return output;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
