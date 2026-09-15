// Regression tests for MatMulBackward's generic fallback failing to reduce a rank-3
// weight gradient back to the rank-2 weight leaf.
//
// A non-contiguous narrow (Tensor.Slice on the last axis) fails the IsContiguous gate on
// every fast path in MatMulBackward -- the selective-linear route, the rank-3 collapsed
// route, and the 2D fast path all require contiguity -- so it lands in the generic
// fallback. That fallback computes `TransposeLastTwoDims(inputs[0]) x gradOutput`, which
// for a rank-3 input and a rank-2 weight produces a rank-3 [1, K, N] gradient for a
// [K, N] leaf and accumulates it unreduced. FusedLinearBackwardCore's equivalent fallback
// already applies SumToShape for exactly this reason (issue #234); MatMulBackward did not.
//
// Symptom when the same weight also receives a correctly-shaped rank-2 contribution:
//   ArgumentException: Tensor shapes must match. Got [80, 80] and [1, 80, 80].

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

[Collection("EngineCurrentGlobalState")]
public sealed class MatMulBackwardRankReductionTests : IDisposable
{
    private const int Rows = 513;
    private const int K = 80;
    private const int N = 80;

    private readonly IEngine _priorEngine;
    private readonly CpuEngine _engine;

    public MatMulBackwardRankReductionTests()
    {
        _priorEngine = AiDotNetEngine.Current;
        _engine = new CpuEngine();
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    /// <summary>
    /// The reported repro. Two rank-3 non-contiguous narrows and one rank-2 activation all
    /// feed the SAME rank-2 weight. Backward runs in reverse-topological order, so the
    /// last-recorded (rank-2) contribution accumulates first and the unreduced rank-3
    /// gradient is added to it second -- which is why the thrown message reads
    /// "Got [80, 80] and [1, 80, 80]" in that operand order.
    /// </summary>
    [Fact]
    public void NonContiguousNarrow_SharedRank2Weight_AccumulatesWithoutShapeMismatch()
    {
        var (wide, flat, weight) = MakeInputs(seed: 7);

        using var tape = new GradientTape<float>();

        var head = wide.Slice(2, 0, K);          // non-contiguous [1, Rows, K]
        var tail = wide.Slice(2, K, K * 2);      // non-contiguous [1, Rows, K]
        Assert.False(head.IsContiguous);          // the defect requires the non-contiguous route
        Assert.False(tail.IsContiguous);

        var y1 = _engine.TensorMatMul(head, weight);
        var y2 = _engine.TensorMatMul(tail, weight);
        var y3 = _engine.TensorMatMul(flat, weight);   // recorded LAST => accumulates FIRST

        var loss = _engine.TensorAdd(
            _engine.TensorAdd(_engine.ReduceSum(y1), _engine.ReduceSum(y2)),
            _engine.ReduceSum(y3));

        // Before the fix this throws:
        //   ArgumentException: Tensor shapes must match. Got [80, 80] and [1, 80, 80].
        var grads = tape.ComputeGradients(loss, new[] { weight });

        Assert.True(grads.ContainsKey(weight));
        Assert.Equal(2, grads[weight].Rank);
        Assert.Equal(K, grads[weight]._shape[0]);
        Assert.Equal(N, grads[weight]._shape[1]);
    }

    /// <summary>
    /// The invariant on its own, with no second contribution to collide with: a rank-2
    /// weight leaf must never receive a higher-rank gradient. Before the fix this returns
    /// [1, 80, 80].
    /// </summary>
    [Fact]
    public void NonContiguousNarrow_RankThreeMatMul_ProducesRank2WeightGradient()
    {
        var (wide, _, weight) = MakeInputs(seed: 11);

        using var tape = new GradientTape<float>();
        var tail = wide.Slice(2, K, K * 2);
        Assert.False(tail.IsContiguous);

        var y = _engine.TensorMatMul(tail, weight);
        var loss = _engine.ReduceSum(y);
        var grads = tape.ComputeGradients(loss, new[] { weight });

        Assert.Equal(2, grads[weight].Rank);
        Assert.Equal(K, grads[weight]._shape[0]);
        Assert.Equal(N, grads[weight]._shape[1]);
    }

    /// <summary>
    /// Shape alone is not enough: the reduced gradient must also hold the right VALUES.
    /// Compares the non-contiguous-narrow route against a materialised contiguous copy of
    /// the same sub-tensor, which takes a different (already-correct) route through
    /// MatMulBackward.
    /// </summary>
    [Fact]
    public void NonContiguousNarrow_WeightGradient_MatchesContiguousEquivalent()
    {
        var (wide, _, weight) = MakeInputs(seed: 23);

        float[] narrowGrad;
        using (var tape = new GradientTape<float>())
        {
            var tail = wide.Slice(2, K, K * 2);
            var loss = _engine.ReduceSum(_engine.TensorMatMul(tail, weight));
            narrowGrad = ToArray(tape.ComputeGradients(loss, new[] { weight })[weight]);
        }

        // Materialise the same [1, Rows, K] block contiguously: wide is [1, Rows, 2K]
        // row-major, so element (0, r, K + j) lives at r * 2K + K + j.
        var wideData = ToArray(wide);
        var contiguousData = new float[Rows * K];
        for (int r = 0; r < Rows; r++)
            for (int j = 0; j < K; j++)
                contiguousData[r * K + j] = wideData[r * (K * 2) + K + j];
        var contiguous = new Tensor<float>(contiguousData, new[] { 1, Rows, K });

        float[] referenceGrad;
        using (var tape = new GradientTape<float>())
        {
            var loss = _engine.ReduceSum(_engine.TensorMatMul(contiguous, weight));
            referenceGrad = ToArray(tape.ComputeGradients(loss, new[] { weight })[weight]);
        }

        Assert.Equal(referenceGrad.Length, narrowGrad.Length);
        for (int i = 0; i < referenceGrad.Length; i++)
        {
            double expected = referenceGrad[i];
            double actual = narrowGrad[i];
            double tolerance = 1e-3 * Math.Max(1.0, Math.Abs(expected));
            Assert.True(
                Math.Abs(expected - actual) <= tolerance,
                $"weight gradient mismatch at {i}: contiguous={expected:G9} narrow={actual:G9}");
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static (Tensor<float> wide, Tensor<float> flat, Tensor<float> weight) MakeInputs(int seed)
    {
        var rng = new Random(seed);
        var wideData = new float[Rows * K * 2];
        for (int i = 0; i < wideData.Length; i++) wideData[i] = (float)(rng.NextDouble() * 2 - 1);
        var flatData = new float[Rows * K];
        for (int i = 0; i < flatData.Length; i++) flatData[i] = (float)(rng.NextDouble() * 2 - 1);
        var weightData = new float[K * N];
        for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1);

        return (
            new Tensor<float>(wideData, new[] { 1, Rows, K * 2 }),
            new Tensor<float>(flatData, new[] { Rows, K }),
            new Tensor<float>(weightData, new[] { K, N }));
    }

    private static float[] ToArray(Tensor<float> tensor)
    {
        var copy = new float[tensor.Length];
        tensor.AsSpan().CopyTo(copy);
        return copy;
    }
}
