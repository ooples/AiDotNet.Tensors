// Copyright (c) AiDotNet. All rights reserved.
// GlobalMeanPoolBackwardGpu had NO test at all, which is why it shipped two distinct bugs:
//   1. TileBatch(grad, out, reduceSize, 1) wrote only reduceSize elements into a buffer sized
//      outerSize*reduceSize and then Scaled all of it — scaling UNINITIALISED memory for outerSize > 1.
//   2. The first fix, TileBatch(grad, out, reduceSize, outerSize), fills the buffer but performs a
//      BLOCKED repeat (out[idx] = grad[idx/reduceSize]). That is correct only when the trailing
//      (inner) extent is 1; for [2,3,4] it places grad[0] where grad[1] belongs.
// The op reduces CONTIGUOUS MIDDLE axes, so the correct broadcast is interleaved:
//      gradInput[(o*R + r)*I + i] = gradOutput[o*I + i] / R
// This test asserts that directly, with I > 1 and o > 1 — the case both bugs get wrong.
#if !NETFRAMEWORK
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

[Collection("OpParity")]
public sealed class GlobalMeanPoolBackwardGpuTests
{
    private readonly OpParityFixture _fx;
    public GlobalMeanPoolBackwardGpuTests(OpParityFixture fx) => _fx = fx;

    [SkippableTheory]
    [InlineData(new[] { 2, 3, 4 })]      // rank 3: axes [1]    -> outer 2, R 3, inner 4
    [InlineData(new[] { 2, 3, 4, 5 })]   // rank 4: axes [1,2]  -> outer 2, R 12, inner 5
    [InlineData(new[] { 3, 7, 1 })]      // inner == 1: the degenerate case the blocked repeat got right
    public void BroadcastsGradientOverReducedAxes(int[] inputShape)
    {
        Skip.If(!_fx.GpuReady, "No DirectGpu backend available.");
        var gpu = _fx.Gpu!;

        int rank = inputShape.Length;
        int[] axes = rank == 4 ? new[] { 1, 2 } : new[] { 1 };
        int reduceSize = axes.Aggregate(1, (a, d) => a * inputShape[d]);
        int outerCount = 1;
        for (int d = 0; d < axes[0]; d++) outerCount *= inputShape[d];
        int innerCount = 1;
        for (int d = axes[^1] + 1; d < rank; d++) innerCount *= inputShape[d];

        int gradLen = outerCount * innerCount;
        var gradData = new float[gradLen];
        for (int i = 0; i < gradLen; i++) gradData[i] = i + 1;   // distinct values expose misrouting
        // GlobalMeanPoolBackwardGpu reads gradOutput.Buffer directly, so the gradient must already be
        // GPU-resident — a host tensor throws "Tensor is not GPU-resident".
        var grad = new Tensor<float>(gradData, new[] { outerCount, innerCount }).Gpu();

        using var result = ((DirectGpuTensorEngine)gpu).GlobalMeanPoolBackwardGpu(grad, inputShape);
        var actual = result.ToArray();

        int total = inputShape.Aggregate(1, (a, b) => a * b);
        Assert.Equal(total, actual.Length);

        for (int o = 0; o < outerCount; o++)
        for (int r = 0; r < reduceSize; r++)
        for (int i = 0; i < innerCount; i++)
        {
            int idx = (o * reduceSize + r) * innerCount + i;
            float expected = gradData[o * innerCount + i] / reduceSize;
            Assert.True(Math.Abs(actual[idx] - expected) <= 1e-5f,
                $"shape [{string.Join(",", inputShape)}] at (o={o}, r={r}, i={i}) idx={idx}: " +
                $"expected {expected}, got {actual[idx]}. A blocked repeat gives grad[idx/R]; " +
                "uninitialised memory gives garbage past the first reduceSize elements.");
        }
    }
}
#endif
