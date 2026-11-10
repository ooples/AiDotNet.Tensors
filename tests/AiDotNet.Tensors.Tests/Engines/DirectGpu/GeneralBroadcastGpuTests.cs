// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Broadcast shapes outside the engine's last-axis / first-axis fast paths go through the rank-generic path. When one
/// operand already has the output shape, that path now broadcasts the other over its leading size-1 axes with the
/// last-axis kernels and tiles it only over the trailing axes, instead of tiling it to a full output-sized copy
/// (e.g. a [1, F, 1] spectral filter times [N, F, 2]). These cases pin all four ops to the CPU engine for that route,
/// for a small operand given first, and for the two-sided / interior-axis broadcasts that still tile.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GeneralBroadcastGpuTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public GeneralBroadcastGpuTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed, bool positive = false)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = positive ? (float)(0.5 + rng.NextDouble()) : (float)(rng.NextDouble() - 0.5);
        return t;
    }

    public static IEnumerable<object[]> Shapes() =>
    [
        [new[] { 1, 33, 1 }, new[] { 7, 33, 2 }],     // spectral filter: middle axis kept, outer + inner broadcast
        [new[] { 7, 33, 2 }, new[] { 1, 33, 1 }],
        [new[] { 1, 33, 1 }, new[] { 1, 1, 2 }],      // two-sided: both operands stretched
        [new[] { 5 }, new[] { 4, 3, 5 }],             // small operand first, rank mismatch
        [new[] { 3, 1 }, new[] { 2, 3, 6 }],          // rank mismatch, inner broadcast
        [new[] { 2, 1, 4, 1 }, new[] { 1, 3, 1, 5 }], // alternating axes on both sides
        [new[] { 1 }, new[] { 3, 4 }],                // scalar-like operand first
        [new[] { 6, 4, 3 }, new[] { 1, 1, 1 }],       // every axis broadcast (inner extent 1)
    ];

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void GeneralBroadcast_MatchesCpu_ForAllOps(int[] shapeA, int[] shapeB)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        var cpu = new CpuEngine();
        var a = Rand(shapeA, 1);
        var b = Rand(shapeB, 2, positive: true);
        var ops = new (string Name, Func<CpuEngine, Tensor<float>> Run)[]
        {
            ("add", e => e.TensorBroadcastAdd(a, b)),
            ("subtract", e => e.TensorBroadcastSubtract(a, b)),
            ("multiply", e => e.TensorBroadcastMultiply(a, b)),
            ("divide", e => e.TensorBroadcastDivide(a, b)),
        };
        foreach (var (name, run) in ops)
        {
            var expected = run(cpu);
            var actual = run(gpu);
            Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
            var e = expected.ToArray();
            var g = actual.ToArray();
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - g[i]) <= 1e-5f * Math.Max(1f, Math.Abs(e[i])),
                    $"{name} {string.Join("x", shapeA)} op {string.Join("x", shapeB)} [{i}]: cpu {e[i]}, gpu {g[i]}");
        }
    }
}
