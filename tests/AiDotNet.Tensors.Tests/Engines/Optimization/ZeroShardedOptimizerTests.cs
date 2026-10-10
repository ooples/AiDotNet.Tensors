using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Distributed;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// ZeRO-1 over a process group: each rank updates only its shard with only its shard's state, group-wide statistics
/// are summed across ranks, and the updated shards are broadcast, so every rank ends each step with the parameters an
/// unsharded optimizer produces.
/// </summary>
public class ZeroShardedOptimizerTests
{
    public enum Kind { Adam, Sgd, DAdaptAdam, Prodigy }

    private static OptimizerBase Create(Kind kind) => kind switch
    {
        Kind.Adam => new AdamOptimizer(),
        Kind.Sgd => new SgdOptimizer(),
        Kind.DAdaptAdam => new DAdaptAdamOptimizer(),
        Kind.Prodigy => new ProdigyOptimizer(),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    private static readonly int[] Lengths = { 7, 12, 5, 9, 3 };

    private static float[] Values(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new float[length];
        for (int i = 0; i < length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static (OptimizerBase Optimizer, float[][] Parameters, float[][] Gradients) Build(Kind kind)
    {
        var optimizer = Create(kind);
        var group = optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = kind is Kind.DAdaptAdam or Kind.Prodigy ? 1.0 : 0.05, ["momentum"] = 0.9 });
        var parameters = Lengths.Select((n, i) => Values(n, i)).ToArray();
        var gradients = Lengths.Select(n => new float[n]).ToArray();
        for (int i = 0; i < Lengths.Length; i++) group.AddParameter(parameters[i], gradients[i]);
        return (optimizer, parameters, gradients);
    }

    private static void SetGradients(float[][] gradients, int step)
    {
        for (int i = 0; i < gradients.Length; i++) Array.Copy(Values(gradients[i].Length, 1000 + 10 * step + i), gradients[i], gradients[i].Length);
    }

    [Theory]
    [InlineData(Kind.Adam)]
    [InlineData(Kind.Sgd)]
    [InlineData(Kind.DAdaptAdam)]
    [InlineData(Kind.Prodigy)]
    public void EveryRank_EndsEachStepWithTheUnshardedParameters(Kind kind)
    {
        const int worldSize = 3, steps = 4;
        var (reference, referenceParameters, referenceGradients) = Build(kind);
        for (int step = 0; step < steps; step++)
        {
            SetGradients(referenceGradients, step);
            reference.Step();
        }

        var groups = InProcessGroup.Create(worldSize);
        var ranks = Enumerable.Range(0, worldSize).Select(_ => Build(kind)).ToArray();
        Parallel.For(0, worldSize, new ParallelOptions { MaxDegreeOfParallelism = worldSize }, rank =>
        {
            var sharded = new ZeroShardedOptimizer(ranks[rank].Optimizer, groups[rank]);
            for (int step = 0; step < steps; step++)
            {
                SetGradients(ranks[rank].Gradients, step);
                sharded.Step();
            }
        });

        for (int rank = 0; rank < worldSize; rank++)
        {
            for (int p = 0; p < Lengths.Length; p++)
                for (int i = 0; i < Lengths[p]; i++)
                {
                    float expected = referenceParameters[p][i], actual = ranks[rank].Parameters[p][i];
                    Assert.True(Math.Abs(expected - actual) <= 1e-5f * Math.Max(1f, Math.Abs(expected)),
                        $"{kind} rank {rank} parameter {p}[{i}]: unsharded {expected}, sharded {actual}");
                }

            // Each rank holds optimizer state for its own shard only (global ids rank, rank + 3, ...).
            var held = ranks[rank].Optimizer.StateDict().State.Keys.OrderBy(k => k).ToArray();
            var owned = Enumerable.Range(0, Lengths.Length).Where(id => id % worldSize == rank).ToArray();
            if (kind != Kind.Sgd || held.Length > 0) Assert.Equal(owned, held);
        }
    }

    [Fact]
    public void GroupStatisticsWithoutAProcessGroup_AreRefused()
    {
        var (optimizer, _, gradients) = Build(Kind.DAdaptAdam);
        SetGradients(gradients, 0);
        var sharded = new ZeroShardedOptimizer(optimizer, rank: 0, worldSize: 2);
        Assert.Throws<InvalidOperationException>(() => sharded.Step());
    }
}
