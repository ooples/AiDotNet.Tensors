// #1624 prototype: liveness-based pooling of the compiled training plan's
// backward gradient buffers. These tests pin the planner's correctness invariant
// (no two tensors with overlapping gradient lifetimes may share a physical
// buffer) and quantify the resident-set reduction on a deep feed-forward stack —
// the shape that OOMs the SimCSE #1624 repro by holding every layer's
// intermediate gradient buffer resident at once.

using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class BackwardGradientBufferPlannerTests
{
    // Recompute each tensor's [firstWrite, lastRead] backward interval from the
    // step list — the independent spec the planner must honour. Mirrors the
    // planner's own definition (a step's backward reads grad(output) and
    // accumulates grad(input)), in backward position units (n-1-s).
    private static (int[] first, int[] last) Intervals(
        int tensorCount, IReadOnlyList<BackwardGradientBufferPlanner.GradStep> steps)
    {
        int n = steps.Count;
        var first = new int[tensorCount];
        var last = new int[tensorCount];
        for (int i = 0; i < tensorCount; i++) { first[i] = int.MaxValue; last[i] = -1; }
        for (int s = 0; s < n; s++)
        {
            int bpos = n - 1 - s;
            var step = steps[s];
            if (step.OutputId >= 0 && bpos > last[step.OutputId]) last[step.OutputId] = bpos;
            foreach (var inp in step.InputIds)
            {
                if (inp < 0) continue;
                if (bpos < first[inp]) first[inp] = bpos;
                if (bpos > last[inp]) last[inp] = bpos;
            }
        }
        return (first, last);
    }

    /// <summary>
    /// Builds an N-layer chain: out[i] = f(out[i-1], param[i]). out[0] is the leaf
    /// input; every activation has the same shape, so the planner is free to share
    /// buffers across the (disjoint-lifetime) layer activations.
    /// </summary>
    private static (int tensorCount, int[] elem, long[] shape,
        List<BackwardGradientBufferPlanner.GradStep> steps, bool[] persistent)
        BuildChain(int layers, int elemPerTensor)
    {
        // ids: 0 = input out0; activations out1..outL = 1..L; params p1..pL = L+1..2L
        int L = layers;
        int tensorCount = 1 + L + L;
        var steps = new List<BackwardGradientBufferPlanner.GradStep>(L);
        for (int i = 1; i <= L; i++)
        {
            int outId = i;
            int prevAct = i - 1;     // out[i-1]
            int paramId = L + i;     // p[i]
            steps.Add(new BackwardGradientBufferPlanner.GradStep(outId, new[] { prevAct, paramId }));
        }
        var elem = new int[tensorCount];
        var shape = new long[tensorCount];
        for (int i = 0; i < tensorCount; i++) { elem[i] = elemPerTensor; shape[i] = 1; }
        var persistent = new bool[tensorCount];
        for (int i = 1; i <= L; i++) persistent[L + i] = true; // params persistent
        persistent[L] = true;                                   // loss output (last activation)
        return (tensorCount, elem, shape, steps, persistent);
    }

    [Fact]
    public void SharedBuffers_NeverHaveOverlappingLifetimes()
    {
        var (tc, elem, shape, steps, persistent) = BuildChain(layers: 12, elemPerTensor: 1000);
        var plan = BackwardGradientBufferPlanner.Plan(tc, elem, shape, steps, persistent, elementBytes: 4);
        var (first, last) = Intervals(tc, steps);

        // For every pair of tensors assigned the SAME physical buffer, their
        // [first,last] intervals must be disjoint — else the backward would
        // overwrite a gradient still in use (silent wrong-gradient corruption).
        for (int a = 0; a < tc; a++)
        {
            if (first[a] == int.MaxValue) continue; // no interval
            for (int b = a + 1; b < tc; b++)
            {
                if (first[b] == int.MaxValue) continue;
                if (plan.BufferOfTensor[a] != plan.BufferOfTensor[b]) continue;
                bool disjoint = last[a] < first[b] || last[b] < first[a];
                Assert.True(disjoint,
                    $"tensors {a} [{first[a]},{last[a]}] and {b} [{first[b]},{last[b]}] " +
                    $"share buffer {plan.BufferOfTensor[a]} but their lifetimes overlap");
            }
        }
    }

    [Fact]
    public void DeepChain_PoolsActivationsToBoundedFrontier()
    {
        // A deep chain's activation gradients are each live across ~one backward
        // step, so the pooled buffer set must be a small constant frontier — NOT
        // grow with depth the way the naive one-buffer-per-tensor scheme does.
        var (tc, elem, shape, steps, persistent) = BuildChain(layers: 50, elemPerTensor: 1000);
        var plan = BackwardGradientBufferPlanner.Plan(tc, elem, shape, steps, persistent, elementBytes: 4);

        // 50 params + loss are persistent (dedicated); the ~50 activations should
        // collapse onto a handful of shared buffers. Total physical buffers must be
        // far below the naive tensor count.
        Assert.True(plan.PhysicalBufferCount < tc,
            $"no pooling happened: {plan.PhysicalBufferCount} buffers for {tc} tensors");
        // The non-persistent activations (~51) should pool down to a single-digit
        // frontier; assert the pooled set saved at least 60% of those buffers.
        int persistentCount = 0;
        foreach (var p in persistent) if (p) persistentCount++;
        int naiveIntermediates = tc - persistentCount;
        int pooledIntermediates = plan.PhysicalBufferCount - persistentCount;
        Assert.True(pooledIntermediates <= naiveIntermediates * 0.4,
            $"weak pooling: {pooledIntermediates} pooled vs {naiveIntermediates} naive intermediates");
        Assert.True(plan.PooledBytes < plan.NaiveBytes);
    }

    [Fact]
    public void Parameters_AlwaysGetDedicatedBuffers()
    {
        var (tc, elem, shape, steps, persistent) = BuildChain(layers: 8, elemPerTensor: 500);
        var plan = BackwardGradientBufferPlanner.Plan(tc, elem, shape, steps, persistent, elementBytes: 4);

        // Each persistent tensor must own a buffer no other tensor shares — its
        // gradient is read by the optimizer AFTER the backward walk ends.
        var bufferUsers = new Dictionary<int, int>();
        for (int t = 0; t < tc; t++)
        {
            int buf = plan.BufferOfTensor[t];
            bufferUsers.TryGetValue(buf, out int c);
            bufferUsers[buf] = c + 1;
        }
        for (int t = 0; t < tc; t++)
            if (persistent[t])
                Assert.Equal(1, bufferUsers[plan.BufferOfTensor[t]]);
    }
}
