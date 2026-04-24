// Copyright (c) AiDotNet. All rights reserved.
// Phase E of issue #225: joint forward+backward graph + passes
// (DCE, min-cut partitioner, in-place candidate detection).

#nullable disable

using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Aot;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class JointGraphTests
{
    // ─── Graph construction ──────────────────────────────────────────

    [Fact]
    public void Graph_AppendForward_TracksForwardNodesInOrder()
    {
        var g = new JointGraph();
        int a = g.AppendForward("LoadA", new int[0], new[] { 4 }, isLeaf: true);
        int b = g.AppendForward("LoadB", new int[0], new[] { 4 }, isLeaf: true);
        int add = g.AppendForward("Add", new[] { a, b }, new[] { 4 });

        Assert.Equal(3, g.Count);
        Assert.Equal(new[] { a, b, add }, g.ForwardNodes);
        Assert.Empty(g.BackwardNodes);
        Assert.True(g.Nodes[a].IsLeaf);
        Assert.False(g.Nodes[add].IsLeaf);
    }

    [Fact]
    public void Graph_AppendBackward_SeparatesKinds()
    {
        var g = new JointGraph();
        int a = g.AppendForward("LoadA", new int[0], new[] { 4 }, isLeaf: true);
        int out_ = g.AppendForward("Relu", new[] { a }, new[] { 4 });
        int gOut = g.AppendBackward("OnesLike", new[] { out_ }, new[] { 4 });
        int gIn = g.AppendBackward("ReluBackward", new[] { a, gOut }, new[] { 4 });

        Assert.Equal(4, g.Count);
        Assert.Equal(new[] { a, out_ }, g.ForwardNodes);
        Assert.Equal(new[] { gOut, gIn }, g.BackwardNodes);
        Assert.Equal(JointNodeKind.Forward, g.Nodes[out_].Kind);
        Assert.Equal(JointNodeKind.Backward, g.Nodes[gIn].Kind);
    }

    [Fact]
    public void Graph_ElementCountAt_MultipliesShape()
    {
        var g = new JointGraph();
        int x = g.AppendForward("Load", new int[0], new[] { 3, 4, 5 }, isLeaf: true);
        Assert.Equal(60, g.ElementCountAt(x));
    }

    // ─── DCE ─────────────────────────────────────────────────────────

    [Fact]
    public void Dce_KeepsNodesReachableFromOutputs()
    {
        // Graph: a → add ; b → add (unused dead branch: c → dead_mul).
        var g = new JointGraph();
        int a = g.AppendForward("A", new int[0], new[] { 4 }, isLeaf: true);
        int b = g.AppendForward("B", new int[0], new[] { 4 }, isLeaf: true);
        int c = g.AppendForward("C", new int[0], new[] { 4 }, isLeaf: true);
        int add = g.AppendForward("Add", new[] { a, b }, new[] { 4 });
        int deadMul = g.AppendForward("DeadMul", new[] { c }, new[] { 4 });

        var live = JointGraphPasses.EliminateDeadCode(g, new[] { add });

        Assert.Contains(a, live);
        Assert.Contains(b, live);
        Assert.Contains(add, live);
        Assert.DoesNotContain(c, live);
        Assert.DoesNotContain(deadMul, live);
    }

    [Fact]
    public void Dce_BackwardOutputs_PullInForwardProducers()
    {
        // Forward: x → relu → y. Backward: grad_y → relu_backward → grad_x.
        var g = new JointGraph();
        int x = g.AppendForward("X", new int[0], new[] { 4 }, isLeaf: true);
        int y = g.AppendForward("Relu", new[] { x }, new[] { 4 });
        int gY = g.AppendBackward("OnesLike", new[] { y }, new[] { 4 });
        int gX = g.AppendBackward("ReluBackward", new[] { x, gY }, new[] { 4 });

        var live = JointGraphPasses.EliminateDeadCode(g, new[] { gX });
        Assert.Contains(x, live);
        Assert.Contains(y, live);
        Assert.Contains(gY, live);
        Assert.Contains(gX, live);
    }

    // ─── Min-cut partitioner ─────────────────────────────────────────

    [Fact]
    public void Partition_RetainsSmallActivations_WithinBudget()
    {
        // Three forward activations of varying sizes; backward
        // consumes all three. With a budget that fits only the two
        // smaller ones, the largest must be recomputed.
        var g = new JointGraph();
        int small = g.AppendForward("Small", new int[0], new[] { 100 }, isLeaf: true);
        int medium = g.AppendForward("Medium", new int[0], new[] { 1000 }, isLeaf: true);
        int large = g.AppendForward("Large", new int[0], new[] { 10000 }, isLeaf: true);
        g.AppendBackward("Sink", new[] { small, medium, large }, new[] { 1 });

        // Budget = 5000 elements → can hold small + medium (1100) but not large.
        // Set per-node cap high enough to allow medium (1000) but still cap at
        // 1250 so only small+medium fit.
        var decision = JointGraphPasses.PartitionActivations(g, memoryBudgetElements: 5000);

        Assert.Contains(small, decision.Retained);
        Assert.Contains(medium, decision.Retained);
        Assert.Contains(large, decision.Recomputed);
        Assert.True(decision.ElementsRetained <= 5000);
    }

    [Fact]
    public void Partition_IgnoresActivationsNotUsedByBackward()
    {
        // Forward-only dead branch should not count against budget.
        // Budget 600 → per-node cap 150, large enough for the 100-elem
        // used activation to be retained.
        var g = new JointGraph();
        int used = g.AppendForward("Used", new int[0], new[] { 100 }, isLeaf: true);
        int unused = g.AppendForward("Unused", new int[0], new[] { 10000 }, isLeaf: true);
        g.AppendBackward("Sink", new[] { used }, new[] { 1 });

        var decision = JointGraphPasses.PartitionActivations(g, memoryBudgetElements: 600);

        Assert.Contains(used, decision.Retained);
        Assert.DoesNotContain(unused, decision.Retained);
        Assert.DoesNotContain(unused, decision.Recomputed); // Not a candidate at all.
    }

    [Fact]
    public void Partition_PerNodeCap_RejectsOversizedActivations()
    {
        // Budget = 1000, so per-node cap = 250. A 500-element
        // activation must be recomputed even if no others compete.
        var g = new JointGraph();
        int tooBig = g.AppendForward("TooBig", new int[0], new[] { 500 }, isLeaf: true);
        g.AppendBackward("Sink", new[] { tooBig }, new[] { 1 });

        var decision = JointGraphPasses.PartitionActivations(g, memoryBudgetElements: 1000);

        Assert.Contains(tooBig, decision.Recomputed);
    }

    // ─── In-place candidate detection ────────────────────────────────

    [Fact]
    public void InPlace_SingleConsumer_IsCandidate()
    {
        var g = new JointGraph();
        int a = g.AppendForward("A", new int[0], new[] { 4 }, isLeaf: true);
        int relu = g.AppendForward("Relu", new[] { a }, new[] { 4 });
        g.AppendForward("Downstream", new[] { relu }, new[] { 4 });

        var candidates = JointGraphPasses.FindInPlaceCandidates(g, new int[0]);
        Assert.Contains(relu, candidates);
        Assert.DoesNotContain(a, candidates); // leaves excluded
    }

    [Fact]
    public void InPlace_MultipleConsumers_NotCandidate()
    {
        var g = new JointGraph();
        int a = g.AppendForward("A", new int[0], new[] { 4 }, isLeaf: true);
        int b = g.AppendForward("B", new[] { a }, new[] { 4 });
        // b has two consumers → can't in-place.
        g.AppendForward("C", new[] { b }, new[] { 4 });
        g.AppendForward("D", new[] { b }, new[] { 4 });

        var candidates = JointGraphPasses.FindInPlaceCandidates(g, new int[0]);
        Assert.DoesNotContain(b, candidates);
    }

    [Fact]
    public void InPlace_RetainedActivation_NotCandidate()
    {
        var g = new JointGraph();
        int a = g.AppendForward("A", new int[0], new[] { 4 }, isLeaf: true);
        int retained = g.AppendForward("Retained", new[] { a }, new[] { 4 });
        g.AppendBackward("UsesRetained", new[] { retained }, new[] { 4 });

        var candidates = JointGraphPasses.FindInPlaceCandidates(g, new[] { retained });
        Assert.DoesNotContain(retained, candidates);
    }
}
