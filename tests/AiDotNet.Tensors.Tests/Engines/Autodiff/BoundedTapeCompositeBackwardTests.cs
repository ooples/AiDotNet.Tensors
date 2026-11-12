using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A create-graph backward differentiates some ops (GLU here) through a composite of recorded ops walked on the
/// running tape. On a bounded tape (<see cref="GradientTapeOptions.MaxEntries"/>) with no room left, the composite's
/// ops were dropped, yet the composite reported success: the op's first-order kernel was skipped and its input got
/// no gradient at all. A full tape must still give every input its exact gradient.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public sealed class BoundedTapeCompositeBackwardTests : IDisposable
{
    private readonly IEngine _previous;
    private readonly CpuEngine _e = new();

    public BoundedTapeCompositeBackwardTests()
    {
        _previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _e;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    private static Tensor<double> Input()
    {
        var x = new Tensor<double>(new[] { 3, 6 });
        var r = new Random(7);
        for (int i = 0; i < x.Length; i++) x[i] = r.NextDouble() * 2 - 1;
        return x;
    }

    private Tensor<double> Forward(Tensor<double> x) => _e.ReduceSum(_e.GLU(x, -1), null);

    private (Tensor<double> Grad, int Entries) Reference(Tensor<double> x)
    {
        using var tape = new GradientTape<double>();
        var loss = Forward(x);
        int entries = tape.EntryCount;
        var g = tape.ComputeGradients(loss, new[] { x });
        Assert.True(g.TryGetValue(x, out var grad) && grad is not null);
        return (grad!, entries);
    }

    // spare = 0: the tape is full before the backward starts. spare = 1, 2: the composite starts recording and the tape
    // fills partway through it, so what it did record has to be taken back.
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void FullTape_CreateGraphBackward_StillGivesEveryInputItsGradient(int spare)
    {
        var x = Input();
        var (expected, forwardEntries) = Reference(x);

        using var tape = new GradientTape<double>(new GradientTapeOptions { MaxEntries = forwardEntries + spare });
        var loss = Forward(x);
        Assert.Equal(forwardEntries, tape.EntryCount);

        var grads = tape.ComputeGradients(loss, new[] { x }, createGraph: true);

        Assert.True(grads.TryGetValue(x, out var grad) && grad is not null,
            $"spare {spare}: the input got no gradient from a create-graph backward on a full tape");
        Assert.Equal(expected._shape, grad!._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], grad[i], 12);
        Assert.True(tape.EntryCount <= forwardEntries + spare);
    }

    // With room to spare the composite path still runs: the gradient is a node of the graph and differentiates again.
    [Fact]
    public void UnboundedTape_CreateGraphBackward_KeepsTheCompositePath()
    {
        var x = Input();
        var (expected, _) = Reference(x);
        using var tape = new GradientTape<double>();
        var loss = Forward(x);
        int before = tape.EntryCount;
        var grads = tape.ComputeGradients(loss, new[] { x }, createGraph: true);
        Assert.True(grads.TryGetValue(x, out var grad) && grad is not null);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], grad![i], 12);
        Assert.True(tape.EntryCount > before, "the create-graph backward recorded nothing");
    }
}
