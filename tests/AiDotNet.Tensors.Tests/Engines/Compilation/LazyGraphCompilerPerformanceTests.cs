// Copyright (c) AiDotNet. All rights reserved.

using System.Diagnostics;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("BlasManaged-Perf-Serial")]
public sealed class LazyGraphCompilerPerformanceTests
{
    [Fact]
    [Trait("Category", "Performance")]
    public async Task Compile_WideFanInGraph_StaysBelowQuadraticRuntime()
    {
        await Task.Yield();

        const int producerCount = 50_000;
        var producers = new StubLazyNode[producerCount];
        var rawNodes = new List<ILazyNode>(producerCount + 1);
        for (int i = 0; i < producers.Length; i++)
        {
            producers[i] = new StubLazyNode();
            rawNodes.Add(producers[i]);
        }

        var sink = new StubLazyNode(producers);
        rawNodes.Add(sink);

        var stopwatch = Stopwatch.StartNew();
        var compiled = new LazyGraphCompiler().Compile(rawNodes);
        stopwatch.Stop();

        Assert.Equal(producerCount + 1, compiled.Count);
        Assert.Same(sink, compiled[^1]);
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(5),
            $"Compiling a {producerCount:N0}-producer fan-in graph took " +
            $"{stopwatch.Elapsed.TotalSeconds:F2}s; the ready scheduler regressed toward O(V^2).");
    }

    private sealed class StubLazyNode : ILazyNode
    {
        private static readonly IEngine StubEngine = new CpuEngine();
        private readonly ILazyNode[] _inputs;

        public StubLazyNode(params ILazyNode[] inputs) => _inputs = inputs;

        public LazyNodeType OpType => LazyNodeType.Negate;
        public int[] OutputShape { get; } = [1];
        public bool IsRealized { get; set; }
        public int TopologicalIndex { get; set; }
        public int ConsumerCount { get; set; }
        public IEngine RecordingEngine => StubEngine;

        public void Realize(IEngine engine) => IsRealized = true;
        public ILazyNode[] GetInputNodes() => _inputs;
        public void ClearOutputLazySource() { }
        public void AddStorageLeases(TensorStorageLeaseSet leases) { }
    }
}
