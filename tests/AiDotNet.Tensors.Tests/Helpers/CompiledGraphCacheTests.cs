using AiDotNet.Tensors.Helpers;
using Xunit;
using static AiDotNet.Tensors.Helpers.ComputationGraph;

namespace AiDotNet.Tensors.Tests.Helpers;

public class CompiledGraphCacheTests
{
    private static ComputationGraph BuildSimpleGraph(int[] inputShape)
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int input = graph.RecordInput(inputShape);
        int conv = graph.RecordOp(OpType.Conv2D, [input], [1, 64, 32, 32]);
        int relu = graph.RecordOp(OpType.ReLU, [conv], [1, 64, 32, 32]);
        graph.RecordOutput(relu);
        graph.EndCapture();
        return graph;
    }

    [Fact]
    public void GetOrCompile_FirstCall_CompileAndCaches()
    {
        var cache = new CompiledGraphCache();
        var graph = BuildSimpleGraph([1, 3, 64, 64]);

        var plan = cache.GetOrCompile(graph, [[1, 3, 64, 64]]);

        Assert.NotNull(plan);
        Assert.Equal(1, cache.CachedGraphCount);
        Assert.Equal(1, cache.Misses);
    }

    [Fact]
    public void GetOrCompile_SameGraph_ReturnsFromCache()
    {
        var cache = new CompiledGraphCache();
        var graph1 = BuildSimpleGraph([1, 3, 64, 64]);
        var graph2 = BuildSimpleGraph([1, 3, 64, 64]);

        var plan1 = cache.GetOrCompile(graph1, [[1, 3, 64, 64]]);
        var plan2 = cache.GetOrCompile(graph2, [[1, 3, 64, 64]]);

        Assert.Same(plan1, plan2); // exact same cached object
        Assert.Equal(1, cache.Hits);
        Assert.Equal(1, cache.Misses);
    }

    [Fact]
    public void GetOrCompile_DifferentInputShape_CompilesSeparately()
    {
        var cache = new CompiledGraphCache();
        var graph1 = BuildSimpleGraph([1, 3, 64, 64]);
        var graph2 = BuildSimpleGraph([4, 3, 64, 64]); // different batch size

        cache.GetOrCompile(graph1, [[1, 3, 64, 64]]);
        cache.GetOrCompile(graph2, [[4, 3, 64, 64]]);

        Assert.Equal(2, cache.CachedGraphCount);
        Assert.Equal(2, cache.Misses);
    }

    [Fact]
    public void TryGet_MissReturnsNull()
    {
        var cache = new CompiledGraphCache();
        var graph = BuildSimpleGraph([1, 3, 64, 64]);

        var plan = cache.TryGet(graph, [[1, 3, 64, 64]]);

        Assert.Null(plan);
        Assert.Equal(1, cache.Misses);
    }

    [Fact]
    public void TryGet_HitAfterCompile()
    {
        var cache = new CompiledGraphCache();
        var graph = BuildSimpleGraph([1, 3, 64, 64]);

        cache.GetOrCompile(graph, [[1, 3, 64, 64]]);

        var graph2 = BuildSimpleGraph([1, 3, 64, 64]);
        var plan = cache.TryGet(graph2, [[1, 3, 64, 64]]);

        Assert.NotNull(plan);
        Assert.Equal(1, cache.Hits);
    }

    [Fact]
    public void Clear_RemovesAll()
    {
        var cache = new CompiledGraphCache();
        cache.GetOrCompile(BuildSimpleGraph([1, 3, 64, 64]), [[1, 3, 64, 64]]);
        cache.GetOrCompile(BuildSimpleGraph([4, 3, 64, 64]), [[4, 3, 64, 64]]);

        Assert.Equal(2, cache.CachedGraphCount);

        cache.Clear();
        Assert.Equal(0, cache.CachedGraphCount);
    }

    [Fact]
    public void EvictOlderThan_RemovesStaleEntries()
    {
        var cache = new CompiledGraphCache();
        cache.GetOrCompile(BuildSimpleGraph([1, 3, 64, 64]), [[1, 3, 64, 64]]);

        // Evict entries older than 1 second from now (should evict the one we just added)
        int evicted = cache.EvictOlderThan(DateTime.UtcNow.AddSeconds(1));
        Assert.Equal(1, evicted);
        Assert.Equal(0, cache.CachedGraphCount);
    }

    [Fact]
    public void HitRatio_ComputedCorrectly()
    {
        var cache = new CompiledGraphCache();
        var graph = BuildSimpleGraph([1, 3, 64, 64]);

        // 1 miss
        cache.GetOrCompile(graph, [[1, 3, 64, 64]]);
        Assert.Equal(0.0, cache.HitRatio);

        // 1 hit
        var graph2 = BuildSimpleGraph([1, 3, 64, 64]);
        cache.GetOrCompile(graph2, [[1, 3, 64, 64]]);
        Assert.Equal(0.5, cache.HitRatio);
    }
}
