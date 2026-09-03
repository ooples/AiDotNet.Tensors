using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class OperationReorderingPassTests
{
    [Fact]
    public void ReorderOperations_PreservesEveryNodeAndDependency()
    {
        var first = new TestNode("first");
        var second = new TestNode("second");
        var join = new TestNode("join", first, second);
        var tail = new TestNode("tail", join);
        var independent = new TestNode("independent");
        var input = new List<ILazyNode> { tail, second, independent, join, first };

        List<ILazyNode> result = LazyGraphCompiler.ReorderOperations(input);

        Assert.Equal(input.Count, result.Count);
        Assert.Equal(input.Count, result.Distinct().Count());
        AssertDependencyPrecedes(result, first, join);
        AssertDependencyPrecedes(result, second, join);
        AssertDependencyPrecedes(result, join, tail);
    }

    [Fact]
    public void ReorderOperations_PrefersProducerWhoseConsumerIsCloserToReady()
    {
        var near = new TestNode("near");
        var far = new TestNode("far");
        var otherFarInput = new TestNode("other-far-input");
        var nearConsumer = new TestNode("near-consumer", near);
        var farConsumer = new TestNode("far-consumer", far, otherFarInput);
        var input = new List<ILazyNode> { far, near, otherFarInput, farConsumer, nearConsumer };

        List<ILazyNode> result = LazyGraphCompiler.ReorderOperations(input);

        Assert.True(result.IndexOf(near) < result.IndexOf(far));
        AssertDependencyPrecedes(result, near, nearConsumer);
        AssertDependencyPrecedes(result, far, farConsumer);
        AssertDependencyPrecedes(result, otherFarInput, farConsumer);
    }

    [Fact]
    public void ReorderOperations_MatchesHistoricalScheduleAcrossGeneratedGraphs()
    {
        for (int seed = 0; seed < 100; seed++)
        {
            var random = new Random(seed);
            var nodes = new List<TestNode>();
            for (int i = 0; i < 250; i++)
            {
                int inputCount = i == 0 ? 0 : random.Next(0, Math.Min(i, 5) + 1);
                var inputs = new HashSet<TestNode>();
                while (inputs.Count < inputCount)
                    inputs.Add(nodes[random.Next(i)]);

                nodes.Add(new TestNode($"node-{i}", inputs.ToArray()));
            }

            var shuffled = nodes.OrderBy(_ => random.Next()).Cast<ILazyNode>().ToList();

            List<ILazyNode> expected = ReorderOperationsReference(shuffled);
            List<ILazyNode> actual = LazyGraphCompiler.ReorderOperations(shuffled);

            Assert.Equal(expected, actual);
        }
    }

    [Fact(Timeout = 30_000)]
    public async Task ReorderOperations_LargeReadySetCompletesWithoutQuadraticScan()
    {
        await Task.Yield();
        const int nodeCount = 20_000;
        var input = new List<ILazyNode>(nodeCount);
        for (int i = 0; i < nodeCount; i++)
            input.Add(new TestNode($"node-{i}"));

        List<ILazyNode> result = LazyGraphCompiler.ReorderOperations(input);

        Assert.Equal(nodeCount, result.Count);
        Assert.Equal(nodeCount, result.Distinct().Count());
    }

    private static List<ILazyNode> ReorderOperationsReference(List<ILazyNode> nodes)
    {
        var nodeSet = new HashSet<ILazyNode>(nodes);
        var inDegree = nodes.ToDictionary(node => node, _ => 0);
        var dependents = nodes.ToDictionary(node => node, _ => new List<ILazyNode>());

        foreach (ILazyNode node in nodes)
        {
            foreach (ILazyNode input in node.GetInputNodes())
            {
                if (!nodeSet.Contains(input))
                    continue;

                inDegree[node]++;
                dependents[input].Add(node);
            }
        }

        var ready = nodes.Where(node => inDegree[node] == 0).ToList();
        var result = new List<ILazyNode>(nodes.Count);
        while (ready.Count > 0)
        {
            int bestIndex = 0;
            int bestScore = int.MaxValue;
            for (int i = 0; i < ready.Count; i++)
            {
                List<ILazyNode> readyDependents = dependents[ready[i]];
                int score = readyDependents.Count > 0 ? inDegree[readyDependents[0]] : int.MaxValue;
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            ILazyNode chosen = ready[bestIndex];
            ready[bestIndex] = ready[^1];
            ready.RemoveAt(ready.Count - 1);
            result.Add(chosen);

            foreach (ILazyNode dependent in dependents[chosen])
            {
                inDegree[dependent]--;
                if (inDegree[dependent] == 0)
                    ready.Add(dependent);
            }
        }

        return result;
    }

    private static void AssertDependencyPrecedes(
        List<ILazyNode> result,
        ILazyNode dependency,
        ILazyNode consumer)
        => Assert.True(result.IndexOf(dependency) < result.IndexOf(consumer));

    private sealed class TestNode : ILazyNode
    {
        private readonly ILazyNode[] _inputs;

        internal TestNode(string name, params ILazyNode[] inputs)
        {
            Name = name;
            _inputs = inputs;
        }

        internal string Name { get; }
        public LazyNodeType OpType => LazyNodeType.Custom;
        public int[] OutputShape { get; } = [1];
        public bool IsRealized { get; set; }
        public int TopologicalIndex { get; set; }
        public int ConsumerCount { get; set; }
        public IEngine RecordingEngine => null!;
        public void Realize(IEngine engine) { }
        public ILazyNode[] GetInputNodes() => _inputs;
        public void ClearOutputLazySource() { }
        public void AddStorageLeases(TensorStorageLeaseSet leases) { }
        public override string ToString() => Name;
    }
}
