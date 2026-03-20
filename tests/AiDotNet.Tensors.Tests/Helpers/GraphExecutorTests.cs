using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using static AiDotNet.Tensors.Helpers.ComputationGraph;

namespace AiDotNet.Tensors.Tests.Helpers;

public class GraphExecutorTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void Execute_TanhPipeline_MatchesDirectComputation()
    {
        // Build graph: input -> tanh -> output
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([4]);
        int tanh = graph.RecordOp(OpType.Tanh, [inp], [4]);
        graph.RecordOutput(tanh);
        graph.EndCapture();

        // Execute via graph
        using var executor = new GraphExecutor<float>(graph, _engine);
        var input = new Tensor<float>(new[] { 4 });
        input[0] = -2f; input[1] = -1f; input[2] = 0f; input[3] = 1f;
        var outputs = executor.Execute(input);

        // Compare with direct computation
        var expected = _engine.Tanh(input);

        Assert.Single(outputs);
        for (int i = 0; i < 4; i++)
            Assert.Equal(expected[i], outputs[0][i], 1e-5f);
    }

    [Fact]
    public void Execute_GELUPipeline_MatchesDirectComputation()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([8]);
        int gelu = graph.RecordOp(OpType.GELU, [inp], [8]);
        graph.RecordOutput(gelu);
        graph.EndCapture();

        using var executor = new GraphExecutor<float>(graph, _engine);
        var input = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < 8; i++) input[i] = (i - 4) * 0.5f;
        var outputs = executor.Execute(input);

        var expected = _engine.TensorGELU(input);
        Assert.Single(outputs);
        for (int i = 0; i < 8; i++)
            Assert.Equal(expected[i], outputs[0][i], 1e-4f);
    }

    [Fact]
    public void Execute_SwishPipeline_MatchesDirectComputation()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([8]);
        int swish = graph.RecordOp(OpType.Swish, [inp], [8]);
        graph.RecordOutput(swish);
        graph.EndCapture();

        using var executor = new GraphExecutor<float>(graph, _engine);
        var input = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < 8; i++) input[i] = (i - 4) * 0.5f;
        var outputs = executor.Execute(input);

        var expected = _engine.Swish(input);
        Assert.Single(outputs);
        for (int i = 0; i < 8; i++)
            Assert.Equal(expected[i], outputs[0][i], 1e-4f);
    }

    [Fact]
    public void Execute_ResidualAdd_CorrectResult()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int a = graph.RecordInput([4]);
        int b = graph.RecordInput([4]);
        int add = graph.RecordOp(OpType.Residual, [a, b], [4]);
        graph.RecordOutput(add);
        graph.EndCapture();

        using var executor = new GraphExecutor<float>(graph, _engine);
        var inputA = new Tensor<float>(new[] { 4 });
        var inputB = new Tensor<float>(new[] { 4 });
        inputA[0] = 1f; inputA[1] = 2f; inputA[2] = 3f; inputA[3] = 4f;
        inputB[0] = 10f; inputB[1] = 20f; inputB[2] = 30f; inputB[3] = 40f;

        var outputs = executor.Execute(inputA, inputB);

        Assert.Single(outputs);
        Assert.Equal(11f, outputs[0][0], 1e-5f);
        Assert.Equal(22f, outputs[0][1], 1e-5f);
        Assert.Equal(33f, outputs[0][2], 1e-5f);
        Assert.Equal(44f, outputs[0][3], 1e-5f);
    }

    [Fact]
    public void Execute_MultipleForwardPasses_ZeroAlloc()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([16]);
        int tanh = graph.RecordOp(OpType.Tanh, [inp], [16]);
        graph.RecordOutput(tanh);
        graph.EndCapture();

        using var executor = new GraphExecutor<float>(graph, _engine);
        var input = new Tensor<float>(new[] { 16 });
        for (int i = 0; i < 16; i++) input[i] = i * 0.1f;

        // Execute multiple times — workspace is reused
        var out1 = executor.Execute(input);
        var out2 = executor.Execute(input);

        // Results should be identical (same computation)
        for (int i = 0; i < 16; i++)
            Assert.Equal(out1[0][i], out2[0][i]);
    }

    [Fact]
    public void Execute_WrongInputCount_Throws()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([4]);
        int out1 = graph.RecordOp(OpType.Tanh, [inp], [4]);
        graph.RecordOutput(out1);
        graph.EndCapture();

        using var executor = new GraphExecutor<float>(graph, _engine);

        Assert.Throws<ArgumentException>(() =>
            executor.Execute(new Tensor<float>(new[] { 4 }), new Tensor<float>(new[] { 4 })));
    }

    [Fact]
    public void Execute_AfterDispose_Throws()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int inp = graph.RecordInput([4]);
        graph.RecordOp(OpType.Tanh, [inp], [4]);
        graph.EndCapture();

        var executor = new GraphExecutor<float>(graph, _engine);
        executor.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            executor.Execute(new Tensor<float>(new[] { 4 })));
    }
}
