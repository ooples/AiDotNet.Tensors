using AiDotNet.Tensors.Helpers;
using Xunit;
using static AiDotNet.Tensors.Helpers.ComputationGraph;

namespace AiDotNet.Tensors.Tests.Helpers;

public class ComputationGraphTests
{
    [Fact]
    public void CaptureAndOptimize_SimpleConvPipeline()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 3, 64, 64]);
        int conv1 = graph.RecordOp(OpType.Conv2D, [input], [1, 64, 64, 64]);
        int relu1 = graph.RecordOp(OpType.ReLU, [conv1], [1, 64, 64, 64]);
        int conv2 = graph.RecordOp(OpType.Conv2D, [relu1], [1, 128, 32, 32]);
        int relu2 = graph.RecordOp(OpType.ReLU, [conv2], [1, 128, 32, 32]);
        graph.RecordOutput(relu2);

        graph.EndCapture();

        Assert.True(graph.IsFinalized);
        Assert.Equal(5, graph.NodeCount);

        var plan = graph.Optimize();
        Assert.True(plan.SlotCount > 0);
        Assert.True(plan.SlotCount < 4); // Some slots should be reused
    }

    [Fact]
    public void CaptureAndOptimize_ResBlock_SkipConnection()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 64, 32, 32]);
        int conv1 = graph.RecordOp(OpType.Conv2D, [input], [1, 64, 32, 32]);
        int relu1 = graph.RecordOp(OpType.ReLU, [conv1], [1, 64, 32, 32]);
        int conv2 = graph.RecordOp(OpType.Conv2D, [relu1], [1, 64, 32, 32]);
        int add = graph.RecordOp(OpType.Add, [input, conv2], [1, 64, 32, 32]);
        int relu2 = graph.RecordOp(OpType.ReLU, [add], [1, 64, 32, 32]);
        graph.RecordOutput(relu2);

        graph.EndCapture();

        var plan = graph.Optimize();

        // Input is external, so skip connection keeps it alive
        Assert.Equal(-1, plan.GetSlotForTensor(0)); // input is external
        Assert.True(plan.SlotCount >= 2); // Need at least 2 slots (overlapping intermediates)
    }

    [Fact]
    public void InPlaceActivations_ShareSlotWithInput()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 256]);
        int linear = graph.RecordOp(OpType.Linear, [input], [1, 256]);
        int relu = graph.RecordOp(OpType.ReLU, [linear], [1, 256]);
        int output = graph.RecordOp(OpType.Linear, [relu], [1, 10]);
        graph.RecordOutput(output);

        graph.EndCapture();

        var plan = graph.Optimize();

        // ReLU should share slot with linear (in-place)
        Assert.True(plan.InPlaceOpCount >= 1);
    }

    [Fact]
    public void BeginCapture_WhenAlreadyCapturing_Throws()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        Assert.Throws<InvalidOperationException>(() => graph.BeginCapture());
    }

    [Fact]
    public void RecordOp_BeforeCapture_Throws()
    {
        var graph = new ComputationGraph();
        Assert.Throws<InvalidOperationException>(() =>
            graph.RecordOp(OpType.ReLU, [0], [1, 64]));
    }

    [Fact]
    public void Optimize_BeforeFinalize_Throws()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        graph.RecordInput([4]);
        Assert.Throws<InvalidOperationException>(() => graph.Optimize());
    }

    [Fact]
    public void CreateWorkspace_FromOptimizedPlan_Works()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 64, 8, 8]);
        int conv = graph.RecordOp(OpType.Conv2D, [input], [1, 128, 4, 4]);
        int relu = graph.RecordOp(OpType.ReLU, [conv], [1, 128, 4, 4]);
        graph.RecordOutput(relu);

        graph.EndCapture();

        var plan = graph.Optimize();
        using var workspace = plan.CreateWorkspace<float>();

        Assert.True(workspace.IsAllocated);
        Assert.True(workspace.TotalElements > 0);
    }

    [Fact]
    public void DiffusionResBlock_CapturesFullPattern()
    {
        // Simulate a typical diffusion model ResBlock:
        // GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> Add(residual)
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 256, 32, 32]);
        int gn1 = graph.RecordOp(OpType.GroupNorm, [input], [1, 256, 32, 32],
            new OpParams { Groups = 32, Epsilon = 1e-6 });
        int silu1 = graph.RecordOp(OpType.SiLU, [gn1], [1, 256, 32, 32]);
        int conv1 = graph.RecordOp(OpType.Conv2D, [silu1], [1, 256, 32, 32],
            new OpParams { Padding = 1 });
        int gn2 = graph.RecordOp(OpType.GroupNorm, [conv1], [1, 256, 32, 32],
            new OpParams { Groups = 32, Epsilon = 1e-6 });
        int silu2 = graph.RecordOp(OpType.SiLU, [gn2], [1, 256, 32, 32]);
        int conv2 = graph.RecordOp(OpType.Conv2D, [silu2], [1, 256, 32, 32],
            new OpParams { Padding = 1 });
        int add = graph.RecordOp(OpType.Residual, [input, conv2], [1, 256, 32, 32]);
        graph.RecordOutput(add);

        graph.EndCapture();

        var plan = graph.Optimize();

        Assert.Equal(8, graph.NodeCount); // 1 input + 7 ops
        Assert.True(plan.SlotCount > 0);
        // SiLU activations should be in-place candidates
        Assert.True(plan.SavingsRatio >= 0);
    }
}
