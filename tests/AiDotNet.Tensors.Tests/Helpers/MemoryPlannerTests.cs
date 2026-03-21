using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class MemoryPlannerTests
{
    [Fact]
    public void SimpleConvPipeline_ProducesValidPlan()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([1, 3, 64, 64]);
        int conv1 = planner.AddOp("Conv2D", [input], [1, 16, 64, 64]);
        int relu1 = planner.AddOp("ReLU", [conv1], [1, 16, 64, 64], canInPlace: true);
        int conv2 = planner.AddOp("Conv2D", [relu1], [1, 32, 32, 32]);

        var plan = planner.Plan();

        Assert.True(plan.SlotCount > 0);
        Assert.Equal(-1, plan.GetSlotForTensor(input)); // external input
        Assert.True(plan.GetSlotForTensor(conv1) >= 0);
        Assert.True(plan.GetSlotForTensor(conv2) >= 0);
    }

    [Fact]
    public void InPlaceReLU_ReducesSlotCount()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([1, 64, 32, 32]);
        int conv = planner.AddOp("Conv2D", [input], [1, 64, 32, 32]);
        int relu = planner.AddOp("ReLU", [conv], [1, 64, 32, 32], canInPlace: true);
        int output = planner.AddOp("Conv2D", [relu], [1, 128, 16, 16]);

        var plan = planner.Plan();

        // ReLU should be in-place: conv and relu share a slot
        Assert.Equal(plan.GetSlotForTensor(conv), plan.GetSlotForTensor(relu));
        Assert.True(plan.InPlaceOpCount >= 1);
    }

    [Fact]
    public void ResidualBlock_SkipConnectionPreventsInPlace()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([1, 64, 32, 32]);

        // Main path
        int conv1 = planner.AddOp("Conv2D", [input], [1, 64, 32, 32]);
        int relu = planner.AddOp("ReLU", [conv1], [1, 64, 32, 32], canInPlace: true);
        int conv2 = planner.AddOp("Conv2D", [relu], [1, 64, 32, 32]);

        // Residual: input + conv2 (input still alive from beginning)
        int add = planner.AddOp("Add", [input, conv2], [1, 64, 32, 32]);

        var plan = planner.Plan();

        Assert.True(plan.SlotCount >= 2);
        Assert.True(plan.GetSlotForTensor(add) >= 0);
    }

    [Fact]
    public void UNetEncoder_Decoder_MemorySavings()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([1, 3, 256, 256]);

        // Encoder
        int enc0 = planner.AddOp("Conv+ReLU", [input], [1, 64, 256, 256]);
        int enc1 = planner.AddOp("Conv+ReLU", [enc0], [1, 128, 128, 128]);
        int enc2 = planner.AddOp("Conv+ReLU", [enc1], [1, 256, 64, 64]);

        // Middle
        int mid = planner.AddOp("Conv+ReLU", [enc2], [1, 256, 64, 64]);

        // Decoder with skip connections
        int dec2 = planner.AddOp("Upsample+Conv", [mid, enc2], [1, 128, 128, 128]);
        int dec1 = planner.AddOp("Upsample+Conv", [dec2, enc1], [1, 64, 256, 256]);
        int dec0 = planner.AddOp("Upsample+Conv", [dec1, enc0], [1, 64, 256, 256]);

        // Output
        int output = planner.AddOp("Conv1x1", [dec0], [1, 3, 256, 256]);

        var plan = planner.Plan();

        // With skip connections, encoder outputs stay alive until decoder uses them
        // But after skip connections are consumed, slots should be reused
        Assert.True(plan.SavingsRatio > 0,
            $"Expected memory savings but got {plan.SavingsRatio:P1}");
        Assert.True(plan.SlotCount < 8,
            $"Expected fewer slots than tensors but got {plan.SlotCount} slots for 8 intermediates");
    }

    [Fact]
    public void CreateWorkspace_ProducesUsableWorkspace()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([4]);
        int t1 = planner.AddOp("Op1", [input], [8]);
        int t2 = planner.AddOp("Op2", [t1], [8]);

        var plan = planner.Plan();
        using var workspace = plan.CreateWorkspace<float>();

        Assert.True(workspace.IsAllocated);
        Assert.True(workspace.TotalElements > 0);

        // Get tensor from workspace slot
        int slot = plan.GetSlotForTensor(t2);
        if (slot >= 0)
        {
            var tensor = workspace.Get(slot);
            Assert.True(tensor.Length > 0);
        }
    }

    [Fact]
    public void MultiOutputOp_TracksAllOutputs()
    {
        var planner = new MemoryPlanner();
        int input = planner.AddExternalInput([1, 256]);

        // GroupNorm produces output, mean, and variance
        int[] outputs = planner.AddMultiOutputOp("GroupNorm",
            [input],
            [[1, 256], [1], [1]]);

        int next = planner.AddOp("SiLU", [outputs[0]], [1, 256], canInPlace: true);

        var plan = planner.Plan();

        Assert.True(plan.GetSlotForTensor(outputs[0]) >= 0);
        Assert.True(plan.GetSlotForTensor(outputs[1]) >= 0);
        Assert.True(plan.GetSlotForTensor(outputs[2]) >= 0);
    }
}
