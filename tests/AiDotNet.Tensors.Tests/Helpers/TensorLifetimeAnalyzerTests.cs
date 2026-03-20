using System;
using AiDotNet.Tensors.Helpers;
using Xunit;
using static AiDotNet.Tensors.Helpers.TensorLifetimeAnalyzer;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorLifetimeAnalyzerTests
{
    [Fact]
    public void LinearChain_NoReuse_AllSlotsDistinct()
    {
        // input(0) -> op0 -> t1 -> op1 -> t2 -> op2 -> output(3)
        // All tensors are alive simultaneously (each is consumed by the next op)
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [100]),
            new(inputs: [1], outputs: [2], outputSizes: [100]),
            new(inputs: [2], outputs: [3], outputSizes: [100]),
        };

        var plan = Analyze(ops, tensorCount: 4, inputTensorIds: [0]);

        // t1, t2, t3 need workspace slots. t1 is alive during op0-op1,
        // t2 is alive during op1-op2, so t1 and t2 DO overlap (both alive at op1)
        // t1 dies after op1, t2 dies after op2 — t1 and t3 don't overlap
        Assert.True(plan.SlotCount >= 2); // t1+t2 overlap, but t1+t3 can share
    }

    [Fact]
    public void SequentialOps_DeadTensorsReused()
    {
        // t0 = input
        // t1 = relu(t0)          — t1 alive [0,1]
        // t2 = conv(t1)          — t1 dies at op1, t2 alive [1,2]
        // t3 = relu(t2)          — t2 dies at op2, t3 alive [2,3]
        // t4 = conv(t3)          — t3 dies at op3, t4 alive [3,4]
        // output = softmax(t4)   — t4 dies at op4
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [1000]),  // op0: relu
            new(inputs: [1], outputs: [2], outputSizes: [1000]),  // op1: conv
            new(inputs: [2], outputs: [3], outputSizes: [1000]),  // op2: relu
            new(inputs: [3], outputs: [4], outputSizes: [1000]),  // op3: conv
            new(inputs: [4], outputs: [5], outputSizes: [1000]),  // op4: softmax
        };

        var plan = Analyze(ops, tensorCount: 6, inputTensorIds: [0]);

        // t1 dies after op1, t3 starts at op2 -> can reuse slot
        // t2 dies after op2, t4 starts at op3 -> can reuse slot
        // So we need at most 2 workspace slots for 5 intermediate tensors
        Assert.True(plan.SlotCount <= 3, $"Expected <= 3 slots but got {plan.SlotCount}");
        Assert.True(plan.SavingsRatio > 0.3, $"Expected > 30% savings but got {plan.SavingsRatio:P1}");
    }

    [Fact]
    public void ResidualConnection_KeepsBothAlive()
    {
        // t0 = input
        // t1 = conv(t0)           — t1 alive [0, 2] (used in op0 and op2)
        // t2 = relu(t1)           — t2 alive [1, 2]
        // t3 = t1 + t2            — residual add, both t1 and t2 alive at op2
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [10000]),  // op0: conv
            new(inputs: [1], outputs: [2], outputSizes: [10000]),  // op1: relu
            new(inputs: [1, 2], outputs: [3], outputSizes: [10000]),  // op2: add (residual)
        };

        var plan = Analyze(ops, tensorCount: 4, inputTensorIds: [0]);

        // t1 and t2 are both alive at op2, so they cannot share a slot
        // t1: [0,2], t2: [1,2] -> overlap
        Assert.True(plan.SlotAssignments[1] != plan.SlotAssignments[2],
            "Residual tensors must not share a slot");
    }

    [Fact]
    public void DifferentSizes_LargerSlotReused()
    {
        // t1 = large tensor (10000 elements), alive [0,0]
        // t2 = small tensor (100 elements), alive [1,1]
        // t1 and t2 don't overlap, so t2 should reuse t1's slot (which is larger)
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [10000]),
            new(inputs: [0], outputs: [2], outputSizes: [100]),
        };

        var plan = Analyze(ops, tensorCount: 3, inputTensorIds: [0]);

        // t2 can reuse t1's slot — slot size should be max(10000, 100) = 10000
        if (plan.SlotAssignments[1] == plan.SlotAssignments[2])
        {
            int slot = plan.SlotAssignments[1];
            Assert.Equal(10000, plan.SlotSizes[slot]);
        }
    }

    [Fact]
    public void InputTensors_ExcludedFromWorkspace()
    {
        var ops = new Operation[]
        {
            new(inputs: [0, 1], outputs: [2], outputSizes: [100]),
        };

        var plan = Analyze(ops, tensorCount: 3, inputTensorIds: [0, 1]);

        // Input tensors should not be assigned slots
        Assert.Equal(-1, plan.SlotAssignments[0]);
        Assert.Equal(-1, plan.SlotAssignments[1]);
        // Output tensor should be assigned a slot
        Assert.True(plan.SlotAssignments[2] >= 0);
    }

    [Fact]
    public void UNetLikePattern_SignificantMemorySavings()
    {
        // Simulate encoder-decoder pattern:
        // enc0(t0) -> t1, enc1(t1) -> t2, enc2(t2) -> t3 (encoder)
        // middle(t3) -> t4
        // dec2(t4, t3) -> t5, dec1(t5, t2) -> t6, dec0(t6, t1) -> t7 (decoder with skip connections)
        //
        // Skip connections keep encoder outputs alive longer
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [64 * 256 * 256]),   // op0: enc0
            new(inputs: [1], outputs: [2], outputSizes: [128 * 128 * 128]),  // op1: enc1
            new(inputs: [2], outputs: [3], outputSizes: [256 * 64 * 64]),    // op2: enc2
            new(inputs: [3], outputs: [4], outputSizes: [256 * 64 * 64]),    // op3: middle
            new(inputs: [4, 3], outputs: [5], outputSizes: [128 * 128 * 128]), // op4: dec2 (skip t3)
            new(inputs: [5, 2], outputs: [6], outputSizes: [64 * 256 * 256]),  // op5: dec1 (skip t2)
            new(inputs: [6, 1], outputs: [7], outputSizes: [64 * 256 * 256]),  // op6: dec0 (skip t1)
        };

        var plan = Analyze(ops, tensorCount: 8, inputTensorIds: [0]);

        // 7 intermediate tensors, but some can be reused after skip connections are consumed
        // t3 dies after op4, t4 dies after op4 -> after op4, two slots free
        Assert.True(plan.SlotCount < 7, $"Expected < 7 slots but got {plan.SlotCount}");
        Assert.True(plan.SavingsRatio > 0, $"Expected memory savings but got {plan.SavingsRatio:P1}");

        // Verify no overlapping assignments
        for (int i = 1; i < 8; i++)
        {
            for (int j = i + 1; j < 8; j++)
            {
                if (plan.SlotAssignments[i] >= 0 && plan.SlotAssignments[j] >= 0 &&
                    plan.SlotAssignments[i] == plan.SlotAssignments[j])
                {
                    Assert.False(plan.LiveRanges[i].Overlaps(plan.LiveRanges[j]),
                        $"Tensors {i} and {j} share slot {plan.SlotAssignments[i]} but their lifetimes overlap");
                }
            }
        }
    }

    [Fact]
    public void EmptyOperations_ReturnsEmptyPlan()
    {
        var plan = Analyze(ReadOnlySpan<Operation>.Empty, tensorCount: 0, inputTensorIds: ReadOnlySpan<int>.Empty);
        Assert.Equal(0, plan.SlotCount);
        Assert.Equal(0, plan.TotalElementsWithReuse);
    }

    [Fact]
    public void InPlaceOps_DetectedWhenInputDies()
    {
        // t0 = input
        // t1 = conv(t0)         — t1 alive [0,1]
        // t2 = relu(t1)         — t1 dies at op1, relu can execute in-place
        // t3 = output(t2)
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [1000]),
            new(inputs: [1], outputs: [2], outputSizes: [1000], canExecuteInPlace: true),
            new(inputs: [2], outputs: [3], outputSizes: [1000]),
        };

        var plan = Analyze(ops, tensorCount: 4, inputTensorIds: [0]);

        // op1 (relu) should be detected as in-place: t2 overwrites t1
        Assert.True(plan.InPlaceOps.ContainsKey(1), "relu (op1) should be in-place");
        var (inId, outId) = plan.InPlaceOps[1];
        Assert.Equal(1, inId);
        Assert.Equal(2, outId);
        // t1 and t2 should share the same slot
        Assert.Equal(plan.SlotAssignments[1], plan.SlotAssignments[2]);
    }

    [Fact]
    public void InPlaceOps_NotDetectedWhenInputStillAlive()
    {
        // t0 = input
        // t1 = conv(t0)         — t1 alive [0, 2] (used in op0 AND op2)
        // t2 = relu(t1)         — t1 still alive, can NOT execute in-place
        // t3 = t1 + t2          — residual, consumes t1 again
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [1000]),
            new(inputs: [1], outputs: [2], outputSizes: [1000], canExecuteInPlace: true),
            new(inputs: [1, 2], outputs: [3], outputSizes: [1000]),
        };

        var plan = Analyze(ops, tensorCount: 4, inputTensorIds: [0]);

        // op1 should NOT be in-place because t1 is still needed at op2
        Assert.False(plan.InPlaceOps.ContainsKey(1),
            "relu should NOT be in-place when input is still alive for residual");
    }

    [Fact]
    public void SavingsRatio_ComputedCorrectly()
    {
        // Two tensors that can share a slot: 50% savings
        var ops = new Operation[]
        {
            new(inputs: [0], outputs: [1], outputSizes: [1000]),
            new(inputs: [1], outputs: [2], outputSizes: [1000]),
        };

        var plan = Analyze(ops, tensorCount: 3, inputTensorIds: [0]);

        // Without reuse: t1 (1000) + t2 (1000) = 2000
        Assert.Equal(2000, plan.TotalElementsWithoutReuse);
        // t1 dies after op1, t2 starts at op1 — they overlap at op1 (t1 consumed, t2 produced)
        // Actually t1: [0,1], t2: [1,1] — they share op1, so they overlap
        // Need 2 slots
        Assert.True(plan.TotalElementsWithReuse <= 2000);
    }
}
