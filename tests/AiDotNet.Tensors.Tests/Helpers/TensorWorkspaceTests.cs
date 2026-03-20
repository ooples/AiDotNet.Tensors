using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorWorkspaceTests
{
    [Fact]
    public void Register_ReturnsSequentialIds()
    {
        using var ws = new TensorWorkspace<float>();
        var id0 = ws.Register([1, 3, 64, 64]);
        var id1 = ws.Register([1, 128, 32, 32]);
        var id2 = ws.Register([1, 256, 16, 16]);

        Assert.Equal(0, id0);
        Assert.Equal(1, id1);
        Assert.Equal(2, id2);
        Assert.Equal(3, ws.SlotCount);
    }

    [Fact]
    public void Allocate_ComputesCorrectTotalSize()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([1, 4, 8, 8]);    // 256
        ws.Register([1, 8, 4, 4]);    // 128
        ws.Allocate();

        Assert.Equal(256 + 128, ws.TotalElements);
        Assert.True(ws.IsAllocated);
    }

    [Fact]
    public void Get_ReturnsTensorWithCorrectShape()
    {
        using var ws = new TensorWorkspace<double>();
        var slot = ws.Register([2, 3, 4]);
        ws.Allocate();

        var tensor = ws.Get(slot);
        Assert.Equal(new[] { 2, 3, 4 }, tensor.Shape);
        Assert.Equal(24, tensor.Length);
    }

    [Fact]
    public void Get_TensorsShareBackingBuffer()
    {
        using var ws = new TensorWorkspace<float>();
        var slot0 = ws.Register([4]);
        var slot1 = ws.Register([4]);
        ws.Allocate();

        var t0 = ws.Get(slot0);
        var t1 = ws.Get(slot1);

        // Write to slot0
        t0[0] = 42f;
        t0[1] = 43f;

        // slot1 is at a different offset — should not see slot0 data
        Assert.NotEqual(42f, t1[0]);

        // But both are backed by the same underlying array
        Assert.Equal(42f, t0[0]);
    }

    [Fact]
    public void Get_ReusedAcrossForwardPasses_ZeroAllocation()
    {
        using var ws = new TensorWorkspace<float>();
        var slot = ws.Register([1, 64, 8, 8]);
        ws.Allocate();

        // Simulate multiple forward passes — should return same memory
        var t1 = ws.Get(slot);
        t1[0] = 1.0f;

        var t2 = ws.Get(slot);
        // Same slot returns same memory — previous data visible
        Assert.Equal(1.0f, t2[0]);
    }

    [Fact]
    public void Register_AfterAllocate_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([4]);
        ws.Allocate();

        Assert.Throws<InvalidOperationException>(() => ws.Register([4]));
    }

    [Fact]
    public void Get_BeforeAllocate_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([4]);

        Assert.Throws<InvalidOperationException>(() => ws.Get(0));
    }

    [Fact]
    public void Reset_AllowsReregistration()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([4]);
        ws.Allocate();

        ws.Reset();
        Assert.False(ws.IsAllocated);
        Assert.Equal(0, ws.SlotCount);

        // Can register again
        var slot = ws.Register([8]);
        ws.Allocate();
        Assert.Equal(8, ws.TotalElements);
    }

    [Fact]
    public void LargeWorkspace_UNetScale()
    {
        // Simulate a small UNet: multiple conv outputs at different spatial sizes
        using var ws = new TensorWorkspace<float>();
        ws.Register([1, 64, 32, 32]);    // encoder level 0
        ws.Register([1, 128, 16, 16]);   // encoder level 1
        ws.Register([1, 256, 8, 8]);     // encoder level 2
        ws.Register([1, 256, 8, 8]);     // middle
        ws.Register([1, 128, 16, 16]);   // decoder level 1
        ws.Register([1, 64, 32, 32]);    // decoder level 0
        ws.Allocate();

        int expected = 64*32*32 + 128*16*16 + 256*8*8 + 256*8*8 + 128*16*16 + 64*32*32;
        Assert.Equal(expected, ws.TotalElements);

        // All slots return valid tensors
        for (int i = 0; i < ws.SlotCount; i++)
        {
            var t = ws.Get(i);
            Assert.True(t.Length > 0);
        }
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        var ws = new TensorWorkspace<float>();
        ws.Register([1024]);
        ws.Allocate();
        ws.Dispose();
        ws.Dispose(); // should not throw
    }
}
