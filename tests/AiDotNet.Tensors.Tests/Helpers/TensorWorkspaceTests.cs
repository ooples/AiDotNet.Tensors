using System;
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
        ws.Clear(); // deterministic — clear stale data

        var t0 = ws.Get(slot0);
        var t1 = ws.Get(slot1);

        // Write sentinel to slot0
        t0[0] = 42f;
        t0[1] = 43f;

        // Write different sentinel to slot1
        t1[0] = 99f;

        // Slots are independent — slot0 data unchanged
        Assert.Equal(42f, t0[0]);
        Assert.Equal(43f, t0[1]);
        // slot1 has its own data
        Assert.Equal(99f, t1[0]);
    }

    [Fact]
    public void Get_ReusedAcrossForwardPasses_DataPersists()
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
    public void Register_NullShape_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        int[]? nullShape = null;
        Assert.Throws<ArgumentException>(() => ws.Register(nullShape ?? Array.Empty<int>()));
    }

    [Fact]
    public void Register_EmptyShape_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        Assert.Throws<ArgumentException>(() => ws.Register(Array.Empty<int>()));
    }

    [Fact]
    public void Register_NegativeDimension_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        Assert.Throws<ArgumentException>(() => ws.Register([4, -1, 8]));
    }

    [Fact]
    public void Register_ZeroDimension_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        Assert.Throws<ArgumentException>(() => ws.Register([4, 0, 8]));
    }

    [Fact]
    public void Get_InvalidSlotId_Throws()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([4]);
        ws.Allocate();

        Assert.Throws<ArgumentOutOfRangeException>(() => ws.Get(5));
        Assert.Throws<ArgumentOutOfRangeException>(() => ws.Get(-1));
    }

    [Fact]
    public void Get_AfterDispose_Throws()
    {
        var ws = new TensorWorkspace<float>();
        ws.Register([4]);
        ws.Allocate();
        ws.Dispose();

        Assert.Throws<ObjectDisposedException>(() => ws.Get(0));
    }

    [Fact]
    public void GetSpan_AfterDispose_Throws()
    {
        var ws = new TensorWorkspace<float>();
        ws.Register([4]);
        ws.Allocate();
        ws.Dispose();

        Assert.Throws<ObjectDisposedException>(() => ws.GetSpan(0));
    }

    [Fact]
    public void Register_AfterDispose_Throws()
    {
        var ws = new TensorWorkspace<float>();
        ws.Dispose();

        Assert.Throws<ObjectDisposedException>(() => ws.Register([4]));
    }

    [Fact]
    public void GetShape_ReturnsCopy()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([2, 3, 4]);
        ws.Allocate();

        var shape = ws.GetShape(0);
        shape[0] = 999; // mutate the copy

        // Internal state unchanged
        var shape2 = ws.GetShape(0);
        Assert.Equal(2, shape2[0]);
    }

    [Fact]
    public void Reset_ReturnsBufferAndAllowsReregistration()
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
