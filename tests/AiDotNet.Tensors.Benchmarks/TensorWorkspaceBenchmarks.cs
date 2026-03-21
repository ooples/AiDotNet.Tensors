using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks proving TensorWorkspace achieves zero allocation during forward passes.
/// Compares workspace-backed Get() vs fresh Tensor allocation for each operation.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class TensorWorkspaceBenchmarks
{
    // Simulates a DiffusionResBlock forward pass with 4 intermediate tensors
    private const int Batch = 1;
    private const int Channels = 256;
    private const int Height = 64;
    private const int Width = 64;

    private TensorWorkspace<float> _workspace = new();
    private int _slot0, _slot1, _slot2, _slot3;

    [GlobalSetup]
    public void Setup()
    {
        _workspace = new TensorWorkspace<float>();
        _slot0 = _workspace.Register([Batch, Channels, Height, Width]);      // conv1 output
        _slot1 = _workspace.Register([Batch, Channels, Height, Width]);      // norm output
        _slot2 = _workspace.Register([Batch, Channels, Height, Width]);      // activation output
        _slot3 = _workspace.Register([Batch, Channels, Height, Width]);      // conv2 output
        _workspace.Allocate();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _workspace.Dispose();
    }

    /// <summary>
    /// TensorWorkspace.Get() — zero allocation, returns pre-mapped Memory view.
    /// This simulates getting all 4 intermediate tensors in a ResBlock forward pass.
    /// </summary>
    [Benchmark(Baseline = true)]
    public int Workspace_Get_4Tensors()
    {
        var t0 = _workspace.Get(_slot0);
        var t1 = _workspace.Get(_slot1);
        var t2 = _workspace.Get(_slot2);
        var t3 = _workspace.Get(_slot3);
        return t0.Length + t1.Length + t2.Length + t3.Length;
    }

    /// <summary>
    /// Standard allocation — allocates 4 fresh tensors per forward pass.
    /// This is what every ML framework does without workspace pre-allocation.
    /// </summary>
    [Benchmark]
    public int StandardAlloc_4Tensors()
    {
        var t0 = new Tensor<float>(new[] { Batch, Channels, Height, Width });
        var t1 = new Tensor<float>(new[] { Batch, Channels, Height, Width });
        var t2 = new Tensor<float>(new[] { Batch, Channels, Height, Width });
        var t3 = new Tensor<float>(new[] { Batch, Channels, Height, Width });
        return t0.Length + t1.Length + t2.Length + t3.Length;
    }

    /// <summary>
    /// TensorAllocator.Rent — pooled allocation (ArrayPool backed).
    /// Better than raw allocation but still has pool lookup overhead.
    /// </summary>
    [Benchmark]
    public int TensorAllocatorRent_4Tensors()
    {
        var shape = new[] { Batch, Channels, Height, Width };
        var t0 = TensorAllocator.Rent<float>(shape);
        var t1 = TensorAllocator.Rent<float>(shape);
        var t2 = TensorAllocator.Rent<float>(shape);
        var t3 = TensorAllocator.Rent<float>(shape);
        int len = t0.Length + t1.Length + t2.Length + t3.Length;
        TensorAllocator.Return(t0);
        TensorAllocator.Return(t1);
        TensorAllocator.Return(t2);
        TensorAllocator.Return(t3);
        return len;
    }

    /// <summary>
    /// Single Get() — measures the per-tensor overhead of workspace access.
    /// Should be near-zero (just Memory constructor + array slice).
    /// </summary>
    [Benchmark]
    public Tensor<float> Workspace_SingleGet()
    {
        return _workspace.Get(_slot0);
    }

    /// <summary>
    /// UNet-scale workspace: 12 slots at various spatial sizes.
    /// Proves workspace scales to real model sizes.
    /// </summary>
    [Benchmark]
    public int Workspace_UNetScale_12Slots()
    {
        using var ws = new TensorWorkspace<float>();
        ws.Register([1, 64, 256, 256]);
        ws.Register([1, 128, 128, 128]);
        ws.Register([1, 256, 64, 64]);
        ws.Register([1, 512, 32, 32]);
        ws.Register([1, 1024, 16, 16]);
        ws.Register([1, 1024, 16, 16]);
        ws.Register([1, 512, 32, 32]);
        ws.Register([1, 256, 64, 64]);
        ws.Register([1, 128, 128, 128]);
        ws.Register([1, 64, 256, 256]);
        ws.Register([1, 64, 256, 256]);
        ws.Register([1, 3, 256, 256]);
        ws.Allocate();
        return ws.TotalElements;
    }
}
