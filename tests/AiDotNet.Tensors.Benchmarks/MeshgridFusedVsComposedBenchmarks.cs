using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Promotion gate for the fused Meshgrid2D kernel (#922): the fused route must beat the composed reshape +
/// broadcast route by >= 1.10x median with p95 no worse than +10%. Run with AIDOTNET_DIRECTGPU_BACKENDS set to
/// one backend (opencl, hip or cuda). Both routes read device-resident sources and synchronize before returning.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
public class MeshgridFusedVsComposedBenchmarks
{
    private const int N1 = 256;
    private DirectGpuTensorEngine? _engine;
    private IDirectGpuBackend? _backend;
    private IMeshgridBackend? _fused;
    private Tensor<float> _source0 = new Tensor<float>(new[] { 1 });
    private Tensor<float> _source1 = new Tensor<float>(new[] { 1 });
    private IGpuBuffer? _device0;
    private IGpuBuffer? _device1;

    [Params(256, 1024)]
    public int N0 { get; set; }

    [Params(false, true)]
    public bool Xy { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _engine = new DirectGpuTensorEngine();
        _backend = _engine.GetBackend()
            ?? throw new InvalidOperationException("No DirectGpu backend is available.");
        _fused = _backend as IMeshgridBackend
            ?? throw new InvalidOperationException($"{_backend.GetType().Name} has no fused meshgrid.");
        var s0 = new float[N0];
        var s1 = new float[N1];
        for (int i = 0; i < N0; i++) s0[i] = i;
        for (int j = 0; j < N1; j++) s1[j] = j;
        _source0 = new Tensor<float>(s0, new[] { N0 });
        _source1 = new Tensor<float>(s1, new[] { N1 });
        _device0 = _backend.AllocateBuffer(s0);
        _device1 = _backend.AllocateBuffer(s1);
        Composed();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _device0?.Dispose();
        _device1?.Dispose();
    }

    /// <summary>The route the engine took before #922: reshape each source onto its axis, then broadcast.</summary>
    [Benchmark(Baseline = true)]
    public void Composed()
    {
        if (_engine is null || _backend is null) return;
        int[] grid = Xy ? new[] { N1, N0 } : new[] { N0, N1 };
        var g0 = _engine.TensorBroadcastTo(_engine.Reshape(_source0, Xy ? new[] { 1, N0 } : new[] { N0, 1 }), grid);
        var g1 = _engine.TensorBroadcastTo(_engine.Reshape(_source1, Xy ? new[] { N1, 1 } : new[] { 1, N1 }), grid);
        _backend.Synchronize();
        g0.Dispose();
        g1.Dispose();
    }

    [Benchmark]
    public void Fused()
    {
        if (_backend is null || _fused is null || _device0 is null || _device1 is null) return;
        using var o0 = _backend.AllocateBuffer(N0 * N1);
        using var o1 = _backend.AllocateBuffer(N0 * N1);
        _fused.Meshgrid2DPair(_device0, _device1, o0, o1, N0, N1, Xy);
        _backend.Synchronize();
    }
}
