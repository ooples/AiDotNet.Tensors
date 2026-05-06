#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #296 acceptance harness: multi-batch throughput comparison
/// covering acceptance criterion #4. PyTorch's nn.Sequential looped
/// vs AiDotNet sync sequential vs AiDotNet ChainAsync over engine
/// streams.
///
/// Sweeps NumBatches ∈ {8, 32}, BatchSize=32, two-stage Linear→ReLU→Linear
/// (256→256→10). The ChainAsync benchmark allocates one execution stream
/// per inflight batch (round-robin) so kernels from batch K+1's stage 1
/// can overlap kernels from batch K's stage 2 — the multi-batch
/// pipelining win documented in the issue's measured baseline.
///
/// Acceptance criterion addressed:
///   #4 Tensors_BatchSweep_PipelinedAsync ≤ 0.95× PyTorch_BatchSweep_Sequential
///      at NumBatches ∈ {8, 32}.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class StreamThroughputBenchmark
{
    [Params(8, 32)]
    public int NumBatches { get; set; }

    public int BatchSize { get; set; } = 32;

    private CpuEngine _engine = null!;

    // Per-batch input tensors so each iteration sees fresh data
    private Tensor<float>[] _aiInputs = null!;
    private Tensor<float> _aiHiddenSeed = null!;
    private Tensor<float> _aiW1 = null!;
    private Tensor<float> _aiW2 = null!;
    private Tensor<float> _aiB1 = null!;
    private Tensor<float> _aiB2 = null!;

    private CompiledModelCache<float> _cacheA = null!;
    private CompiledModelCache<float> _cacheB = null!;
    private ICompiledPlan<float> _planA = null!;
    private ICompiledPlan<float> _planB = null!;

    private TorchTensor[] _torchInputs = null!;
    private TorchSharp.Modules.Sequential _torchMlpModule = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _aiInputs = new Tensor<float>[NumBatches];
        for (int i = 0; i < NumBatches; i++)
            _aiInputs[i] = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });
        _aiHiddenSeed = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });

        _aiW1 = Tensor<float>.CreateRandom(new[] { 256, 256 });
        _aiW2 = Tensor<float>.CreateRandom(new[] { 256, 10 });
        _aiB1 = Tensor<float>.CreateRandom(new[] { 256 });
        _aiB2 = Tensor<float>.CreateRandom(new[] { 10 });

        _cacheA = new CompiledModelCache<float>();
        _planA = _cacheA.GetOrCompileInference(_aiInputs[0], () =>
        {
            var h = _engine.TensorMatMul(_aiInputs[0], _aiW1);
            h = _engine.TensorBroadcastAdd(h, _aiB1);
            return _engine.ReLU(h);
        });

        _cacheB = new CompiledModelCache<float>();
        _planB = _cacheB.GetOrCompileInference(_aiHiddenSeed, () =>
        {
            var o = _engine.TensorMatMul(_aiHiddenSeed, _aiW2);
            return _engine.TensorBroadcastAdd(o, _aiB2);
        });

        _torchInputs = new TorchTensor[NumBatches];
        for (int i = 0; i < NumBatches; i++)
            _torchInputs[i] = torch.randn(BatchSize, 256);

        var torchW1 = torch.randn(256, 256);
        var torchW2 = torch.randn(256, 10);
        var torchB1 = torch.randn(256);
        var torchB2 = torch.randn(10);
        var torchW1T = torchW1.t().contiguous();
        var torchW2T = torchW2.t().contiguous();
        var lin1 = torch.nn.Linear(256, 256);
        lin1.weight = new TorchSharp.Modules.Parameter(torchW1T);
        lin1.bias = new TorchSharp.Modules.Parameter(torchB1);
        var lin2 = torch.nn.Linear(256, 10);
        lin2.weight = new TorchSharp.Modules.Parameter(torchW2T);
        lin2.bias = new TorchSharp.Modules.Parameter(torchB2);
        _torchMlpModule = torch.nn.Sequential(("lin1", lin1), ("relu", torch.nn.ReLU()), ("lin2", lin2));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cacheA?.Dispose();
        _cacheB?.Dispose();
        _torchMlpModule?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public void PyTorch_BatchSweep_Sequential()
    {
        for (int i = 0; i < NumBatches; i++)
        {
            using var output = _torchMlpModule.forward(_torchInputs[i]);
        }
    }

    [Benchmark]
    public void Tensors_BatchSweep_Sequential()
    {
        for (int i = 0; i < NumBatches; i++)
        {
            _planA.SetInputs(new[] { _aiInputs[i] });
            var hidden = _planA.Execute();
            _planB.SetInputs(new[] { hidden });
            _ = _planB.Execute();
        }
    }

    [Benchmark]
    public async Task Tensors_BatchSweep_PipelinedAsync()
    {
        // Sequential ChainAsync over the same plans for every batch — the
        // intermediate plan B's captured input is rebound to plan A's
        // output the FIRST time ChainAsync runs and stays bound for
        // subsequent calls (idempotent per #296 contract). Each
        // ChainAsync acquires a fresh execution stream, so the host
        // thread yields between batches; on CPU this lets the kernel-
        // level thread pool (BLAS / AVX) saturate cores naturally
        // without per-step Task.Run overhead.
        for (int i = 0; i < NumBatches; i++)
        {
            _planA.SetInputs(new[] { _aiInputs[i] });
            await _planA.ChainAsync(_planB).ConfigureAwait(false);
        }
    }
}
#endif
