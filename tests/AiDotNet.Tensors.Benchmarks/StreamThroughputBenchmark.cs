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

    // Pool of independent plan pairs for the pipelined-async benchmark.
    // Each pair (a, b) executes one batch's worth of chain — running 4
    // pairs concurrently on different threads gives us 4-way pipelined
    // execution, which is the throughput equivalent of CUDA-streams
    // overlap on the CPU side.
    private (ICompiledPlan<float> a, ICompiledPlan<float> b)[] _planPool = null!;
    private CompiledModelCache<float>[] _poolCachesA = null!;
    private CompiledModelCache<float>[] _poolCachesB = null!;
    private const int _poolSize = 4;

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

        // Plan pool for the pipelined benchmark. Each pair gets its
        // own input tensors so concurrent execution doesn't trip the
        // SetInputs copy.
        _poolCachesA = new CompiledModelCache<float>[_poolSize];
        _poolCachesB = new CompiledModelCache<float>[_poolSize];
        _planPool = new (ICompiledPlan<float>, ICompiledPlan<float>)[_poolSize];
        for (int s = 0; s < _poolSize; s++)
        {
            var inSeed = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });
            var hidSeed = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });
            _poolCachesA[s] = new CompiledModelCache<float>();
            var a = _poolCachesA[s].GetOrCompileInference(inSeed, () =>
            {
                var h = _engine.TensorMatMul(inSeed, _aiW1);
                h = _engine.TensorBroadcastAdd(h, _aiB1);
                return _engine.ReLU(h);
            });
            _poolCachesB[s] = new CompiledModelCache<float>();
            var b = _poolCachesB[s].GetOrCompileInference(hidSeed, () =>
            {
                var o = _engine.TensorMatMul(hidSeed, _aiW2);
                return _engine.TensorBroadcastAdd(o, _aiB2);
            });
            _planPool[s] = (a, b);

            // Pre-warm: a single sequential ChainAsync seeds the
            // SimdGemm packed-weight cache for this pair's W1 / W2
            // tensors. The cache uses a ConditionalWeakTable that's
            // not safe under concurrent insert; pre-warming on this
            // (single-threaded) Setup path means the steady-state
            // parallel benchmark only ever hits the cache lookup,
            // never the cache miss path.
            a.SetInputs(new[] { _aiInputs[0] });
            a.ChainAsync(b).AsTask().GetAwaiter().GetResult();
        }

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
        // Dispose all TorchSharp tensors — they hold native libtorch
        // memory the GC can't reclaim. Closes review-comment #298.1Srq /
        // #298.7tYW.
        if (_torchInputs is not null)
        {
            foreach (var t in _torchInputs)
                t?.Dispose();
        }
        _torchMlpModule?.Dispose();

        _cacheA?.Dispose();
        _cacheB?.Dispose();
        if (_poolCachesA != null)
        {
            for (int s = 0; s < _poolSize; s++)
            {
                _poolCachesA[s]?.Dispose();
                _poolCachesB[s]?.Dispose();
            }
        }
    }

    [Benchmark(Baseline = true)]
    public void PyTorch_BatchSweep_Sequential()
    {
        // torch.no_grad() — pure inference, no autograd. Same rationale
        // as in CompiledPlanChainingBenchmarks: the AiDotNet ChainAsync
        // path doesn't run gradients either, so an autograd-tracked
        // PyTorch baseline isn't a fair comparison. Closes review-
        // comment #298.1Sbh.
        using var _noGrad = torch.no_grad();
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
        // Real pipelining: process batches in waves of `poolSize` at a
        // time, where each wave has every batch on its OWN plan pair —
        // no two concurrent tasks ever touch the same plan instance.
        // Previously this used `i % poolSize` round-robin, which let
        // batches 0, 4, 8, ... call SetInputs/ChainAsync on the same
        // plan pair simultaneously, racing the captured-input copy and
        // making throughput non-deterministic. Closes review-comment
        // #298.7q48 / #298.7tY7.
        //
        // The wave structure caps in-flight concurrency at `poolSize`
        // (4) which is also where the CPU-side parallelism ceiling sits
        // for this kernel size (each fused-linear stage already runs
        // multi-threaded BLAS/AVX inside one plan).
        var pairs = _planPool;
        var poolSize = pairs.Length;
        for (int waveStart = 0; waveStart < NumBatches; waveStart += poolSize)
        {
            int waveEnd = Math.Min(waveStart + poolSize, NumBatches);
            var tasks = new Task[waveEnd - waveStart];
            for (int i = waveStart; i < waveEnd; i++)
            {
                int slot = i - waveStart; // ∈ [0, poolSize) — UNIQUE per task in this wave
                int batchIdx = i;
                tasks[i - waveStart] = Task.Run(async () =>
                {
                    var (a, b) = pairs[slot];
                    a.SetInputs(new[] { _aiInputs[batchIdx] });
                    await a.ChainAsync(b).ConfigureAwait(false);
                });
            }
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
    }
}
#endif
