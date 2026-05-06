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
/// Issue #296 acceptance harness: head-to-head latency comparison of
/// PyTorch (TorchSharp) vs AiDotNet.Tensors compiled-plan chaining
/// across batch sizes {1, 32, 128}. Two-stage Linear→ReLU→Linear
/// (256→256→10) — matches the issue's measured-baseline benchmark.
///
/// Compares four implementations:
///   - PyTorch_TwoStageSequential : eager two-stage call (the 1.00× baseline)
///   - PyTorch_Sequential          : nn.Sequential single forward
///   - Tensors_TwoStageSequential : current sync 2-stage Execute (the gap to close)
///   - Tensors_ChainAsync          : the new ICompiledPlan&lt;T&gt;.ChainAsync path
///
/// Acceptance criteria addressed by this benchmark (issue #296):
///   #1 BS=1   latency: Tensors_ChainAsync ≤ 1.05× PyTorch_Sequential
///   #2 BS=32  latency: Tensors_ChainAsync ≤ 1.05× PyTorch_Sequential
///   #3 BS=128 latency: Tensors_ChainAsync ≤ 1.05× PyTorch_Sequential
///   #5 alloc:  Tensors per-call ≤ 2× PyTorch's at every batch size
///   #6 alloc:  ChainAsync inter-stage boundary alloc = 0 bytes (rebind, not memcpy)
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class CompiledPlanChainingBenchmarks
{
    [Params(1, 32, 128)]
    public int BatchSize { get; set; }

    private CpuEngine _engine = null!;

    // AiDotNet — two-stage MLP, 256 → 256 → 10 (matches issue #296 baseline)
    private Tensor<float> _aiInput = null!;
    private Tensor<float> _aiHiddenSeed = null!;
    private Tensor<float> _aiW1 = null!;
    private Tensor<float> _aiW2 = null!;
    private Tensor<float> _aiB1 = null!;
    private Tensor<float> _aiB2 = null!;
    // Cached input array reused across iterations so the benchmark
    // measures the API's per-call alloc, not the harness's
    // `new[] { _aiInput }` boxing per call.
    private Tensor<float>[] _aiInputArray = null!;
    private Tensor<float>[] _aiHiddenArray = null!;

    // Pre-compiled plans for the new chained-async path. Plan A: Linear+ReLU
    // (input → hidden). Plan B: Linear (hidden → output). ChainAsync
    // pipelines them onto a single execution stream with zero-byte
    // boundary handoff.
    private CompiledModelCache<float> _cacheA = null!;
    private CompiledModelCache<float> _cacheB = null!;
    private ICompiledPlan<float> _planA = null!;
    private ICompiledPlan<float> _planB = null!;

    // TorchSharp equivalents — exact same shapes for fair comparison
    private TorchTensor _torchInput = null!;
    private TorchTensor _torchW1T = null!;
    private TorchTensor _torchW2T = null!;
    private TorchTensor _torchB1 = null!;
    private TorchTensor _torchB2 = null!;
    private TorchSharp.Modules.Sequential _torchMlpModule = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _aiInput = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });
        _aiHiddenSeed = Tensor<float>.CreateRandom(new[] { BatchSize, 256 });
        _aiInputArray = new[] { _aiInput };
        _aiHiddenArray = new[] { _aiHiddenSeed };
        _aiW1 = Tensor<float>.CreateRandom(new[] { 256, 256 });
        _aiW2 = Tensor<float>.CreateRandom(new[] { 256, 10 });
        _aiB1 = Tensor<float>.CreateRandom(new[] { 256 });
        _aiB2 = Tensor<float>.CreateRandom(new[] { 10 });

        // Plan A: input → ReLU(input·W1 + B1). Forward lambda runs once
        // under GraphMode.Enable() to record the lazy graph; the compiler
        // pipeline (CpuFusionPass first) folds the MatMul+Bias+ReLU chain
        // into a single FusedLinearReLU kernel.
        _cacheA = new CompiledModelCache<float>();
        _planA = _cacheA.GetOrCompileInference(_aiInput, () =>
        {
            var h = _engine.TensorMatMul(_aiInput, _aiW1);
            h = _engine.TensorBroadcastAdd(h, _aiB1);
            return _engine.ReLU(h);
        });

        // Plan B: hidden → hidden·W2 + B2. Compiles against a placeholder
        // hidden tensor; ChainAsync rebinds plan B's captured input to
        // plan A's output buffer at chain time.
        _cacheB = new CompiledModelCache<float>();
        _planB = _cacheB.GetOrCompileInference(_aiHiddenSeed, () =>
        {
            var o = _engine.TensorMatMul(_aiHiddenSeed, _aiW2);
            return _engine.TensorBroadcastAdd(o, _aiB2);
        });

        // Mirror to TorchSharp with identical shapes
        _torchInput = torch.randn(BatchSize, 256);
        var torchW1 = torch.randn(256, 256);
        var torchW2 = torch.randn(256, 10);
        _torchB1 = torch.randn(256);
        _torchB2 = torch.randn(10);
        _torchW1T = torchW1.t().contiguous();
        _torchW2T = torchW2.t().contiguous();

        // nn.Sequential single-forward (the strongest TorchSharp baseline)
        var lin1 = torch.nn.Linear(256, 256);
        lin1.weight = new TorchSharp.Modules.Parameter(_torchW1T);
        lin1.bias = new TorchSharp.Modules.Parameter(_torchB1);
        var lin2 = torch.nn.Linear(256, 10);
        lin2.weight = new TorchSharp.Modules.Parameter(_torchW2T);
        lin2.bias = new TorchSharp.Modules.Parameter(_torchB2);
        _torchMlpModule = torch.nn.Sequential(("lin1", lin1), ("relu", torch.nn.ReLU()), ("lin2", lin2));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // TorchSharp tensors hold native libtorch memory; the GC won't
        // free them — explicit Dispose() is required to release the
        // underlying ATen storage. Without this every benchmark run
        // accumulates ~7-10 KB of native libtorch heap. Closes review-
        // comments #298.1Sre / #298.7tYW / #298.75AA.
        _torchInput?.Dispose();
        _torchW1T?.Dispose();
        _torchW2T?.Dispose();
        _torchB1?.Dispose();
        _torchB2?.Dispose();
        _torchMlpModule?.Dispose();

        _cacheA?.Dispose();
        _cacheB?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public TorchTensor PyTorch_TwoStageSequential()
    {
        // Two explicit stages, eager — closest analog to "two compiled
        // plans". Intermediate `hidden` and `activated` are TorchSharp
        // tensors holding native ATen storage; `using` ensures they're
        // released when this scope exits so BenchmarkDotNet's per-call
        // memory measurement isn't polluted by leaked native bytes
        // (closes #298.7tYk / #298.75AL). The returned tensor is the
        // caller's to dispose; BenchmarkDotNet's harness handles that.
        using var hidden = torch.nn.functional.linear(_torchInput, _torchW1T, _torchB1);
        using var activated = torch.nn.functional.relu(hidden);
        return torch.nn.functional.linear(activated, _torchW2T, _torchB2);
    }

    [Benchmark]
    public TorchTensor PyTorch_Sequential()
    {
        // Highest-level TorchSharp API: nn.Sequential treats lin1+relu+lin2
        // as a single forward call. This is the upper-bound baseline the
        // issue's acceptance criteria target (≤1.05×).
        return _torchMlpModule.forward(_torchInput);
    }

    [Benchmark]
    public Tensor<float> Tensors_TwoStageSequential()
    {
        // Current sync two-stage path: run plan A, copy its output into plan
        // B's captured input via SetInputs, run plan B. This is the
        // pre-#296 baseline that the new ChainAsync replaces.
        _planA.SetInputs(_aiInputArray);
        var hidden = _planA.Execute();
        _aiHiddenArray[0] = hidden;
        _planB.SetInputs(_aiHiddenArray);
        return _planB.Execute();
    }

    [Benchmark]
    public async ValueTask<Tensor<float>> Tensors_ChainAsync()
    {
        // The new path. SetInputs once, then ChainAsync queues both plans
        // onto a single execution stream and rebinds the boundary tensor
        // zero-copy (criterion #6). On CPU the stream is a Channel<>-backed
        // worker; on GPU it would wrap the engine's cudaStream_t.
        _planA.SetInputs(_aiInputArray);
        return await _planA.ChainAsync(_planB).ConfigureAwait(false);
    }
}
#endif
