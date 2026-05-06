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
/// Issue #296 acceptance harness for criterion #7: synthetic diffusion-
/// style denoising loop. A 50-step iterated forward pass through a
/// two-layer noise-predictor (Linear→ReLU→Linear), measured wall-clock.
/// Compares PyTorch nn.Sequential looped 50 times vs Tensors ChainAsync
/// looped 50 times.
///
/// Note: TorchSharp does not expose torch.compile. The closest available
/// baseline on the .NET side is nn.Sequential with the per-step JIT
/// path libtorch uses; the criterion target is ≤ 1.05× of that.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class DiffusionPipelineBenchmark
{
    public int DenoisingSteps { get; set; } = 50;
    public int BatchSize { get; set; } = 4;     // typical diffusion BS
    public int Hidden { get; set; } = 256;       // small UNet-like dim

    private CpuEngine _engine = null!;

    // Per-step state: input (the noisy sample), w1/b1/w2/b2 weights of
    // the noise predictor (kept fixed across steps — real diffusion
    // would condition on time-step but the perf shape is the same).
    private Tensor<float> _aiInput = null!;
    private Tensor<float> _aiHiddenSeed = null!;
    private Tensor<float> _aiW1 = null!;
    private Tensor<float> _aiW2 = null!;
    private Tensor<float> _aiB1 = null!;
    private Tensor<float> _aiB2 = null!;

    private CompiledModelCache<float> _cacheA = null!;
    private CompiledModelCache<float> _cacheB = null!;
    private ICompiledPlan<float> _planA = null!;
    private ICompiledPlan<float> _planB = null!;

    private TorchTensor _torchInput = null!;
    private TorchSharp.Modules.Sequential _torchUnet = null!;

    // Cached single-element input array reused across the 50-step
    // denoising loop. Without it, each iteration's `new[] { _aiInput }`
    // would allocate a fresh Tensor<float>[1] (24 B × 50 = 1.2 KB per
    // benchmark call) and skew BenchmarkDotNet's MemoryDiagnoser
    // measurement away from the API's actual per-call alloc.
    // Closes review-comment #298.75Ac.
    private Tensor<float>[] _aiInputArray = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _aiInput = Tensor<float>.CreateRandom(new[] { BatchSize, Hidden });
        _aiHiddenSeed = Tensor<float>.CreateRandom(new[] { BatchSize, Hidden });
        _aiInputArray = new[] { _aiInput };
        _aiW1 = Tensor<float>.CreateRandom(new[] { Hidden, Hidden });
        _aiW2 = Tensor<float>.CreateRandom(new[] { Hidden, Hidden });
        _aiB1 = Tensor<float>.CreateRandom(new[] { Hidden });
        _aiB2 = Tensor<float>.CreateRandom(new[] { Hidden });

        _cacheA = new CompiledModelCache<float>();
        _planA = _cacheA.GetOrCompileInference(_aiInput, () =>
        {
            var h = _engine.TensorMatMul(_aiInput, _aiW1);
            h = _engine.TensorBroadcastAdd(h, _aiB1);
            return _engine.ReLU(h);
        });
        _cacheB = new CompiledModelCache<float>();
        _planB = _cacheB.GetOrCompileInference(_aiHiddenSeed, () =>
        {
            var o = _engine.TensorMatMul(_aiHiddenSeed, _aiW2);
            return _engine.TensorBroadcastAdd(o, _aiB2);
        });

        _torchInput = torch.randn(BatchSize, Hidden);
        var torchW1 = torch.randn(Hidden, Hidden);
        var torchW2 = torch.randn(Hidden, Hidden);
        var torchB1 = torch.randn(Hidden);
        var torchB2 = torch.randn(Hidden);
        var torchW1T = torchW1.t().contiguous();
        var torchW2T = torchW2.t().contiguous();
        var lin1 = torch.nn.Linear(Hidden, Hidden);
        lin1.weight = new TorchSharp.Modules.Parameter(torchW1T);
        lin1.bias = new TorchSharp.Modules.Parameter(torchB1);
        var lin2 = torch.nn.Linear(Hidden, Hidden);
        lin2.weight = new TorchSharp.Modules.Parameter(torchW2T);
        lin2.bias = new TorchSharp.Modules.Parameter(torchB2);
        _torchUnet = torch.nn.Sequential(("lin1", lin1), ("relu", torch.nn.ReLU()), ("lin2", lin2));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // Dispose TorchSharp tensors / module — they hold native
        // libtorch memory the GC won't reclaim. Closes review-comment
        // #298.75AR.
        _torchInput?.Dispose();
        _torchUnet?.Dispose();

        _cacheA?.Dispose();
        _cacheB?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public void PyTorch_DenoisingLoop()
    {
        // 50 sequential forward passes through the noise predictor —
        // closest available analog to a real diffusion sampling loop
        // (without time-step conditioning, which would only add a
        // constant per-step cost the same on both sides). Wrapped in
        // torch.no_grad() so the measurement is pure inference,
        // matching the AiDotNet ChainAsync path. Closes #298.1Sbh.
        using var _noGrad = torch.no_grad();
        for (int step = 0; step < DenoisingSteps; step++)
        {
            using var output = _torchUnet.forward(_torchInput);
        }
    }

    [Benchmark]
    public async Task Tensors_DenoisingLoop_ChainAsync()
    {
        // The new path: 50× ChainAsync over the same compiled plan
        // pair. Boundary tensor is rebound on the first iteration and
        // stays bound for subsequent steps (idempotent rebind per
        // #296 contract). On CPU each call inlines on the calling
        // thread — no async state machine, no channel write, no
        // Task.Run.
        for (int step = 0; step < DenoisingSteps; step++)
        {
            _planA.SetInputs(_aiInputArray);
            await _planA.ChainAsync(_planB).ConfigureAwait(false);
        }
    }
}
#endif
