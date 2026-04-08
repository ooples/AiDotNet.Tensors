#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Apples-to-apples comparison of AiDotNet vs TorchSharp (libtorch via P/Invoke).
///
/// Each benchmark group compares EQUIVALENT operations:
/// - Both sides do the same math (matmul+bias+relu or raw matmul+relu)
/// - Both sides use eager execution (no graph compilation)
/// - Compiled plan benchmarks are shown separately as AiDotNet-only speedup
///
/// TorchSharp does NOT support torch.compile/jit, so torch.nn.Sequential
/// with nn.Linear modules is the highest-level API available.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class AutogradComparisonBenchmarks
{
    private CpuEngine _engine = null!;

    // AiDotNet tensors
    private Tensor<float> _aiInput = null!;
    private Tensor<float> _aiW1 = null!;
    private Tensor<float> _aiW2 = null!;
    private Tensor<float> _aiW3 = null!;
    private Tensor<float> _aiB1 = null!;
    private Tensor<float> _aiB2 = null!;
    private Tensor<float> _aiB3 = null!;
    private Tensor<float> _aiSmallA = null!;
    private Tensor<float> _aiSmallB = null!;

    // TorchSharp tensors
    private TorchTensor _torchInput = null!;
    private TorchTensor _torchW1 = null!;
    private TorchTensor _torchW2 = null!;
    private TorchTensor _torchW3 = null!;
    private TorchTensor _torchB1 = null!;
    private TorchTensor _torchB2 = null!;
    private TorchTensor _torchB3 = null!;
    private TorchTensor _torchSmallA = null!;
    private TorchTensor _torchSmallB = null!;

    // TorchSharp nn.Sequential module (highest-level API available)
    private TorchSharp.Modules.Sequential _torchMlpModule = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        // MLP tensors: 32x128 -> 64 -> 32 -> 10
        _aiInput = Tensor<float>.CreateRandom([32, 128]);
        _aiW1 = Tensor<float>.CreateRandom([128, 64]);
        _aiW2 = Tensor<float>.CreateRandom([64, 32]);
        _aiW3 = Tensor<float>.CreateRandom([32, 10]);
        _aiB1 = Tensor<float>.CreateRandom([64]);
        _aiB2 = Tensor<float>.CreateRandom([32]);
        _aiB3 = Tensor<float>.CreateRandom([10]);

        // Small tensors for recording overhead measurement
        _aiSmallA = Tensor<float>.CreateRandom([64]);
        _aiSmallB = Tensor<float>.CreateRandom([64]);

        // TorchSharp equivalents — same shapes
        _torchInput = torch.randn([32, 128], requires_grad: false);
        _torchW1 = torch.randn([128, 64], requires_grad: true);
        _torchW2 = torch.randn([64, 32], requires_grad: true);
        _torchW3 = torch.randn([32, 10], requires_grad: true);
        _torchB1 = torch.randn([64], requires_grad: true);
        _torchB2 = torch.randn([32], requires_grad: true);
        _torchB3 = torch.randn([10], requires_grad: true);
        _torchSmallA = torch.randn([64], requires_grad: true);
        _torchSmallB = torch.randn([64], requires_grad: true);

        // TorchSharp nn.Sequential — the highest-level API available in TorchSharp.
        // This uses nn.Linear (matmul + bias) internally, matching AiDotNet FusedLinear.
        _torchMlpModule = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _torchInput?.Dispose();
        _torchW1?.Dispose(); _torchW2?.Dispose(); _torchW3?.Dispose();
        _torchB1?.Dispose(); _torchB2?.Dispose(); _torchB3?.Dispose();
        _torchSmallA?.Dispose(); _torchSmallB?.Dispose();
        _torchMlpModule?.Dispose();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 1. SINGLE OP: Add forward + backward (measures per-op overhead)
    // Both sides: add two [64] tensors, sum, backward. Fair comparison.
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: Add[64] forward+backward")]
    public Tensor<float> AiDotNet_Add64_ForwardBackward()
    {
        using var tape = new GradientTape<float>();
        var c = _engine.TensorAdd(_aiSmallA, _aiSmallB);
        var loss = _engine.ReduceSum(c, null);
        var grads = tape.ComputeGradients(loss, new[] { _aiSmallA });
        return grads[_aiSmallA];
    }

    [Benchmark(Description = "TorchSharp Eager: Add[64] forward+backward")]
    public TorchTensor TorchSharp_Add64_ForwardBackward()
    {
        _torchSmallA.grad?.zero_();
        var c = _torchSmallA + _torchSmallB;
        var loss = c.sum();
        loss.backward();
        return _torchSmallA.grad!;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. RAW MATMUL: forward + backward (no bias — identical ops)
    // Both sides: matmul [32x128]@[128x64], sum, backward. Fair.
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: MatMul[32x128,128x64] forward+backward")]
    public Tensor<float> AiDotNet_MatMul_ForwardBackward()
    {
        using var tape = new GradientTape<float>();
        var h = _engine.TensorMatMul(_aiInput, _aiW1);
        var loss = _engine.ReduceSum(h, null);
        var grads = tape.ComputeGradients(loss, new[] { _aiW1 });
        return grads[_aiW1];
    }

    [Benchmark(Description = "TorchSharp Eager: MatMul[32x128,128x64] forward+backward")]
    public TorchTensor TorchSharp_MatMul_ForwardBackward()
    {
        _torchW1.grad?.zero_();
        var h = torch.matmul(_torchInput, _torchW1);
        var loss = h.sum();
        loss.backward();
        return _torchW1.grad!;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3a. MLP RAW MATMUL: Both sides use matmul+relu (no bias). Fair.
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: MLP-NoBias[32x128->64->32->10] train")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MLP_NoBias_TrainStep()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.ReLU(_engine.TensorMatMul(_aiInput, _aiW1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, _aiW2));
        var output = _engine.TensorMatMul(h2, _aiW3);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "TorchSharp Eager: MLP-NoBias[32x128->64->32->10] train")]
    public (TorchTensor, TorchTensor, TorchTensor) TorchSharp_MLP_NoBias_TrainStep()
    {
        _torchW1.grad?.zero_();
        _torchW2.grad?.zero_();
        _torchW3.grad?.zero_();
        var h1 = torch.nn.functional.relu(torch.matmul(_torchInput, _torchW1));
        var h2 = torch.nn.functional.relu(torch.matmul(h1, _torchW2));
        var output = torch.matmul(h2, _torchW3);
        var loss = output.sum();
        loss.backward();
        return (_torchW1.grad!, _torchW2.grad!, _torchW3.grad!);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3b. MLP WITH BIAS: Both sides use linear(matmul+bias)+relu. Fair.
    // AiDotNet uses FusedLinear, TorchSharp uses functional.linear.
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: MLP-WithBias[32x128->64->32->10] train")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MLP_WithBias_TrainStep()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.FusedLinear(_aiInput, _aiW1, _aiB1, FusedActivationType.ReLU);
        var h2 = _engine.FusedLinear(h1, _aiW2, _aiB2, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h2, _aiW3, _aiB3, FusedActivationType.None);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "TorchSharp Eager: MLP-WithBias[32x128->64->32->10] train")]
    public (TorchTensor, TorchTensor, TorchTensor) TorchSharp_MLP_WithBias_TrainStep()
    {
        _torchW1.grad?.zero_(); _torchW2.grad?.zero_(); _torchW3.grad?.zero_();
        _torchB1.grad?.zero_(); _torchB2.grad?.zero_(); _torchB3.grad?.zero_();
        // functional.linear = matmul + bias, matching AiDotNet FusedLinear
        // Note: functional.linear expects weight as [out, in], so we transpose
        var h1 = torch.nn.functional.relu(torch.nn.functional.linear(_torchInput, _torchW1.t(), _torchB1));
        var h2 = torch.nn.functional.relu(torch.nn.functional.linear(h1, _torchW2.t(), _torchB2));
        var output = torch.nn.functional.linear(h2, _torchW3.t(), _torchB3);
        var loss = output.sum();
        loss.backward();
        return (_torchW1.grad!, _torchW2.grad!, _torchW3.grad!);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3c. MLP nn.Sequential: TorchSharp's highest-level API (nn.Linear)
    // This is the closest TorchSharp equivalent to a "model" abstraction.
    // AiDotNet doesn't have an nn.Module equivalent, so FusedLinear is
    // the closest comparison. Both use matmul+bias+activation internally.
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "TorchSharp Eager: MLP-nn.Sequential[32x128->64->32->10] train")]
    public void TorchSharp_MLP_Sequential_TrainStep()
    {
        _torchMlpModule.zero_grad();
        var output = _torchMlpModule.call(_torchInput);
        var loss = output.sum();
        loss.backward();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. INFERENCE: No-grad / no-tape (verify zero overhead)
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: MatMul[32x128,128x64] inference (no tape)")]
    public Tensor<float> AiDotNet_MatMul_Inference()
    {
        return _engine.TensorMatMul(_aiInput, _aiW1);
    }

    [Benchmark(Description = "TorchSharp Eager: MatMul[32x128,128x64] inference (no_grad)")]
    public TorchTensor TorchSharp_MatMul_Inference()
    {
        using var _ = torch.no_grad();
        return torch.matmul(_torchInput, _torchW1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. COMPILED PLANS: AiDotNet-only (no TorchSharp equivalent)
    // Shows speedup from graph compilation over AiDotNet eager.
    // NOT compared against TorchSharp — that would be unfair.
    // ═══════════════════════════════════════════════════════════════════

    private Engines.Compilation.CompiledInferencePlan<float>? _compiledMlpPlan;
    private Engines.Compilation.CompiledInferencePlan<float>? _compiledChainPlan;

    [IterationSetup(Target = nameof(AiDotNet_MLP_CompiledPlan))]
    public void SetupMlpPlan()
    {
        if (_compiledMlpPlan is not null) return;
        using var scope = Engines.Compilation.GraphMode.Enable();
        var h1 = _engine.ReLU(_engine.TensorMatMul(_aiInput, _aiW1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, _aiW2));
        _engine.TensorMatMul(h2, _aiW3);
        _compiledMlpPlan = scope.CompileInference<float>();
    }

    [Benchmark(Description = "AiDotNet Compiled: MLP inference (vs AiDotNet Eager above)")]
    public Tensor<float> AiDotNet_MLP_CompiledPlan()
    {
        return _compiledMlpPlan!.Execute();
    }

    [IterationSetup(Target = nameof(AiDotNet_ElementwiseChain_CompiledPlan))]
    public void SetupChainPlan()
    {
        if (_compiledChainPlan is not null) return;
        using var scope = Engines.Compilation.GraphMode.Enable();
        var r = _engine.TensorAdd(_aiSmallA, _aiSmallB);
        r = _engine.TensorMultiply(r, _aiSmallA);
        r = _engine.ReLU(r);
        r = _engine.TensorSubtract(r, _aiSmallB);
        _engine.ReLU(r);
        _compiledChainPlan = scope.CompileInference<float>();
    }

    [Benchmark(Description = "AiDotNet Compiled: 5-op chain[64] (vs TorchSharp Eager below)")]
    public Tensor<float> AiDotNet_ElementwiseChain_CompiledPlan()
    {
        return _compiledChainPlan!.Execute();
    }

    [Benchmark(Description = "TorchSharp Eager: 5-op chain[64] (no_grad)")]
    public TorchTensor TorchSharp_ElementwiseChain()
    {
        using var _ = torch.no_grad();
        var r = _torchSmallA + _torchSmallB;
        r = r * _torchSmallA;
        r = torch.nn.functional.relu(r);
        r = r - _torchSmallB;
        return torch.nn.functional.relu(r);
    }

    private Engines.Compilation.CompiledTrainingPlan<float>? _compiledTrainPlan;

    [IterationSetup(Target = nameof(AiDotNet_MLP_CompiledTrainStep))]
    public void SetupTrainPlan()
    {
        if (_compiledTrainPlan is not null) return;
        using var scope = Engines.Compilation.GraphMode.Enable();
        var h1 = _engine.ReLU(_engine.TensorMatMul(_aiInput, _aiW1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, _aiW2));
        var output = _engine.TensorMatMul(h2, _aiW3);
        _compiledTrainPlan = scope.CompileTraining(new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "AiDotNet Compiled: MLP train step (vs AiDotNet Eager above)")]
    public Tensor<float> AiDotNet_MLP_CompiledTrainStep()
    {
        return _compiledTrainPlan!.Step();
    }
}
#endif
