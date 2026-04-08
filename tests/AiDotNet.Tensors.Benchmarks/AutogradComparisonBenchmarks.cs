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
/// Head-to-head comparison of AiDotNet gradient tape vs PyTorch autograd.
/// Measures full forward+backward training steps to compare real-world performance.
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
    private TorchTensor _torchSmallA = null!;
    private TorchTensor _torchSmallB = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        // MLP tensors
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

        // TorchSharp equivalents
        _torchInput = torch.randn([32, 128], requires_grad: false);
        _torchW1 = torch.randn([128, 64], requires_grad: true);
        _torchW2 = torch.randn([64, 32], requires_grad: true);
        _torchW3 = torch.randn([32, 10], requires_grad: true);
        _torchSmallA = torch.randn([64], requires_grad: true);
        _torchSmallB = torch.randn([64], requires_grad: true);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _torchInput?.Dispose();
        _torchW1?.Dispose();
        _torchW2?.Dispose();
        _torchW3?.Dispose();
        _torchSmallA?.Dispose();
        _torchSmallB?.Dispose();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Recording Overhead: Single op forward + backward
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet: Add[64] forward+backward")]
    public Tensor<float> AiDotNet_Add64_ForwardBackward()
    {
        using var tape = new GradientTape<float>();
        var c = _engine.TensorAdd(_aiSmallA, _aiSmallB);
        var loss = _engine.ReduceSum(c, null);
        var grads = tape.ComputeGradients(loss, new[] { _aiSmallA });
        return grads[_aiSmallA];
    }

    [Benchmark(Description = "TorchSharp Eager: Add[64] forward+backward")]
    public TorchTensor PyTorch_Add64_ForwardBackward()
    {
        _torchSmallA.grad?.zero_();
        var c = _torchSmallA + _torchSmallB;
        var loss = c.sum();
        loss.backward();
        return _torchSmallA.grad!;
    }

    // ═══════════════════════════════════════════════════════════════════
    // MatMul: Single matmul forward + backward
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet: MatMul[32x128,128x64] forward+backward")]
    public Tensor<float> AiDotNet_MatMul_ForwardBackward()
    {
        using var tape = new GradientTape<float>();
        var h = _engine.TensorMatMul(_aiInput, _aiW1);
        var loss = _engine.ReduceSum(h, null);
        var grads = tape.ComputeGradients(loss, new[] { _aiW1 });
        return grads[_aiW1];
    }

    [Benchmark(Description = "TorchSharp Eager: MatMul[32x128,128x64] forward+backward")]
    public TorchTensor PyTorch_MatMul_ForwardBackward()
    {
        _torchW1.grad?.zero_();
        var h = torch.matmul(_torchInput, _torchW1);
        var loss = h.sum();
        loss.backward();
        return _torchW1.grad!;
    }

    // ═══════════════════════════════════════════════════════════════════
    // MLP 3-layer: Full training step (forward + backward + gradient)
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet: MLP[32x128->64->32->10] train step")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MLP_TrainStep()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.ReLU(_engine.TensorMatMul(_aiInput, _aiW1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, _aiW2));
        var output = _engine.TensorMatMul(h2, _aiW3);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "AiDotNet: MLP FusedLinear[32x128->64->32->10] train step")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MLP_FusedLinear_TrainStep()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.FusedLinear(_aiInput, _aiW1, _aiB1, FusedActivationType.ReLU);
        var h2 = _engine.FusedLinear(h1, _aiW2, _aiB2, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h2, _aiW3, _aiB3, FusedActivationType.None);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "TorchSharp Eager: MLP[32x128->64->32->10] train step")]
    public (TorchTensor, TorchTensor, TorchTensor) PyTorch_MLP_TrainStep()
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
    // Inference (no grad): Verify zero overhead when tape is inactive
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet: MatMul[32x128,128x64] inference (no tape)")]
    public Tensor<float> AiDotNet_MatMul_Inference()
    {
        return _engine.TensorMatMul(_aiInput, _aiW1);
    }

    [Benchmark(Description = "TorchSharp Eager: MatMul[32x128,128x64] inference (no_grad)")]
    public TorchTensor PyTorch_MatMul_Inference()
    {
        using var _ = torch.no_grad();
        return torch.matmul(_torchInput, _torchW1);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Compiled Plan: Graph compiler + pre-allocated buffer replay
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

    [Benchmark(Description = "AiDotNet: MLP[32x128->64->32->10] compiled plan inference")]
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

    [Benchmark(Description = "AiDotNet: 5-op chain[64] compiled plan")]
    public Tensor<float> AiDotNet_ElementwiseChain_CompiledPlan()
    {
        return _compiledChainPlan!.Execute();
    }

    [Benchmark(Description = "TorchSharp Eager: 5-op chain[64] (no_grad)")]
    public TorchTensor PyTorch_ElementwiseChain()
    {
        using var _ = torch.no_grad();
        var r = _torchSmallA + _torchSmallB;
        r = r * _torchSmallA;
        r = torch.nn.functional.relu(r);
        r = r - _torchSmallB;
        return torch.nn.functional.relu(r);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Compiled Training Plan: compile-once forward+backward replay
    // This is the revolutionary part — replaces tape entirely
    // ═══════════════════════════════════════════════════════════════════

    private Engines.Compilation.CompiledTrainingPlan<float>? _compiledTrainPlan;

    [IterationSetup(Target = nameof(AiDotNet_MLP_CompiledTrainStep))]
    public void SetupTrainPlan()
    {
        if (_compiledTrainPlan is not null) return;
        // Compile once: record the forward graph, auto-generate backward
        using var scope = Engines.Compilation.GraphMode.Enable();
        var h1 = _engine.ReLU(_engine.TensorMatMul(_aiInput, _aiW1));
        var h2 = _engine.ReLU(_engine.TensorMatMul(h1, _aiW2));
        var output = _engine.TensorMatMul(h2, _aiW3);
        _compiledTrainPlan = scope.CompileTraining(new[] { _aiW1, _aiW2, _aiW3 });
    }

    [Benchmark(Description = "AiDotNet: MLP[32x128->64->32->10] compiled train step (fwd+bwd)")]
    public Tensor<float> AiDotNet_MLP_CompiledTrainStep()
    {
        return _compiledTrainPlan!.Step();
    }
}
#endif
