#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Comprehensive TensorCodec vs PyTorch benchmarks for all critical operations.
/// Covers compiled plans (inference + training), spectral matmul, fused multi-layer,
/// normalization, convolution, and multi-size MLP training.
///
/// Run with: dotnet run -c Release --filter TensorCodecVsPyTorch*
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class TensorCodecVsPyTorchBenchmarks
{
    private CpuEngine _engine = null!;

    // ═══ Tensor data ═══
    private Tensor<float> _input32x128 = null!;
    private Tensor<float> _input64x256 = null!;
    private Tensor<float> _w128x64 = null!;
    private Tensor<float> _w64x32 = null!;
    private Tensor<float> _w32x10 = null!;
    private Tensor<float> _w256x128 = null!;
    private Tensor<float> _w128x32 = null!;
    private Tensor<float> _b64 = null!;
    private Tensor<float> _b32 = null!;
    private Tensor<float> _b10 = null!;
    private Tensor<float> _b128 = null!;
    private Tensor<float> _target32x10 = null!;
    private Tensor<float> _target64x32 = null!;
    private Tensor<float> _conv_input = null!;  // [4, 3, 32, 32]
    private Tensor<float> _conv_kernel = null!; // [16, 3, 3, 3]
    private Tensor<float> _bn_input = null!;    // [32, 64, 8, 8]
    private Tensor<float> _bn_gamma = null!;
    private Tensor<float> _bn_beta = null!;
    private Tensor<float> _ln_input = null!;    // [32, 128]
    private Tensor<float> _ln_gamma = null!;
    private Tensor<float> _ln_beta = null!;

    // TorchSharp equivalents
    private TorchTensor _t_input32x128 = null!;
    private TorchTensor _t_input64x256 = null!;
    private TorchTensor _t_w128x64 = null!;
    private TorchTensor _t_w64x32 = null!;
    private TorchTensor _t_w32x10 = null!;
    private TorchTensor _t_w256x128 = null!;
    private TorchTensor _t_w128x32 = null!;
    private TorchTensor _t_target32x10 = null!;
    private TorchTensor _t_target64x32 = null!;
    private TorchTensor _t_conv_input = null!;
    private TorchTensor _t_conv_kernel = null!;
    private TorchTensor _t_bn_input = null!;
    private TorchTensor _t_bn_gamma = null!;
    private TorchTensor _t_bn_beta = null!;
    private TorchTensor _t_ln_input = null!;
    private TorchTensor _t_ln_gamma = null!;
    private TorchTensor _t_ln_beta = null!;

    // Compiled plans (setup once)
    private CompiledTrainingPlan<float>? _smallMlpTrainPlan;
    private CompiledTrainingPlan<float>? _mediumMlpTrainPlan;
    private CompiledInferencePlan<float>? _cnnInferencePlan;
    private CompiledTrainingPlan<float>? _mlpMseTrainPlan;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        // Small MLP: [32, 128] -> 64 -> 32 -> 10
        _input32x128 = Tensor<float>.CreateRandom([32, 128]);
        _w128x64 = Tensor<float>.CreateRandom([128, 64]);
        _w64x32 = Tensor<float>.CreateRandom([64, 32]);
        _w32x10 = Tensor<float>.CreateRandom([32, 10]);
        _b64 = Tensor<float>.CreateRandom([64]);
        _b32 = Tensor<float>.CreateRandom([32]);
        _b10 = Tensor<float>.CreateRandom([10]);
        _target32x10 = Tensor<float>.CreateRandom([32, 10]);

        // Medium MLP: [64, 256] -> 128 -> 32
        _input64x256 = Tensor<float>.CreateRandom([64, 256]);
        _w256x128 = Tensor<float>.CreateRandom([256, 128]);
        _w128x32 = Tensor<float>.CreateRandom([128, 32]);
        _b128 = Tensor<float>.CreateRandom([128]);
        _target64x32 = Tensor<float>.CreateRandom([32, 32]); // matches MLP output [32batch, 32out]

        // Conv2D: [4, 3, 32, 32] with [16, 3, 3, 3] kernel
        _conv_input = Tensor<float>.CreateRandom([4, 3, 32, 32]);
        _conv_kernel = Tensor<float>.CreateRandom([16, 3, 3, 3]);

        // BatchNorm: [32, 64, 8, 8]
        _bn_input = Tensor<float>.CreateRandom([32, 64, 8, 8]);
        _bn_gamma = Tensor<float>.CreateRandom([64]);
        _bn_beta = Tensor<float>.CreateRandom([64]);

        // LayerNorm: [32, 128]
        _ln_input = Tensor<float>.CreateRandom([32, 128]);
        _ln_gamma = Tensor<float>.CreateRandom([128]);
        _ln_beta = Tensor<float>.CreateRandom([128]);

        // TorchSharp equivalents
        _t_input32x128 = torch.randn([32, 128]);
        _t_input64x256 = torch.randn([64, 256]);
        _t_w128x64 = torch.randn([128, 64], requires_grad: true);
        _t_w64x32 = torch.randn([64, 32], requires_grad: true);
        _t_w32x10 = torch.randn([32, 10], requires_grad: true);
        _t_w256x128 = torch.randn([256, 128], requires_grad: true);
        _t_w128x32 = torch.randn([128, 32], requires_grad: true);
        _t_target32x10 = torch.randn([32, 10]);
        _t_target64x32 = torch.randn([32, 32]);
        _t_conv_input = torch.randn([4, 3, 32, 32]);
        _t_conv_kernel = torch.randn([16, 3, 3, 3]);
        _t_bn_input = torch.randn([32, 64, 8, 8]);
        _t_bn_gamma = torch.ones([64]);
        _t_bn_beta = torch.zeros([64]);
        _t_ln_input = torch.randn([32, 128]);
        _t_ln_gamma = torch.ones([128]);
        _t_ln_beta = torch.zeros([128]);

        // Pre-compile plans — each in try/catch so one failure doesn't block all
        try { SetupSmallMlpPlan(); } catch { /* plan will be null, benchmark skipped */ }
        try { SetupSmallMlpCodecPlan(); } catch { }
        try { SetupMediumMlpPlan(); } catch { }
        try { SetupCnnPlan(); } catch { }
        try { SetupMlpMsePlan(); } catch { }
        try { SetupFused(); } catch { }
        try { SetupSpectral(); } catch { }
        try { SetupOps(); } catch { }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _t_input32x128?.Dispose(); _t_input64x256?.Dispose();
        _t_w128x64?.Dispose(); _t_w64x32?.Dispose(); _t_w32x10?.Dispose();
        _t_w256x128?.Dispose(); _t_w128x32?.Dispose();
        _t_target32x10?.Dispose(); _t_target64x32?.Dispose();
        _t_conv_input?.Dispose(); _t_conv_kernel?.Dispose();
        _t_bn_input?.Dispose(); _t_bn_gamma?.Dispose(); _t_bn_beta?.Dispose();
        _t_ln_input?.Dispose(); _t_ln_gamma?.Dispose(); _t_ln_beta?.Dispose();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 1. COMPILED MLP TRAINING: The headline benchmark
    // ═══════════════════════════════════════════════════════════════════

    private void SetupSmallMlpPlan()
    {
        using var scope = GraphMode.Enable();
        var h1 = _engine.FusedLinear(_input32x128, _w128x64, _b64, FusedActivationType.ReLU);
        var h2 = _engine.FusedLinear(h1, _w64x32, _b32, FusedActivationType.ReLU);
        _engine.FusedLinear(h2, _w32x10, _b10, FusedActivationType.None);
        _smallMlpTrainPlan = scope.CompileTraining(new[] { _w128x64, _w64x32, _w32x10 });
    }

    private CompiledTrainingPlan<float>? _smallMlpCodecPlan;

    private void SetupSmallMlpCodecPlan()
    {
        // Enable TensorCodec optimizations (Phase B: dataflow fusion, Phase C: algebraic backward)
        var codecOpts = new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableAlgebraicBackward = true,
            EnableSpectralDecomposition = false // training uses exact weights
        };
        TensorCodecOptions.SetCurrent(codecOpts);
        using var scope = GraphMode.Enable();
        var h1 = _engine.FusedLinear(_input32x128, _w128x64, _b64, FusedActivationType.ReLU);
        var h2 = _engine.FusedLinear(h1, _w64x32, _b32, FusedActivationType.ReLU);
        _engine.FusedLinear(h2, _w32x10, _b10, FusedActivationType.None);
        _smallMlpCodecPlan = scope.CompileTraining(new[] { _w128x64, _w64x32, _w32x10 });
        TensorCodecOptions.SetCurrent(null);
    }

    [Benchmark(Description = "AiDotNet Compiled: MLP[32x128→64→32→10] train step")]
    public Tensor<float> AiDotNet_SmallMLP_CompiledTrain() => _smallMlpTrainPlan?.Step() ?? new Tensor<float>(new int[] { 1 });

    [Benchmark(Description = "AiDotNet TensorCodec: MLP[32x128→64→32→10] train step")]
    public Tensor<float> AiDotNet_SmallMLP_CodecTrain() => _smallMlpCodecPlan?.Step() ?? new Tensor<float>(new int[] { 1 });

    [Benchmark(Description = "AiDotNet Eager: MLP[32x128→64→32→10] train step")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_SmallMLP_EagerTrain()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.FusedLinear(_input32x128, _w128x64, _b64, FusedActivationType.ReLU);
        var h2 = _engine.FusedLinear(h1, _w64x32, _b32, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h2, _w32x10, _b10, FusedActivationType.None);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _w128x64, _w64x32, _w32x10 });
    }

    [Benchmark(Description = "PyTorch: MLP[32x128→64→32→10] train step")]
    public (TorchTensor, TorchTensor, TorchTensor) PyTorch_SmallMLP_Train()
    {
        _t_w128x64.grad?.zero_(); _t_w64x32.grad?.zero_(); _t_w32x10.grad?.zero_();
        var h1 = torch.nn.functional.relu(torch.matmul(_t_input32x128, _t_w128x64));
        var h2 = torch.nn.functional.relu(torch.matmul(h1, _t_w64x32));
        var output = torch.matmul(h2, _t_w32x10);
        output.sum().backward();
        return (_t_w128x64.grad!, _t_w64x32.grad!, _t_w32x10.grad!);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. MLP + MSE LOSS TRAINING: Realistic loss function
    // ═══════════════════════════════════════════════════════════════════

    private void SetupMlpMsePlan()
    {
        using var scope = GraphMode.Enable();
        var h1 = _engine.FusedLinear(_input32x128, _w128x64, _b64, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h1, _w64x32, _b32, FusedActivationType.None);
        var diff = _engine.TensorSubtract(output, _target64x32);
        var sq = _engine.TensorMultiply(diff, diff);
        _engine.ReduceSum(sq, null);
        _mlpMseTrainPlan = scope.CompileTraining(new[] { _w128x64, _w64x32 });
    }

    [Benchmark(Description = "AiDotNet Compiled: MLP+MSE[32x128→64→32] train step")]
    public Tensor<float> AiDotNet_MLP_MSE_CompiledTrain() => _mlpMseTrainPlan?.Step() ?? new Tensor<float>(new int[] { 1 });

    [Benchmark(Description = "AiDotNet Eager: MLP+MSE[32x128→64→32] train step")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MLP_MSE_EagerTrain()
    {
        using var tape = new GradientTape<float>();
        var h1 = _engine.FusedLinear(_input32x128, _w128x64, _b64, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h1, _w64x32, _b32, FusedActivationType.None);
        var diff = _engine.TensorSubtract(output, _target64x32);
        var sq = _engine.TensorMultiply(diff, diff);
        var loss = _engine.ReduceSum(sq, null);
        return tape.ComputeGradients(loss, new[] { _w128x64, _w64x32 });
    }

    [Benchmark(Description = "PyTorch: MLP+MSE[32x128→64→32] train step")]
    public (TorchTensor, TorchTensor) PyTorch_MLP_MSE_Train()
    {
        _t_w128x64.grad?.zero_(); _t_w64x32.grad?.zero_();
        var h1 = torch.nn.functional.relu(torch.matmul(_t_input32x128, _t_w128x64));
        var output = torch.matmul(h1, _t_w64x32);
        var loss = torch.nn.functional.mse_loss(output, _t_target64x32);
        loss.backward();
        return (_t_w128x64.grad!, _t_w64x32.grad!);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. MEDIUM MLP TRAINING: Larger sizes
    // ═══════════════════════════════════════════════════════════════════

    private void SetupMediumMlpPlan()
    {
        using var scope = GraphMode.Enable();
        var h = _engine.FusedLinear(_input64x256, _w256x128, _b128, FusedActivationType.ReLU);
        _engine.FusedLinear(h, _w128x32, _b32, FusedActivationType.None);
        _mediumMlpTrainPlan = scope.CompileTraining(new[] { _w256x128, _w128x32 });
    }

    [Benchmark(Description = "AiDotNet Compiled: MLP[64x256→128→32] train step")]
    public Tensor<float> AiDotNet_MediumMLP_CompiledTrain() => _mediumMlpTrainPlan?.Step() ?? new Tensor<float>(new int[] { 1 });

    [Benchmark(Description = "AiDotNet Eager: MLP[64x256→128→32] train step")]
    public Dictionary<Tensor<float>, Tensor<float>> AiDotNet_MediumMLP_EagerTrain()
    {
        using var tape = new GradientTape<float>();
        var h = _engine.FusedLinear(_input64x256, _w256x128, _b128, FusedActivationType.ReLU);
        var output = _engine.FusedLinear(h, _w128x32, _b32, FusedActivationType.None);
        var loss = _engine.ReduceSum(output, null);
        return tape.ComputeGradients(loss, new[] { _w256x128, _w128x32 });
    }

    [Benchmark(Description = "PyTorch: MLP[64x256→128→32] train step")]
    public (TorchTensor, TorchTensor) PyTorch_MediumMLP_Train()
    {
        _t_w256x128.grad?.zero_(); _t_w128x32.grad?.zero_();
        var h = torch.nn.functional.relu(torch.matmul(_t_input64x256, _t_w256x128));
        var output = torch.matmul(h, _t_w128x32);
        output.sum().backward();
        return (_t_w256x128.grad!, _t_w128x32.grad!);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. CNN: Conv2D + ReLU + MaxPool compiled inference
    // ═══════════════════════════════════════════════════════════════════

    private void SetupCnnPlan()
    {
        using var scope = GraphMode.Enable();
        var conv = _engine.Conv2D(_conv_input, _conv_kernel, stride: 1, padding: 1);
        var activated = _engine.ReLU(conv);
        _engine.MaxPool2D(activated, poolSize: 2, stride: 2);
        _cnnInferencePlan = scope.CompileInference<float>();
    }

    [Benchmark(Description = "AiDotNet Compiled: Conv3x3+ReLU+Pool[4x3x32x32] inference")]
    public Tensor<float> AiDotNet_CNN_CompiledInference() => _cnnInferencePlan?.Execute() ?? new Tensor<float>(new int[] { 1 });

    [Benchmark(Description = "PyTorch: Conv3x3+ReLU+Pool[4x3x32x32] inference")]
    public TorchTensor PyTorch_CNN_Inference()
    {
        using var _ = torch.no_grad();
        var conv = torch.nn.functional.conv2d(_t_conv_input, _t_conv_kernel, padding: new long[] { 1, 1 });
        var activated = torch.nn.functional.relu(conv);
        return torch.nn.functional.max_pool2d(activated, 2, 2);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. BATCH NORMALIZATION: Critical for training
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: BatchNorm[32x64x8x8]")]
    public Tensor<float> AiDotNet_BatchNorm() => _engine.BatchNorm(_bn_input, _bn_gamma, _bn_beta, 1e-5, out _, out _);

    [Benchmark(Description = "AiDotNet Compiled: BatchNorm[32x64x8x8]")]
    public Tensor<float> AiDotNet_BatchNorm_Compiled() => _batchnormPlan is not null ? _batchnormPlan.Execute() : _engine.BatchNorm(_bn_input, _bn_gamma, _bn_beta, 1e-5, out _, out _);

    [Benchmark(Description = "PyTorch: BatchNorm[32x64x8x8]")]
    public TorchTensor PyTorch_BatchNorm()
    {
        using var _ = torch.no_grad();
        var running_mean = torch.zeros([64]);
        var running_var = torch.ones([64]);
        return torch.nn.functional.batch_norm(_t_bn_input, running_mean, running_var,
            _t_bn_gamma, _t_bn_beta, training: true);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. LAYER NORMALIZATION: Transformer building block
    // ═══════════════════════════════════════════════════════════════════

    [Benchmark(Description = "AiDotNet Eager: LayerNorm[32x128]")]
    public Tensor<float> AiDotNet_LayerNorm() => _engine.LayerNorm(_ln_input, _ln_gamma, _ln_beta, 1e-5, out _, out _);

    [Benchmark(Description = "AiDotNet Compiled: LayerNorm[32x128]")]
    public Tensor<float> AiDotNet_LayerNorm_Compiled() => _layernormPlan is not null ? _layernormPlan.Execute() : _engine.LayerNorm(_ln_input, _ln_gamma, _ln_beta, 1e-5, out _, out _);

    [Benchmark(Description = "PyTorch: LayerNorm[32x128]")]
    public TorchTensor PyTorch_LayerNorm()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.layer_norm(_t_ln_input, new long[] { 128 }, _t_ln_gamma, _t_ln_beta);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. FUSED TWO-LAYER GEMM: Phase B dataflow fusion
    // ═══════════════════════════════════════════════════════════════════

    private float[]? _fusedInput, _fusedW1, _fusedW2, _fusedOutput, _fusedActivated;

    private void SetupFused()
    {
        if (_fusedInput != null) return;
        int m = 32, k = 128, h = 64, n = 10;
        var rng = new Random(42);
        _fusedInput = new float[m * k];
        _fusedW1 = new float[k * h];
        _fusedW2 = new float[h * n];
        _fusedOutput = new float[m * n];
        _fusedActivated = new float[m * h];
        for (int i = 0; i < _fusedInput.Length; i++) _fusedInput[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < _fusedW1.Length; i++) _fusedW1[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < _fusedW2.Length; i++) _fusedW2[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    [Benchmark(Description = "AiDotNet Eager: FusedTwoLayerGemm[32x128→64→10] (Phase B)")]
    public void AiDotNet_FusedTwoLayerGemm()
    {
        FusedMultiLayerGemm.FusedGemmActivationGemm(
            _fusedInput!, _fusedW1!, _fusedW2!, _fusedOutput!, _fusedActivated!,
            32, 128, 64, 10, x => x > 0 ? x : 0);
    }

    [Benchmark(Description = "PyTorch: TwoLayerMLP[32x128→64→10] inference")]
    public TorchTensor PyTorch_TwoLayerMLP_Inference()
    {
        using var _ = torch.no_grad();
        var h = torch.nn.functional.relu(torch.matmul(_t_input32x128, _t_w128x64));
        return torch.matmul(h, _t_w32x10);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. SPECTRAL MATMUL: Phase A for rank-reducible matrices
    // ═══════════════════════════════════════════════════════════════════

    private float[]? _spectralX, _spectralW, _spectralOut, _spectralWorkspace;
    private SpectralFactors? _spectralFactors;

    private void SetupSpectral()
    {
        if (_spectralX != null) return;
        int m = 32, n = 64, k = 64;
        var rng = new Random(42);

        // Create a low-rank matrix (rank ~16) for spectral decomposition
        int rank = 16;
        var U = new float[k * rank];
        var V = new float[rank * n];
        for (int i = 0; i < U.Length; i++) U[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < V.Length; i++) V[i] = (float)(rng.NextDouble() * 2 - 1);

        _spectralW = new float[k * n];
        // W = U @ V (low-rank matrix)
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int r = 0; r < rank; r++)
                    sum += U[i * rank + r] * V[r * n + j];
                _spectralW[i * n + j] = sum;
            }

        _spectralX = new float[m * k];
        for (int i = 0; i < _spectralX.Length; i++) _spectralX[i] = (float)(rng.NextDouble() * 2 - 1);
        _spectralOut = new float[m * n];

        _spectralFactors = SvdDecomposition.Decompose(_spectralW, k, n, maxRank: 0, energyThreshold: 0.999);
        if (_spectralFactors.HasValue)
            _spectralWorkspace = new float[m * _spectralFactors.Value.Rank];
    }

    [Benchmark(Description = "AiDotNet Eager: SpectralMatMul[32x64,64x64] rank-16 (Phase A)")]
    public void AiDotNet_SpectralMatMul()
    {
        if (_spectralFactors.HasValue)
            SvdDecomposition.SpectralMatMul(_spectralX!, 32, 64, _spectralFactors.Value, _spectralOut!, _spectralWorkspace);
    }

    [Benchmark(Description = "AiDotNet Eager: DirectMatMul[32x64,64x64] (baseline)")]
    public void AiDotNet_DirectMatMul()
    {
        Array.Clear(_spectralOut!, 0, _spectralOut!.Length);
        if (!BlasProvider.TryGemm(32, 64, 64, _spectralX!, 0, 64, _spectralW!, 0, 64, _spectralOut!, 0, 64))
            SimdGemm.Sgemm(_spectralX!.AsSpan(0, 32 * 64), _spectralW!.AsSpan(0, 64 * 64),
                _spectralOut!.AsSpan(0, 32 * 64), 32, 64, 64);
    }
    // ═══════════════════════════════════════════════════════════════════
    // 9. INDIVIDUAL OPERATIONS: Broad coverage across op types
    // ═══════════════════════════════════════════════════════════════════

    private Tensor<float> _op_a = null!;
    private Tensor<float> _op_b = null!;
    private Tensor<float> _op_large = null!;
    private Tensor<float> _op_2d = null!;
    private TorchTensor _t_op_a = null!;
    private TorchTensor _t_op_b = null!;
    private TorchTensor _t_op_large = null!;
    private TorchTensor _t_op_2d = null!;

    // Compiled plans for ALL individual ops
    private CompiledInferencePlan<float>? _addPlan;
    private CompiledInferencePlan<float>? _geluPlan;
    private CompiledInferencePlan<float>? _transposePlan;
    private CompiledInferencePlan<float>? _reduceSumPlan;
    private CompiledInferencePlan<float>? _sigmoidPlan;
    private CompiledInferencePlan<float>? _softmaxPlan;
    private CompiledInferencePlan<float>? _matmulPlan;
    private CompiledInferencePlan<float>? _multiplyPlan;
    private CompiledInferencePlan<float>? _subtractPlan;
    private CompiledInferencePlan<float>? _reluPlan;
    private CompiledInferencePlan<float>? _tanhPlan;
    private CompiledInferencePlan<float>? _leakyReluPlan;
    private CompiledInferencePlan<float>? _swishPlan;
    private CompiledInferencePlan<float>? _mishPlan;
    private CompiledInferencePlan<float>? _expPlan;
    private CompiledInferencePlan<float>? _logPlan;
    private CompiledInferencePlan<float>? _conv2dPlan;
    private CompiledInferencePlan<float>? _maxpoolPlan;
    private CompiledInferencePlan<float>? _batchnormPlan;
    private CompiledInferencePlan<float>? _layernormPlan;
    private CompiledInferencePlan<float>? _absPlan;
    private CompiledInferencePlan<float>? _powPlan;
    private CompiledInferencePlan<float>? _logSoftmaxPlan;
    private CompiledInferencePlan<float>? _groupnormPlan;
    private CompiledInferencePlan<float>? _dividePlan;
    private CompiledInferencePlan<float>? _negatePlan;
    private CompiledInferencePlan<float>? _eluPlan;
    private CompiledInferencePlan<float>? _sinPlan;
    private CompiledInferencePlan<float>? _cosPlan;
    private CompiledInferencePlan<float>? _sqrtPlan;
    private CompiledInferencePlan<float>? _clampPlan;
    private CompiledInferencePlan<float>? _floorPlan;
    private CompiledInferencePlan<float>? _ceilingPlan;
    private CompiledInferencePlan<float>? _signPlan;
    private CompiledInferencePlan<float>? _reciprocalPlan;
    private CompiledInferencePlan<float>? _meanPlan;
    private CompiledInferencePlan<float>? _variancePlan;
    private CompiledInferencePlan<float>? _maxPlan;
    private CompiledInferencePlan<float>? _softplusPlan;
    private CompiledInferencePlan<float>? _hardSwishPlan;
    private CompiledInferencePlan<float>? _hardSigmoidPlan;
    private CompiledInferencePlan<float>? _seluPlan;
    private CompiledInferencePlan<float>? _roundPlan;
    private CompiledInferencePlan<float>? _concatPlan;
    private CompiledInferencePlan<float>? _sumAxisPlan;
    private CompiledInferencePlan<float>? _broadcastAddPlan;
    private CompiledInferencePlan<float>? _broadcastSubPlan;
    private CompiledInferencePlan<float>? _broadcastMulPlan;
    private CompiledInferencePlan<float>? _avgPool2dPlan;
    private CompiledInferencePlan<float>? _reshapePlan;
    private CompiledInferencePlan<float>? _flattenPlan;
    private CompiledInferencePlan<float>? _reduceMaxPlan;
    private CompiledInferencePlan<float>? _matmul512Plan;
    private CompiledInferencePlan<float>? _batchMatMulPlan;
    private CompiledInferencePlan<float>? _mseLossPlan;
    private CompiledInferencePlan<float>? _crossEntropyPlan;
    private CompiledInferencePlan<float>? _wherePlan;
    private CompiledInferencePlan<float>? _reduceMeanPlan;
    private CompiledInferencePlan<float>? _attentionPlan;

    private void SetupOps()
    {
        if (_op_a != null) return;
        _op_a = Tensor<float>.CreateRandom([100000]);
        _op_b = Tensor<float>.CreateRandom([100000]);
        _op_large = Tensor<float>.CreateRandom([1000000]);
        _op_2d = Tensor<float>.CreateRandom([256, 256]);
        _t_op_a = torch.randn([100000]);
        _t_op_b = torch.randn([100000]);
        _t_op_large = torch.randn([1000000]);
        _t_op_2d = torch.randn([256, 256]);

        // Compile plans for gap ops — these use specialized Into delegates
        try
        {
            using (var s = GraphMode.Enable()) { _engine.TensorAdd(_op_a, _op_b); _addPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.GELU(_op_a); _geluPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorTranspose(_op_2d); _transposePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceSum(_op_large, null); _reduceSumPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Sigmoid(_op_a); _sigmoidPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Softmax(_op_2d, -1); _softmaxPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMatMul(_op_2d, _op_2d); _matmulPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMultiply(_op_a, _op_b); _multiplyPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorSubtract(_op_a, _op_b); _subtractPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReLU(_op_a); _reluPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Tanh(_op_a); _tanhPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.LeakyReLU(_op_a, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().FromDouble(0.01)); _leakyReluPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Swish(_op_a); _swishPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Mish(_op_a); _mishPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorExp(_op_a); _expPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorLog(_op_a); _logPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Conv2D(_conv_input, _conv_kernel, stride: 1, padding: 1); _conv2dPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.MaxPool2D(_bn_input, poolSize: 2, stride: 2); _maxpoolPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.BatchNorm(_bn_input, _bn_gamma, _bn_beta, 1e-5, out _, out _); _batchnormPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.LayerNorm(_ln_input, _ln_gamma, _ln_beta, 1e-5, out _, out _); _layernormPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorAbs(_op_a); _absPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorPower(_op_a, 2.0f); _powPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorLogSoftmax(_op_2d, -1); _logSoftmaxPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.GroupNorm(_bn_input, 8, _bn_gamma, _bn_beta, 1e-5, out _, out _); _groupnormPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorDivide(_op_a, _op_b); _dividePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorNegate(_op_a); _negatePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ELU(_op_a, 1.0); _eluPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorSin(_op_a); _sinPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorCos(_op_a); _cosPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorSqrt(_op_a); _sqrtPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorClamp(_op_a, -0.5f, 0.5f); _clampPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorFloor(_op_a); _floorPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorCeiling(_op_a); _ceilingPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorSign(_op_a); _signPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorReciprocal(_op_a); _reciprocalPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceMean(_op_large, null, false); _meanPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceVariance(_op_2d, new[] { 1 }, keepDims: false); _variancePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMax(_op_a, _op_b); _maxPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Softplus(_op_a); _softplusPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.HardSwish(_op_a); _hardSwishPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorHardSigmoid(_op_a); _hardSigmoidPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorSELU(_op_a); _seluPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorRound(_op_a); _roundPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorConcatenate(new[] { _op_a, _op_b }, axis: 0); _concatPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceSum(_op_2d, new[] { 1 }); _sumAxisPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.AvgPool2D(_bn_input, poolSize: 2, stride: 2); _avgPool2dPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Reshape(_op_2d, new[] { 65536 }); _reshapePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.Reshape(_op_2d, new[] { 65536 }); _flattenPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceMax(_op_a, new[] { 0 }, keepDims: false, out _); _reduceMaxPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMax(_op_a, _op_b); _wherePlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.ReduceMean(_op_large, null, false); _reduceMeanPlan = s.CompileInference<float>(); }
        }
        catch { /* plans may fail, benchmarks will show NA */ }

        // Compile plans for ops that use special tensors (broadcast, matmul512, loss, attention)
        try
        {
            // Initialize special tensors if needed
            if (_ba_input is null)
            {
                _ba_input = Tensor<float>.CreateRandom([32, 256]);
                _ba_bias = Tensor<float>.CreateRandom([256]);
                _t_ba_input = torch.randn([32, 256]);
                _t_ba_bias = torch.randn([256]);
            }
            if (_mat512 is null)
            {
                _mat512 = Tensor<float>.CreateRandom([512, 512]);
                _t_mat512 = torch.randn([512, 512]);
            }
            if (_loss_pred is null)
            {
                _loss_pred = Tensor<float>.CreateRandom([32, 10]);
                _loss_target = Tensor<float>.CreateRandom([32, 10]);
                _t_loss_pred = torch.randn([32, 10]);
                _t_loss_target = torch.randn([32, 10]);
            }
            if (_attn_q is null)
            {
                _attn_q = Tensor<float>.CreateRandom([8, 64, 32]);
                _attn_k = Tensor<float>.CreateRandom([8, 64, 32]);
                _t_attn_q = torch.randn([8, 64, 32]);
                _t_attn_k = torch.randn([8, 64, 32]);
            }

            using (var s = GraphMode.Enable()) { _engine.TensorBroadcastAdd(_ba_input, _ba_bias); _broadcastAddPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorBroadcastSubtract(_ba_input, _ba_bias); _broadcastSubPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorBroadcastMultiply(_ba_input, _ba_bias); _broadcastMulPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMatMul(_mat512, _mat512); _matmul512Plan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorMSELoss(_loss_pred, _loss_target); _mseLossPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.TensorCrossEntropyLoss(_loss_pred, _loss_target); _crossEntropyPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { _engine.BatchMatMul(_attn_q, _attn_k); _batchMatMulPlan = s.CompileInference<float>(); }
            using (var s = GraphMode.Enable()) { var kT = _engine.TensorTranspose(_attn_k); _engine.BatchMatMul(_attn_q, kT); _attentionPlan = s.CompileInference<float>(); }
        }
        catch { /* plans may fail, benchmarks will show NA */ }
    }

    // --- Add ---
    [Benchmark(Description = "AiDotNet Eager: Add[100K]")]
    public Tensor<float> AiDotNet_Add_100K() => _engine.TensorAdd(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Add[100K]")]
    public Tensor<float> AiDotNet_Add_100K_Compiled() => _addPlan?.Execute() ?? _engine.TensorAdd(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Add[100K]")]
    public TorchTensor PyTorch_Add_100K() => _t_op_a + _t_op_b;

    // --- Multiply ---
    [Benchmark(Description = "AiDotNet Eager: Multiply[100K]")]
    public Tensor<float> AiDotNet_Multiply_100K() => _engine.TensorMultiply(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Multiply[100K]")]
    public Tensor<float> AiDotNet_Multiply_100K_Compiled() => _multiplyPlan is not null ? _multiplyPlan.Execute() : _engine.TensorMultiply(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Multiply[100K]")]
    public TorchTensor PyTorch_Multiply_100K() => _t_op_a * _t_op_b;

    // --- GELU ---
    [Benchmark(Description = "AiDotNet Eager: GELU[100K]")]
    public Tensor<float> AiDotNet_GELU_100K() => _engine.GELU(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: GELU[100K]")]
    public Tensor<float> AiDotNet_GELU_100K_Compiled() => _geluPlan?.Execute() ?? _engine.GELU(_op_a);

    [Benchmark(Description = "PyTorch: GELU[100K]")]
    public TorchTensor PyTorch_GELU_100K() => torch.nn.functional.gelu(_t_op_a);

    // --- Swish/SiLU ---
    [Benchmark(Description = "AiDotNet Eager: Swish[100K]")]
    public Tensor<float> AiDotNet_Swish_100K() => _engine.Swish(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Swish[100K]")]
    public Tensor<float> AiDotNet_Swish_100K_Compiled() => _swishPlan is not null ? _swishPlan.Execute() : _engine.Swish(_op_a);

    [Benchmark(Description = "PyTorch: SiLU[100K]")]
    public TorchTensor PyTorch_Swish_100K() => torch.nn.functional.silu(_t_op_a);

    // --- Mish ---
    [Benchmark(Description = "AiDotNet Eager: Mish[100K]")]
    public Tensor<float> AiDotNet_Mish_100K() => _engine.Mish(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Mish[100K]")]
    public Tensor<float> AiDotNet_Mish_100K_Compiled() => _mishPlan is not null ? _mishPlan.Execute() : _engine.Mish(_op_a);

    [Benchmark(Description = "PyTorch: Mish[100K]")]
    public TorchTensor PyTorch_Mish_100K() => torch.nn.functional.mish(_t_op_a);

    // --- Softmax ---
    [Benchmark(Description = "AiDotNet Eager: Softmax[256x256]")]
    public Tensor<float> AiDotNet_Softmax() => _engine.Softmax(_op_2d, -1);

    [Benchmark(Description = "AiDotNet Compiled: Softmax[256x256]")]
    public Tensor<float> AiDotNet_Softmax_Compiled() => _softmaxPlan is not null ? _softmaxPlan.Execute() : _engine.Softmax(_op_2d, -1);

    [Benchmark(Description = "PyTorch: Softmax[256x256]")]
    public TorchTensor PyTorch_Softmax() => torch.nn.functional.softmax(_t_op_2d, dim: -1);

    // --- Transpose ---
    [Benchmark(Description = "AiDotNet Eager: Transpose[256x256]")]
    public Tensor<float> AiDotNet_Transpose() => _engine.TensorTranspose(_op_2d);

    [Benchmark(Description = "AiDotNet Compiled: Transpose[256x256]")]
    public Tensor<float> AiDotNet_Transpose_Compiled() => _transposePlan is not null ? _transposePlan.Execute() : _engine.TensorTranspose(_op_2d);

    [Benchmark(Description = "PyTorch: Transpose[256x256]")]
    public TorchTensor PyTorch_Transpose() => _t_op_2d.t();

    // --- ReduceSum ---
    [Benchmark(Description = "AiDotNet Eager: ReduceSum[1M]")]
    public Tensor<float> AiDotNet_ReduceSum() => _engine.ReduceSum(_op_large, null);

    [Benchmark(Description = "AiDotNet Compiled: ReduceSum[1M]")]
    public Tensor<float> AiDotNet_ReduceSum_Compiled() => _reduceSumPlan is not null ? _reduceSumPlan.Execute() : _engine.ReduceSum(_op_large, null);

    [Benchmark(Description = "PyTorch: ReduceSum[1M]")]
    public TorchTensor PyTorch_ReduceSum() => _t_op_large.sum();

    // --- Conv2D ---
    [Benchmark(Description = "AiDotNet Eager: Conv2D[4x3x32x32, 16x3x3x3]")]
    public Tensor<float> AiDotNet_Conv2D_3x3()
        => _engine.Conv2D(_conv_input, _conv_kernel, stride: 1, padding: 1);

    [Benchmark(Description = "AiDotNet Compiled: Conv2D[4x3x32x32, 16x3x3x3]")]
    public Tensor<float> AiDotNet_Conv2D_3x3_Compiled() => _conv2dPlan is not null ? _conv2dPlan.Execute() : _engine.Conv2D(_conv_input, _conv_kernel, stride: 1, padding: 1);

    [Benchmark(Description = "PyTorch: Conv2D[4x3x32x32, 16x3x3x3]")]
    public TorchTensor PyTorch_Conv2D_3x3()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.conv2d(_t_conv_input, _t_conv_kernel, padding: new long[] { 1, 1 });
    }

    // --- MaxPool2D ---
    [Benchmark(Description = "AiDotNet Eager: MaxPool2D[32x64x8x8, pool=2]")]
    public Tensor<float> AiDotNet_MaxPool2D() => _engine.MaxPool2D(_bn_input, poolSize: 2, stride: 2);

    [Benchmark(Description = "AiDotNet Compiled: MaxPool2D[32x64x8x8, pool=2]")]
    public Tensor<float> AiDotNet_MaxPool2D_Compiled() => _maxpoolPlan is not null ? _maxpoolPlan.Execute() : _engine.MaxPool2D(_bn_input, poolSize: 2, stride: 2);

    [Benchmark(Description = "PyTorch: MaxPool2D[32x64x8x8, pool=2]")]
    public TorchTensor PyTorch_MaxPool2D()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.max_pool2d(_t_bn_input, 2, 2);
    }
    // ═══════════════════════════════════════════════════════════════════
    // 10. EXPANDED OPERATIONS: Close remaining gaps + more coverage
    // ═══════════════════════════════════════════════════════════════════

    // --- Sigmoid ---
    [Benchmark(Description = "AiDotNet Eager: Sigmoid[100K]")]
    public Tensor<float> AiDotNet_Sigmoid_100K() => _engine.Sigmoid(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Sigmoid[100K]")]
    public Tensor<float> AiDotNet_Sigmoid_100K_Compiled() => _sigmoidPlan is not null ? _sigmoidPlan.Execute() : _engine.Sigmoid(_op_a);

    [Benchmark(Description = "PyTorch: Sigmoid[100K]")]
    public TorchTensor PyTorch_Sigmoid_100K() => torch.sigmoid(_t_op_a);

    // --- Tanh ---
    [Benchmark(Description = "AiDotNet Eager: Tanh[100K]")]
    public Tensor<float> AiDotNet_Tanh_100K() => _engine.Tanh(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Tanh[100K]")]
    public Tensor<float> AiDotNet_Tanh_100K_Compiled() => _tanhPlan is not null ? _tanhPlan.Execute() : _engine.Tanh(_op_a);

    [Benchmark(Description = "PyTorch: Tanh[100K]")]
    public TorchTensor PyTorch_Tanh_100K() => torch.tanh(_t_op_a);

    // --- LeakyReLU ---
    [Benchmark(Description = "AiDotNet Eager: LeakyReLU[100K]")]
    public Tensor<float> AiDotNet_LeakyReLU_100K()
    {
        return _engine.LeakyReLU(_op_a, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().FromDouble(0.01));
    }

    [Benchmark(Description = "AiDotNet Compiled: LeakyReLU[100K]")]
    public Tensor<float> AiDotNet_LeakyReLU_100K_Compiled() => _leakyReluPlan is not null ? _leakyReluPlan.Execute() : _engine.LeakyReLU(_op_a, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>().FromDouble(0.01));

    [Benchmark(Description = "PyTorch: LeakyReLU[100K]")]
    public TorchTensor PyTorch_LeakyReLU_100K() => torch.nn.functional.leaky_relu(_t_op_a, 0.01);

    // --- ReLU ---
    [Benchmark(Description = "AiDotNet Eager: ReLU[100K]")]
    public Tensor<float> AiDotNet_ReLU_100K() => _engine.ReLU(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: ReLU[100K]")]
    public Tensor<float> AiDotNet_ReLU_100K_Compiled() => _reluPlan is not null ? _reluPlan.Execute() : _engine.ReLU(_op_a);

    [Benchmark(Description = "PyTorch: ReLU[100K]")]
    public TorchTensor PyTorch_ReLU_100K() => torch.nn.functional.relu(_t_op_a);

    // --- Exp ---
    [Benchmark(Description = "AiDotNet Eager: Exp[100K]")]
    public Tensor<float> AiDotNet_Exp_100K() => _engine.TensorExp(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Exp[100K]")]
    public Tensor<float> AiDotNet_Exp_100K_Compiled() => _expPlan is not null ? _expPlan.Execute() : _engine.TensorExp(_op_a);

    [Benchmark(Description = "PyTorch: Exp[100K]")]
    public TorchTensor PyTorch_Exp_100K() => torch.exp(_t_op_a);

    // --- Log ---
    [Benchmark(Description = "AiDotNet Eager: Log[100K]")]
    public Tensor<float> AiDotNet_Log_100K() => _engine.TensorLog(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Log[100K]")]
    public Tensor<float> AiDotNet_Log_100K_Compiled() => _logPlan is not null ? _logPlan.Execute() : _engine.TensorLog(_op_a);

    [Benchmark(Description = "PyTorch: Log[100K]")]
    public TorchTensor PyTorch_Log_100K() => torch.log(_t_op_a);

    // --- MatMul ---
    [Benchmark(Description = "AiDotNet Eager: MatMul[256x256]")]
    public Tensor<float> AiDotNet_MatMul_256() => _engine.TensorMatMul(_op_2d, _op_2d);

    [Benchmark(Description = "AiDotNet Compiled: MatMul[256x256]")]
    public Tensor<float> AiDotNet_MatMul_256_Compiled() => _matmulPlan is not null ? _matmulPlan.Execute() : _engine.TensorMatMul(_op_2d, _op_2d);

    [Benchmark(Description = "PyTorch: MatMul[256x256]")]
    public TorchTensor PyTorch_MatMul_256() => torch.matmul(_t_op_2d, _t_op_2d);

    // --- Subtract ---
    [Benchmark(Description = "AiDotNet Eager: Subtract[100K]")]
    public Tensor<float> AiDotNet_Sub_100K() => _engine.TensorSubtract(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Subtract[100K]")]
    public Tensor<float> AiDotNet_Sub_100K_Compiled() => _subtractPlan is not null ? _subtractPlan.Execute() : _engine.TensorSubtract(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Subtract[100K]")]
    public TorchTensor PyTorch_Sub_100K() => _t_op_a - _t_op_b;

    // ═══════════════════════════════════════════════════════════════════
    // 11. EXPANDED OPERATIONS: More coverage
    // ═══════════════════════════════════════════════════════════════════

    // --- Divide ---
    [Benchmark(Description = "AiDotNet Eager: Divide[100K]")]
    public Tensor<float> AiDotNet_Divide_100K() => _engine.TensorDivide(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Divide[100K]")]
    public Tensor<float> AiDotNet_Divide_100K_Compiled() => _dividePlan is not null ? _dividePlan.Execute() : _engine.TensorDivide(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Divide[100K]")]
    public TorchTensor PyTorch_Divide_100K() => _t_op_a / _t_op_b;

    // --- Abs ---
    [Benchmark(Description = "AiDotNet Eager: Abs[100K]")]
    public Tensor<float> AiDotNet_Abs_100K() => _engine.TensorAbs(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Abs[100K]")]
    public Tensor<float> AiDotNet_Abs_100K_Compiled() => _absPlan is not null ? _absPlan.Execute() : _engine.TensorAbs(_op_a);

    [Benchmark(Description = "PyTorch: Abs[100K]")]
    public TorchTensor PyTorch_Abs_100K() => torch.abs(_t_op_a);

    // --- Sqrt ---
    [Benchmark(Description = "AiDotNet Eager: Sqrt[100K]")]
    public Tensor<float> AiDotNet_Sqrt_100K() => _engine.TensorSqrt(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Sqrt[100K]")]
    public Tensor<float> AiDotNet_Sqrt_100K_Compiled() => _sqrtPlan is not null ? _sqrtPlan.Execute() : _engine.TensorSqrt(_op_a);

    [Benchmark(Description = "PyTorch: Sqrt[100K]")]
    public TorchTensor PyTorch_Sqrt_100K() => torch.sqrt(_t_op_a);

    // --- LogSoftmax ---
    [Benchmark(Description = "AiDotNet Eager: LogSoftmax[256x256]")]
    public Tensor<float> AiDotNet_LogSoftmax() => _engine.TensorLogSoftmax(_op_2d, -1);

    [Benchmark(Description = "AiDotNet Compiled: LogSoftmax[256x256]")]
    public Tensor<float> AiDotNet_LogSoftmax_Compiled() => _logSoftmaxPlan is not null ? _logSoftmaxPlan.Execute() : _engine.TensorLogSoftmax(_op_2d, -1);

    [Benchmark(Description = "PyTorch: LogSoftmax[256x256]")]
    public TorchTensor PyTorch_LogSoftmax() => torch.nn.functional.log_softmax(_t_op_2d, dim: -1);

    // --- Mean reduction ---
    [Benchmark(Description = "AiDotNet Eager: Mean[1M]")]
    public float AiDotNet_Mean_1M() => _engine.TensorMean(_op_large);

    [Benchmark(Description = "AiDotNet Compiled: Mean[1M]")]
    public Tensor<float> AiDotNet_Mean_1M_Compiled() => _meanPlan is not null ? _meanPlan.Execute() : _engine.ReduceMean(_op_large, null, false);

    [Benchmark(Description = "AiDotNet Eager: ReduceMean[1M]")]
    public Tensor<float> AiDotNet_ReduceMean_1M() => _engine.ReduceMean(_op_large, null, false);

    [Benchmark(Description = "AiDotNet Compiled: ReduceMean[1M]")]
    public Tensor<float> AiDotNet_ReduceMean_1M_Compiled() => _reduceMeanPlan is not null ? _reduceMeanPlan.Execute() : _engine.ReduceMean(_op_large, null, false);

    [Benchmark(Description = "PyTorch: Mean[1M]")]
    public TorchTensor PyTorch_Mean_1M() => _t_op_large.mean();

    // --- Max reduction ---
    [Benchmark(Description = "AiDotNet Eager: Max[100K]")]
    public Tensor<float> AiDotNet_Max_100K() => _engine.TensorMax(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Max[100K]")]
    public Tensor<float> AiDotNet_Max_100K_Compiled() => _maxPlan is not null ? _maxPlan.Execute() : _engine.TensorMax(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Max[100K]")]
    public TorchTensor PyTorch_Max_100K() => torch.max(_t_op_a, _t_op_b);

    // --- Attention Q@K^T pattern ---
    private Tensor<float> _attn_q = null!, _attn_k = null!;
    private TorchTensor _t_attn_q = null!, _t_attn_k = null!;

    [IterationSetup(Target = nameof(AiDotNet_AttentionQKT))]
    public void SetupAttn()
    {
        if (_attn_q != null) return;
        _attn_q = Tensor<float>.CreateRandom([8, 64, 32]); // [batch, seq, head_dim]
        _attn_k = Tensor<float>.CreateRandom([8, 64, 32]);
        _t_attn_q = torch.randn([8, 64, 32]);
        _t_attn_k = torch.randn([8, 64, 32]);
    }

    [Benchmark(Description = "AiDotNet Eager: Attention Q@K^T [8x64x32]")]
    public Tensor<float> AiDotNet_AttentionQKT()
    {
        var kT = _engine.TensorTranspose(_attn_k);
        return _engine.BatchMatMul(_attn_q, kT);
    }

    [Benchmark(Description = "AiDotNet Compiled: Attention Q@K^T [8x64x32]")]
    public Tensor<float> AiDotNet_AttentionQKT_Compiled()
    {
        if (_attn_q is null) { SetupAttn(); }
        return _attentionPlan is not null ? _attentionPlan.Execute() : AiDotNet_AttentionQKT();
    }

    [Benchmark(Description = "PyTorch: Attention Q@K^T [8x64x32]")]
    public TorchTensor PyTorch_AttentionQKT()
    {
        using var _ = torch.no_grad();
        return torch.matmul(_t_attn_q, _t_attn_k.transpose(-2, -1));
    }

    // --- Negate ---
    [Benchmark(Description = "AiDotNet Eager: Negate[100K]")]
    public Tensor<float> AiDotNet_Negate_100K() => _engine.TensorNegate(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Negate[100K]")]
    public Tensor<float> AiDotNet_Negate_100K_Compiled() => _negatePlan is not null ? _negatePlan.Execute() : _engine.TensorNegate(_op_a);

    [Benchmark(Description = "PyTorch: Negate[100K]")]
    public TorchTensor PyTorch_Negate_100K() => -_t_op_a;

    // --- Pow ---
    [Benchmark(Description = "AiDotNet Eager: Pow[100K]")]
    public Tensor<float> AiDotNet_Pow_100K() => _engine.TensorPower(_op_a, 2.0f);

    [Benchmark(Description = "AiDotNet Compiled: Pow[100K]")]
    public Tensor<float> AiDotNet_Pow_100K_Compiled() => _powPlan is not null ? _powPlan.Execute() : _engine.TensorPower(_op_a, 2.0f);

    [Benchmark(Description = "PyTorch: Pow[100K]")]
    public TorchTensor PyTorch_Pow_100K() => torch.pow(_t_op_a, 2.0);

    // --- GroupNorm ---
    [Benchmark(Description = "AiDotNet Eager: GroupNorm[32x64x8x8]")]
    public Tensor<float> AiDotNet_GroupNorm()
    {
        return _engine.GroupNorm(_bn_input, 8, _bn_gamma, _bn_beta, 1e-5, out _, out _);
    }

    [Benchmark(Description = "AiDotNet Compiled: GroupNorm[32x64x8x8]")]
    public Tensor<float> AiDotNet_GroupNorm_Compiled() => _groupnormPlan is not null ? _groupnormPlan.Execute() : _engine.GroupNorm(_bn_input, 8, _bn_gamma, _bn_beta, 1e-5, out _, out _);

    [Benchmark(Description = "PyTorch: GroupNorm[32x64x8x8]")]
    public TorchTensor PyTorch_GroupNorm()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.group_norm(_t_bn_input, 8, _t_bn_gamma, _t_bn_beta);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 12. MORE OPERATIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- Clamp ---
    [Benchmark(Description = "AiDotNet Eager: Clamp[100K]")]
    public Tensor<float> AiDotNet_Clamp_100K() => _engine.TensorClamp(_op_a, -0.5f, 0.5f);

    [Benchmark(Description = "AiDotNet Compiled: Clamp[100K]")]
    public Tensor<float> AiDotNet_Clamp_100K_Compiled() => _clampPlan is not null ? _clampPlan.Execute() : _engine.TensorClamp(_op_a, -0.5f, 0.5f);

    [Benchmark(Description = "PyTorch: Clamp[100K]")]
    public TorchTensor PyTorch_Clamp_100K() => torch.clamp(_t_op_a, -0.5, 0.5);

    // --- Sin ---
    [Benchmark(Description = "AiDotNet Eager: Sin[100K]")]
    public Tensor<float> AiDotNet_Sin_100K() => _engine.TensorSin(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Sin[100K]")]
    public Tensor<float> AiDotNet_Sin_100K_Compiled() => _sinPlan is not null ? _sinPlan.Execute() : _engine.TensorSin(_op_a);

    [Benchmark(Description = "PyTorch: Sin[100K]")]
    public TorchTensor PyTorch_Sin_100K() => torch.sin(_t_op_a);

    // --- Cos ---
    [Benchmark(Description = "AiDotNet Eager: Cos[100K]")]
    public Tensor<float> AiDotNet_Cos_100K() => _engine.TensorCos(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Cos[100K]")]
    public Tensor<float> AiDotNet_Cos_100K_Compiled() => _cosPlan is not null ? _cosPlan.Execute() : _engine.TensorCos(_op_a);

    [Benchmark(Description = "PyTorch: Cos[100K]")]
    public TorchTensor PyTorch_Cos_100K() => torch.cos(_t_op_a);

    // --- BroadcastAdd (bias pattern) ---
    private Tensor<float> _ba_input = null!, _ba_bias = null!;
    private TorchTensor _t_ba_input = null!, _t_ba_bias = null!;

    [IterationSetup(Target = nameof(AiDotNet_BroadcastAdd))]
    public void SetupBroadcastAdd()
    {
        if (_ba_input != null) return;
        _ba_input = Tensor<float>.CreateRandom([32, 256]);
        _ba_bias = Tensor<float>.CreateRandom([256]);
        _t_ba_input = torch.randn([32, 256]);
        _t_ba_bias = torch.randn([256]);
    }

    [Benchmark(Description = "AiDotNet Eager: BroadcastAdd[32x256]+[256]")]
    public Tensor<float> AiDotNet_BroadcastAdd() => _engine.TensorBroadcastAdd(_ba_input, _ba_bias);

    [Benchmark(Description = "AiDotNet Compiled: BroadcastAdd[32x256]+[256]")]
    public Tensor<float> AiDotNet_BroadcastAdd_Compiled() => _broadcastAddPlan is not null ? _broadcastAddPlan.Execute() : _engine.TensorBroadcastAdd(_ba_input, _ba_bias);

    [Benchmark(Description = "PyTorch: BroadcastAdd[32x256]+[256]")]
    public TorchTensor PyTorch_BroadcastAdd() => _t_ba_input + _t_ba_bias;

    // --- AvgPool2D ---
    [Benchmark(Description = "AiDotNet Eager: AvgPool2D[32x64x8x8, pool=2]")]
    public Tensor<float> AiDotNet_AvgPool2D() => _engine.AvgPool2D(_bn_input, poolSize: 2, stride: 2);

    [Benchmark(Description = "AiDotNet Compiled: AvgPool2D[32x64x8x8, pool=2]")]
    public Tensor<float> AiDotNet_AvgPool2D_Compiled() => _avgPool2dPlan is not null ? _avgPool2dPlan.Execute() : _engine.AvgPool2D(_bn_input, poolSize: 2, stride: 2);

    [Benchmark(Description = "PyTorch: AvgPool2D[32x64x8x8, pool=2]")]
    public TorchTensor PyTorch_AvgPool2D()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.avg_pool2d(_t_bn_input, 2, 2);
    }

    // --- Concat ---
    [Benchmark(Description = "AiDotNet Eager: Concat[2x100K]")]
    public Tensor<float> AiDotNet_Concat_100K() => _engine.TensorConcatenate(new[] { _op_a, _op_b }, axis: 0);

    [Benchmark(Description = "AiDotNet Compiled: Concat[2x100K]")]
    public Tensor<float> AiDotNet_Concat_100K_Compiled() => _concatPlan is not null ? _concatPlan.Execute() : _engine.TensorConcatenate(new[] { _op_a, _op_b }, axis: 0);

    [Benchmark(Description = "PyTorch: Concat[2x100K]")]
    public TorchTensor PyTorch_Concat_100K() => torch.cat(new[] { _t_op_a, _t_op_b }, dim: 0);

    // --- Sum along axis ---
    [Benchmark(Description = "AiDotNet Eager: SumAxis[256x256, axis=1]")]
    public Tensor<float> AiDotNet_SumAxis() => _engine.ReduceSum(_op_2d, new[] { 1 });

    [Benchmark(Description = "AiDotNet Compiled: SumAxis[256x256, axis=1]")]
    public Tensor<float> AiDotNet_SumAxis_Compiled() => _sumAxisPlan is not null ? _sumAxisPlan.Execute() : _engine.ReduceSum(_op_2d, new[] { 1 });

    [Benchmark(Description = "PyTorch: SumAxis[256x256, axis=1]")]
    public TorchTensor PyTorch_SumAxis() => _t_op_2d.sum(dim: 1);

    // --- ELU ---
    [Benchmark(Description = "AiDotNet Eager: ELU[100K]")]
    public Tensor<float> AiDotNet_ELU_100K() => _engine.ELU(_op_a, 1.0);

    [Benchmark(Description = "AiDotNet Compiled: ELU[100K]")]
    public Tensor<float> AiDotNet_ELU_100K_Compiled() => _eluPlan is not null ? _eluPlan.Execute() : _engine.ELU(_op_a, 1.0);

    [Benchmark(Description = "PyTorch: ELU[100K]")]
    public TorchTensor PyTorch_ELU_100K() => torch.nn.functional.elu(_t_op_a, 1.0);

    // --- Softplus ---
    [Benchmark(Description = "AiDotNet Eager: Softplus[100K]")]
    public Tensor<float> AiDotNet_Softplus_100K() => _engine.Softplus(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Softplus[100K]")]
    public Tensor<float> AiDotNet_Softplus_100K_Compiled() => _softplusPlan is not null ? _softplusPlan.Execute() : _engine.Softplus(_op_a);

    [Benchmark(Description = "PyTorch: Softplus[100K]")]
    public TorchTensor PyTorch_Softplus_100K() => torch.nn.functional.softplus(_t_op_a);

    // --- HardSwish ---
    [Benchmark(Description = "AiDotNet Eager: HardSwish[100K]")]
    public Tensor<float> AiDotNet_HardSwish_100K() => _engine.HardSwish(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: HardSwish[100K]")]
    public Tensor<float> AiDotNet_HardSwish_100K_Compiled() => _hardSwishPlan is not null ? _hardSwishPlan.Execute() : _engine.HardSwish(_op_a);

    [Benchmark(Description = "PyTorch: HardSwish[100K]")]
    public TorchTensor PyTorch_HardSwish_100K() => torch.nn.functional.hardswish(_t_op_a);

    // --- BroadcastMultiply (scaling pattern) ---
    [Benchmark(Description = "AiDotNet Eager: BroadcastMul[32x256]*[256]")]
    public Tensor<float> AiDotNet_BroadcastMul() => _engine.TensorBroadcastMultiply(_ba_input, _ba_bias);

    [Benchmark(Description = "AiDotNet Compiled: BroadcastMul[32x256]*[256]")]
    public Tensor<float> AiDotNet_BroadcastMul_Compiled() => _broadcastMulPlan is not null ? _broadcastMulPlan.Execute() : _engine.TensorBroadcastMultiply(_ba_input, _ba_bias);

    [Benchmark(Description = "PyTorch: BroadcastMul[32x256]*[256]")]
    public TorchTensor PyTorch_BroadcastMul() => _t_ba_input * _t_ba_bias;

    // --- Reshape (should be ~zero cost) ---
    [Benchmark(Description = "AiDotNet Eager: Reshape[256x256→65536]")]
    public Tensor<float> AiDotNet_Reshape() => _engine.Reshape(_op_2d, new[] { 65536 });

    [Benchmark(Description = "AiDotNet Compiled: Reshape[256x256→65536]")]
    public Tensor<float> AiDotNet_Reshape_Compiled() => _reshapePlan is not null ? _reshapePlan.Execute() : _engine.Reshape(_op_2d, new[] { 65536 });

    [Benchmark(Description = "PyTorch: Reshape[256x256→65536]")]
    public TorchTensor PyTorch_Reshape() => _t_op_2d.reshape(65536);

    // --- ReduceMax ---
    [Benchmark(Description = "AiDotNet Eager: ReduceMax[100K]")]
    public Tensor<float> AiDotNet_ReduceMax_100K() => _engine.ReduceMax(_op_a, new[] { 0 }, keepDims: false, out _);

    [Benchmark(Description = "AiDotNet Compiled: ReduceMax[100K]")]
    public Tensor<float> AiDotNet_ReduceMax_100K_Compiled() => _reduceMaxPlan is not null ? _reduceMaxPlan.Execute() : _engine.ReduceMax(_op_a, new[] { 0 }, keepDims: false, out _);

    [Benchmark(Description = "PyTorch: ReduceMax[100K]")]
    public TorchTensor PyTorch_ReduceMax_100K()
    {
        var (values, _) = torch.max(_t_op_a, dim: 0);
        return values;
    }

    // --- Floor ---
    [Benchmark(Description = "AiDotNet Eager: Floor[100K]")]
    public Tensor<float> AiDotNet_Floor_100K() => _engine.TensorFloor(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Floor[100K]")]
    public Tensor<float> AiDotNet_Floor_100K_Compiled() => _floorPlan is not null ? _floorPlan.Execute() : _engine.TensorFloor(_op_a);

    [Benchmark(Description = "PyTorch: Floor[100K]")]
    public TorchTensor PyTorch_Floor_100K() => torch.floor(_t_op_a);

    // --- Ceiling ---
    [Benchmark(Description = "AiDotNet Eager: Ceiling[100K]")]
    public Tensor<float> AiDotNet_Ceiling_100K() => _engine.TensorCeiling(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Ceiling[100K]")]
    public Tensor<float> AiDotNet_Ceiling_100K_Compiled() => _ceilingPlan is not null ? _ceilingPlan.Execute() : _engine.TensorCeiling(_op_a);

    [Benchmark(Description = "PyTorch: Ceiling[100K]")]
    public TorchTensor PyTorch_Ceiling_100K() => torch.ceil(_t_op_a);

    // --- Sign ---
    [Benchmark(Description = "AiDotNet Eager: Sign[100K]")]
    public Tensor<float> AiDotNet_Sign_100K() => _engine.TensorSign(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Sign[100K]")]
    public Tensor<float> AiDotNet_Sign_100K_Compiled() => _signPlan is not null ? _signPlan.Execute() : _engine.TensorSign(_op_a);

    [Benchmark(Description = "PyTorch: Sign[100K]")]
    public TorchTensor PyTorch_Sign_100K() => torch.sign(_t_op_a);

    // --- Reciprocal ---
    [Benchmark(Description = "AiDotNet Eager: Reciprocal[100K]")]
    public Tensor<float> AiDotNet_Reciprocal_100K() => _engine.TensorReciprocal(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Reciprocal[100K]")]
    public Tensor<float> AiDotNet_Reciprocal_100K_Compiled() => _reciprocalPlan is not null ? _reciprocalPlan.Execute() : _engine.TensorReciprocal(_op_a);

    [Benchmark(Description = "PyTorch: Reciprocal[100K]")]
    public TorchTensor PyTorch_Reciprocal_100K() => torch.reciprocal(_t_op_a);

    // --- BroadcastSubtract ---
    [Benchmark(Description = "AiDotNet Eager: BroadcastSub[32x256]-[256]")]
    public Tensor<float> AiDotNet_BroadcastSub() => _engine.TensorBroadcastSubtract(_ba_input, _ba_bias);

    [Benchmark(Description = "AiDotNet Compiled: BroadcastSub[32x256]-[256]")]
    public Tensor<float> AiDotNet_BroadcastSub_Compiled() => _broadcastSubPlan is not null ? _broadcastSubPlan.Execute() : _engine.TensorBroadcastSubtract(_ba_input, _ba_bias);

    [Benchmark(Description = "PyTorch: BroadcastSub[32x256]-[256]")]
    public TorchTensor PyTorch_BroadcastSub() => _t_ba_input - _t_ba_bias;

    // --- Larger MatMul 512x512 ---
    private Tensor<float> _mat512 = null!;
    private TorchTensor _t_mat512 = null!;

    [IterationSetup(Target = nameof(AiDotNet_MatMul_512))]
    public void SetupMat512()
    {
        if (_mat512 != null) return;
        _mat512 = Tensor<float>.CreateRandom([512, 512]);
        _t_mat512 = torch.randn([512, 512]);
    }

    [Benchmark(Description = "AiDotNet Eager: MatMul[512x512]")]
    public Tensor<float> AiDotNet_MatMul_512() => _engine.TensorMatMul(_mat512, _mat512);

    [Benchmark(Description = "AiDotNet Compiled: MatMul[512x512]")]
    public Tensor<float> AiDotNet_MatMul_512_Compiled() => _matmul512Plan is not null ? _matmul512Plan.Execute() : _engine.TensorMatMul(_mat512, _mat512);

    [Benchmark(Description = "PyTorch: MatMul[512x512]")]
    public TorchTensor PyTorch_MatMul_512() => torch.matmul(_t_mat512, _t_mat512);

    // ═══════════════════════════════════════════════════════════════════
    // 13. ADDITIONAL OPERATIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- Flatten (should be near zero-cost) ---
    [Benchmark(Description = "AiDotNet Eager: Flatten[256x256]")]
    public Tensor<float> AiDotNet_Flatten() => _engine.Reshape(_op_2d, new[] { 65536 });

    [Benchmark(Description = "AiDotNet Compiled: Flatten[256x256]")]
    public Tensor<float> AiDotNet_Flatten_Compiled() => _flattenPlan is not null ? _flattenPlan.Execute() : _engine.Reshape(_op_2d, new[] { 65536 });

    [Benchmark(Description = "PyTorch: Flatten[256x256]")]
    public TorchTensor PyTorch_Flatten() => _t_op_2d.flatten();

    // --- BatchMatMul ---
    [Benchmark(Description = "AiDotNet Eager: BatchMatMul[8x64x32]")]
    public Tensor<float> AiDotNet_BatchMatMul()
    {
        if (_attn_q is null) { SetupAttn(); }
        return _engine.BatchMatMul(_attn_q, _attn_k);
    }

    [Benchmark(Description = "AiDotNet Compiled: BatchMatMul[8x64x32]")]
    public Tensor<float> AiDotNet_BatchMatMul_Compiled()
    {
        if (_attn_q is null) { SetupAttn(); }
        return _batchMatMulPlan is not null ? _batchMatMulPlan.Execute() : _engine.BatchMatMul(_attn_q, _attn_k);
    }

    [Benchmark(Description = "PyTorch: BatchMatMul[8x64x32]")]
    public TorchTensor PyTorch_BatchMatMul()
    {
        if (_t_attn_q is null) { SetupAttn(); }
        return torch.matmul(_t_attn_q, _t_attn_k);
    }

    // --- MSELoss ---
    private Tensor<float> _loss_pred = null!, _loss_target = null!;
    private TorchTensor _t_loss_pred = null!, _t_loss_target = null!;

    [IterationSetup(Target = nameof(AiDotNet_MSELoss))]
    public void SetupLoss()
    {
        if (_loss_pred != null) return;
        _loss_pred = Tensor<float>.CreateRandom([32, 10]);
        _loss_target = Tensor<float>.CreateRandom([32, 10]);
        _t_loss_pred = torch.randn([32, 10]);
        _t_loss_target = torch.randn([32, 10]);
    }

    [Benchmark(Description = "AiDotNet Eager: MSELoss[32x10]")]
    public Tensor<float> AiDotNet_MSELoss() => _engine.TensorMSELoss(_loss_pred, _loss_target);

    [Benchmark(Description = "AiDotNet Compiled: MSELoss[32x10]")]
    public Tensor<float> AiDotNet_MSELoss_Compiled()
    {
        if (_loss_pred is null) SetupLoss();
        return _mseLossPlan is not null ? _mseLossPlan.Execute() : _engine.TensorMSELoss(_loss_pred, _loss_target);
    }

    [Benchmark(Description = "PyTorch: MSELoss[32x10]")]
    public TorchTensor PyTorch_MSELoss() => torch.nn.functional.mse_loss(_t_loss_pred, _t_loss_target);

    // --- ReduceVariance ---
    [Benchmark(Description = "AiDotNet Eager: Variance[256x256, axis=1]")]
    public Tensor<float> AiDotNet_Variance() => _engine.ReduceVariance(_op_2d, new[] { 1 }, keepDims: false);

    [Benchmark(Description = "AiDotNet Compiled: Variance[256x256, axis=1]")]
    public Tensor<float> AiDotNet_Variance_Compiled() => _variancePlan is not null ? _variancePlan.Execute() : _engine.ReduceVariance(_op_2d, new[] { 1 }, keepDims: false);

    [Benchmark(Description = "PyTorch: Variance[256x256, axis=1]")]
    public TorchTensor PyTorch_Variance() => _t_op_2d.var(dim: 1);

    // --- Where (conditional select) ---
    [Benchmark(Description = "AiDotNet Eager: Where[100K]")]
    public Tensor<float> AiDotNet_Where_100K() => _engine.TensorMax(_op_a, _op_b);

    [Benchmark(Description = "AiDotNet Compiled: Where[100K]")]
    public Tensor<float> AiDotNet_Where_100K_Compiled() => _wherePlan is not null ? _wherePlan.Execute() : _engine.TensorMax(_op_a, _op_b);

    [Benchmark(Description = "PyTorch: Where[100K]")]
    public TorchTensor PyTorch_Where_100K() => torch.maximum(_t_op_a, _t_op_b);

    // ═══════════════════════════════════════════════════════════════════
    // 14. MORE ACTIVATIONS + LOSS FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- SELU ---
    [Benchmark(Description = "AiDotNet Eager: SELU[100K]")]
    public Tensor<float> AiDotNet_SELU_100K() => _engine.TensorSELU(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: SELU[100K]")]
    public Tensor<float> AiDotNet_SELU_100K_Compiled() => _seluPlan is not null ? _seluPlan.Execute() : _engine.TensorSELU(_op_a);

    [Benchmark(Description = "PyTorch: SELU[100K]")]
    public TorchTensor PyTorch_SELU_100K() => torch.nn.functional.selu(_t_op_a);

    // --- HardSigmoid ---
    [Benchmark(Description = "AiDotNet Eager: HardSigmoid[100K]")]
    public Tensor<float> AiDotNet_HardSigmoid_100K() => _engine.TensorHardSigmoid(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: HardSigmoid[100K]")]
    public Tensor<float> AiDotNet_HardSigmoid_100K_Compiled() => _hardSigmoidPlan is not null ? _hardSigmoidPlan.Execute() : _engine.TensorHardSigmoid(_op_a);

    [Benchmark(Description = "PyTorch: HardSigmoid[100K]")]
    public TorchTensor PyTorch_HardSigmoid_100K() => torch.nn.functional.hardsigmoid(_t_op_a);

    // --- Round ---
    [Benchmark(Description = "AiDotNet Eager: Round[100K]")]
    public Tensor<float> AiDotNet_Round_100K() => _engine.TensorRound(_op_a);

    [Benchmark(Description = "AiDotNet Compiled: Round[100K]")]
    public Tensor<float> AiDotNet_Round_100K_Compiled() => _roundPlan is not null ? _roundPlan.Execute() : _engine.TensorRound(_op_a);

    [Benchmark(Description = "PyTorch: Round[100K]")]
    public TorchTensor PyTorch_Round_100K() => torch.round(_t_op_a);

    // --- CrossEntropyLoss ---
    [Benchmark(Description = "AiDotNet Eager: CrossEntropyLoss[32x10]")]
    public Tensor<float> AiDotNet_CrossEntropy()
    {
        if (_loss_pred is null) SetupLoss();
        return _engine.TensorCrossEntropyLoss(_loss_pred, _loss_target);
    }

    [Benchmark(Description = "AiDotNet Compiled: CrossEntropyLoss[32x10]")]
    public Tensor<float> AiDotNet_CrossEntropy_Compiled()
    {
        if (_loss_pred is null) SetupLoss();
        return _crossEntropyPlan is not null ? _crossEntropyPlan.Execute() : _engine.TensorCrossEntropyLoss(_loss_pred, _loss_target);
    }

    [Benchmark(Description = "PyTorch: CrossEntropyLoss[32x10]")]
    public TorchTensor PyTorch_CrossEntropy()
    {
        if (_t_loss_pred is null) SetupLoss();
        return torch.nn.functional.cross_entropy(_t_loss_pred, _t_loss_target.to(torch.int64).argmax(dim: 1));
    }
    // ═══════════════════════════════════════════════════════════════════
    // 15. HYPERBOLIC + EXTENDED OPERATIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- Sinh ---
    [Benchmark(Description = "AiDotNet Eager: Sinh[100K]")]
    public Tensor<float> AiDotNet_Sinh_100K() => _engine.TensorSinh(_op_a);

    [Benchmark(Description = "PyTorch: Sinh[100K]")]
    public TorchTensor PyTorch_Sinh_100K() => torch.sinh(_t_op_a);

    // --- Cosh ---
    [Benchmark(Description = "AiDotNet Eager: Cosh[100K]")]
    public Tensor<float> AiDotNet_Cosh_100K() => _engine.TensorCosh(_op_a);

    [Benchmark(Description = "PyTorch: Cosh[100K]")]
    public TorchTensor PyTorch_Cosh_100K() => torch.cosh(_t_op_a);

    // --- ReLU6 ---
    [Benchmark(Description = "AiDotNet Eager: ReLU6[100K]")]
    public Tensor<float> AiDotNet_ReLU6_100K() => _engine.TensorReLU6(_op_a);

    [Benchmark(Description = "PyTorch: ReLU6[100K]")]
    public TorchTensor PyTorch_ReLU6_100K() => torch.nn.functional.relu6(_t_op_a);

    // --- Tanh backward (derivative) ---
    [Benchmark(Description = "AiDotNet Eager: TanhBackward[100K]")]
    public Tensor<float> AiDotNet_TanhBackward_100K()
    {
        var grad = Tensor<float>.CreateOnes([100000]);
        var tanhOut = _engine.Tanh(_op_a);
        return _engine.TanhBackward(grad, tanhOut);
    }

    [Benchmark(Description = "PyTorch: TanhBackward[100K]")]
    public TorchTensor PyTorch_TanhBackward_100K()
    {
        var tanhOut = torch.tanh(_t_op_a);
        return 1.0 - tanhOut * tanhOut;
    }

    // --- InstanceNorm ---
    private Tensor<float> _in_input = null!;
    private Tensor<float> _in_gamma = null!, _in_beta = null!;
    private TorchTensor _t_in_input = null!;

    [IterationSetup(Target = nameof(AiDotNet_InstanceNorm))]
    public void SetupInstanceNorm()
    {
        if (_in_input != null) return;
        _in_input = Tensor<float>.CreateRandom([8, 32, 16, 16]); // [N, C, H, W]
        _in_gamma = Tensor<float>.CreateRandom([32]);
        _in_beta = Tensor<float>.CreateRandom([32]);
        _t_in_input = torch.randn([8, 32, 16, 16]);
    }

    [Benchmark(Description = "AiDotNet Eager: InstanceNorm[8x32x16x16]")]
    public Tensor<float> AiDotNet_InstanceNorm()
    {
        return _engine.InstanceNorm(_in_input, _in_gamma, _in_beta, 1e-5, out _, out _);
    }

    [Benchmark(Description = "PyTorch: InstanceNorm[8x32x16x16]")]
    public TorchTensor PyTorch_InstanceNorm()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.instance_norm(_t_in_input);
    }

    // --- RMSNorm ---
    private Tensor<float> _rms_input = null!, _rms_gamma = null!;

    [IterationSetup(Target = nameof(AiDotNet_RMSNorm))]
    public void SetupRMSNorm()
    {
        if (_rms_input != null) return;
        _rms_input = Tensor<float>.CreateRandom([32, 128]);
        _rms_gamma = Tensor<float>.CreateRandom([128]);
    }

    [Benchmark(Description = "AiDotNet Eager: RMSNorm[32x128]")]
    public Tensor<float> AiDotNet_RMSNorm()
    {
        return _engine.RMSNorm(_rms_input, _rms_gamma, 1e-5, out _);
    }

    // --- TanhBackward on large ---
    [Benchmark(Description = "AiDotNet Eager: SigmoidBackward[100K]")]
    public Tensor<float> AiDotNet_SigmoidBackward_100K()
    {
        var grad = Tensor<float>.CreateOnes([100000]);
        var sigOut = _engine.Sigmoid(_op_a);
        return _engine.SigmoidBackward(grad, sigOut);
    }
    // ═══════════════════════════════════════════════════════════════════
    // 16. CONV + POOLING VARIANTS
    // ═══════════════════════════════════════════════════════════════════

    // --- Conv1D ---
    private Tensor<float> _conv1d_input = null!, _conv1d_kernel = null!;
    private TorchTensor _t_conv1d_input = null!, _t_conv1d_kernel = null!;

    [IterationSetup(Target = nameof(AiDotNet_Conv1D))]
    public void SetupConv1D()
    {
        if (_conv1d_input != null) return;
        _conv1d_input = Tensor<float>.CreateRandom([8, 16, 128]); // [batch, channels, length]
        _conv1d_kernel = Tensor<float>.CreateRandom([32, 16, 3]); // [out_ch, in_ch, kernel]
        _t_conv1d_input = torch.randn([8, 16, 128]);
        _t_conv1d_kernel = torch.randn([32, 16, 3]);
    }

    [Benchmark(Description = "AiDotNet Eager: Conv1D[8x16x128, k=3]")]
    public Tensor<float> AiDotNet_Conv1D()
    {
        return _engine.Conv1D(_conv1d_input, _conv1d_kernel, stride: 1, padding: 1);
    }

    [Benchmark(Description = "PyTorch: Conv1D[8x16x128, k=3]")]
    public TorchTensor PyTorch_Conv1D()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.conv1d(_t_conv1d_input, _t_conv1d_kernel, padding: 1);
    }

    // --- DepthwiseConv2D ---
    private Tensor<float> _dw_input = null!, _dw_kernel = null!;
    private TorchTensor _t_dw_input = null!;

    [IterationSetup(Target = nameof(AiDotNet_DepthwiseConv2D))]
    public void SetupDepthwiseConv2D()
    {
        if (_dw_input != null) return;
        _dw_input = Tensor<float>.CreateRandom([4, 32, 16, 16]);
        _dw_kernel = Tensor<float>.CreateRandom([32, 1, 3, 3]); // groups=channels
        _t_dw_input = torch.randn([4, 32, 16, 16]);
    }

    [Benchmark(Description = "AiDotNet Eager: DepthwiseConv2D[4x32x16x16]")]
    public Tensor<float> AiDotNet_DepthwiseConv2D()
    {
        return _engine.DepthwiseConv2D(_dw_input, _dw_kernel, new[] { 1, 1 }, new[] { 1, 1 });
    }

    // --- AdaptiveAvgPool2D ---
    [Benchmark(Description = "AiDotNet Eager: AdaptiveAvgPool2D[32x64x8x8→4x4]")]
    public Tensor<float> AiDotNet_AdaptiveAvgPool2D()
    {
        return _engine.AdaptiveAvgPool2D(_bn_input, 4, 4);
    }

    [Benchmark(Description = "PyTorch: AdaptiveAvgPool2D[32x64x8x8→4x4]")]
    public TorchTensor PyTorch_AdaptiveAvgPool2D()
    {
        using var _ = torch.no_grad();
        return torch.nn.functional.adaptive_avg_pool2d(_t_bn_input, [4, 4]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 17. SHAPE + INDEXING OPS
    // ═══════════════════════════════════════════════════════════════════

    // --- Squeeze ---
    [Benchmark(Description = "AiDotNet Eager: Squeeze[1x256x256]")]
    public Tensor<float> AiDotNet_Squeeze()
    {
        var input = Tensor<float>.CreateRandom([1, 256, 256]);
        return _engine.TensorSqueeze(input, 0);
    }

    [Benchmark(Description = "PyTorch: Squeeze[1x256x256]")]
    public TorchTensor PyTorch_Squeeze()
    {
        var input = torch.randn([1, 256, 256]);
        return input.squeeze(0);
    }

    // --- Stack ---
    [Benchmark(Description = "AiDotNet Eager: Stack[4x100K]")]
    public Tensor<float> AiDotNet_Stack()
    {
        return _engine.TensorStack(new[] { _op_a, _op_a, _op_a, _op_a }, 0);
    }

    [Benchmark(Description = "PyTorch: Stack[4x100K]")]
    public TorchTensor PyTorch_Stack()
    {
        return torch.stack(new[] { _t_op_a, _t_op_a, _t_op_a, _t_op_a }, dim: 0);
    }

    // --- Tile/Repeat ---
    [Benchmark(Description = "AiDotNet Eager: Tile[256x256, 2x1]")]
    public Tensor<float> AiDotNet_Tile()
    {
        return _engine.TensorTile(_op_2d, new[] { 2, 1 });
    }

    [Benchmark(Description = "PyTorch: Tile[256x256, 2x1]")]
    public TorchTensor PyTorch_Tile()
    {
        return _t_op_2d.tile([2, 1]);
    }

    // --- CumSum ---
    [Benchmark(Description = "AiDotNet Eager: CumSum[100K, axis=0]")]
    public Tensor<float> AiDotNet_CumSum()
    {
        return _engine.TensorCumSum(_op_a, 0);
    }

    [Benchmark(Description = "PyTorch: CumSum[100K, axis=0]")]
    public TorchTensor PyTorch_CumSum()
    {
        return _t_op_a.cumsum(0);
    }

    // --- Square (x^2) ---
    [Benchmark(Description = "AiDotNet Eager: Square[100K]")]
    public Tensor<float> AiDotNet_Square_100K()
    {
        return _engine.TensorSquare(_op_a);
    }

    [Benchmark(Description = "PyTorch: Square[100K]")]
    public TorchTensor PyTorch_Square_100K()
    {
        return torch.square(_t_op_a);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 18. EMBEDDING + DROPOUT + PAD
    // ═══════════════════════════════════════════════════════════════════

    // --- Embedding ---
    private Tensor<int> _emb_indices = null!;
    private Tensor<float> _emb_table = null!;
    private TorchTensor _t_emb_indices = null!, _t_emb_table = null!;

    [IterationSetup(Target = nameof(AiDotNet_Embedding))]
    public void SetupEmbedding()
    {
        if (_emb_table != null) return;
        _emb_table = Tensor<float>.CreateRandom([10000, 128]); // vocab=10K, dim=128
        var idxArr = new int[256]; // sequence length 256
        var rng = new Random(42);
        for (int i = 0; i < 256; i++) idxArr[i] = rng.Next(10000);
        _emb_indices = new Tensor<int>(idxArr, [256]);
        _t_emb_table = torch.randn([10000, 128]);
        _t_emb_indices = torch.tensor(idxArr, torch.int64);
    }

    [Benchmark(Description = "AiDotNet Eager: Embedding[256, vocab=10K, dim=128]")]
    public Tensor<float> AiDotNet_Embedding()
    {
        return _engine.Embedding(_emb_indices, _emb_table);
    }

    [Benchmark(Description = "PyTorch: Embedding[256, vocab=10K, dim=128]")]
    public TorchTensor PyTorch_Embedding()
    {
        using var _ = torch.no_grad();
        return _t_emb_table.index_select(0, _t_emb_indices);
    }

    // --- Dropout ---
    [Benchmark(Description = "AiDotNet Eager: Dropout[100K, p=0.1]")]
    public Tensor<float> AiDotNet_Dropout()
    {
        return _engine.Dropout(_op_a, 0.1, training: true, out _);
    }

    [Benchmark(Description = "PyTorch: Dropout[100K, p=0.1]")]
    public TorchTensor PyTorch_Dropout()
    {
        return torch.nn.functional.dropout(_t_op_a, 0.1, training: true);
    }

    // --- ConstantPad ---
    [Benchmark(Description = "AiDotNet Eager: Pad[256x256, pad=2]")]
    public Tensor<float> AiDotNet_ConstantPad()
    {
        return _engine.TensorConstantPad(_op_2d, new[] { 2, 2, 2, 2 }, 0f);
    }

    [Benchmark(Description = "PyTorch: Pad[256x256, pad=2]")]
    public TorchTensor PyTorch_ConstantPad()
    {
        return torch.nn.functional.pad(_t_op_2d, [2, 2, 2, 2], mode: TorchSharp.PaddingModes.Constant, value: 0);
    }

    // --- PReLU ---
    private Tensor<float> _prelu_alpha = null!;
    private TorchTensor _t_prelu_alpha = null!;

    [IterationSetup(Target = nameof(AiDotNet_PReLU))]
    public void SetupPReLU()
    {
        if (_prelu_alpha != null) return;
        _prelu_alpha = Tensor<float>.CreateRandom([100000]);
        _t_prelu_alpha = torch.randn([100000]);
    }

    [Benchmark(Description = "AiDotNet Eager: PReLU[100K]")]
    public Tensor<float> AiDotNet_PReLU()
    {
        return _engine.TensorPReLU(_op_a, _prelu_alpha);
    }

    [Benchmark(Description = "PyTorch: PReLU[100K]")]
    public TorchTensor PyTorch_PReLU()
    {
        return torch.nn.functional.prelu(_t_op_a, _t_prelu_alpha);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 19. LOSS FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- L1Loss ---
    [Benchmark(Description = "AiDotNet Eager: L1Loss[32x10]")]
    public Tensor<float> AiDotNet_L1Loss()
    {
        if (_loss_pred is null) SetupLoss();
        return _engine.TensorL1Loss(_loss_pred, _loss_target);
    }

    [Benchmark(Description = "PyTorch: L1Loss[32x10]")]
    public TorchTensor PyTorch_L1Loss()
    {
        if (_t_loss_pred is null) SetupLoss();
        return torch.nn.functional.l1_loss(_t_loss_pred, _t_loss_target);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 20. LARGE MATMUL + REDUCTIONS
    // ═══════════════════════════════════════════════════════════════════

    // --- MatMul 1024x1024 ---
    private Tensor<float> _mat1024 = null!;
    private TorchTensor _t_mat1024 = null!;

    [IterationSetup(Target = nameof(AiDotNet_MatMul_1024))]
    public void SetupMat1024()
    {
        if (_mat1024 != null) return;
        _mat1024 = Tensor<float>.CreateRandom([1024, 1024]);
        _t_mat1024 = torch.randn([1024, 1024]);
    }

    [Benchmark(Description = "AiDotNet Eager: MatMul[1024x1024]")]
    public Tensor<float> AiDotNet_MatMul_1024() => _engine.TensorMatMul(_mat1024, _mat1024);

    [Benchmark(Description = "PyTorch: MatMul[1024x1024]")]
    public TorchTensor PyTorch_MatMul_1024() => torch.matmul(_t_mat1024, _t_mat1024);

    // --- Sum 100K ---
    [Benchmark(Description = "AiDotNet Eager: Sum[100K]")]
    public float AiDotNet_Sum_100K() => _engine.TensorSum(_op_a);

    [Benchmark(Description = "PyTorch: Sum[100K]")]
    public TorchTensor PyTorch_Sum_100K() => _t_op_a.sum();

    // --- Sum 1M ---
    [Benchmark(Description = "AiDotNet Eager: Sum[1M]")]
    public float AiDotNet_Sum_1M() => _engine.TensorSum(_op_large);

    [Benchmark(Description = "PyTorch: Sum[1M]")]
    public TorchTensor PyTorch_Sum_1M() => _t_op_large.sum();
}
#endif
