using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Head-to-head GPU benchmarks: AiDotNet fused GPU kernels vs TorchSharp GPU.
/// Measures throughput and latency for critical neural network operations.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
public class GpuVsTorchSharpBenchmarks
{
    private DirectGpuTensorEngine? _gpu;
    private Tensor<float> _input1M = null!;
    private Tensor<float> _input2_1M = null!;
    private Tensor<float> _softmaxInput = null!;
    private Matrix<float> _matA = null!;
    private Matrix<float> _matB = null!;

    // TorchSharp tensors
    private TorchSharp.torch.Tensor? _torchA;
    private TorchSharp.torch.Tensor? _torchB;
    private TorchSharp.torch.Tensor? _torchSoftmax;
    private TorchSharp.torch.Tensor? _torchMatA;
    private TorchSharp.torch.Tensor? _torchMatB;
    private TorchSharp.torch.Device _torchDevice = TorchSharp.torch.CPU;
    private bool _torchGpuAvailable;

    [GlobalSetup]
    public void Setup()
    {
        try { _gpu = new DirectGpuTensorEngine(); if (!_gpu.IsGpuAvailable) _gpu = null; }
        catch { _gpu = null; }

        var rng = new Random(42);
        float[] d1 = new float[1_000_000], d2 = new float[1_000_000];
        for (int i = 0; i < d1.Length; i++) { d1[i] = (float)(rng.NextDouble() * 2 - 1); d2[i] = (float)(rng.NextDouble() * 2 - 1); }
        _input1M = new Tensor<float>(d1, new[] { 1000, 1000 });
        _input2_1M = new Tensor<float>(d2, new[] { 1000, 1000 });
        _softmaxInput = new Tensor<float>(d1, new[] { 1000, 1000 });

        _matA = new Matrix<float>(512, 512);
        _matB = new Matrix<float>(512, 512);
        for (int i = 0; i < 512; i++)
            for (int j = 0; j < 512; j++)
            { _matA[i, j] = (float)rng.NextDouble(); _matB[i, j] = (float)rng.NextDouble(); }

        // Setup TorchSharp
        try
        {
            _torchGpuAvailable = TorchSharp.torch.cuda.is_available();
            _torchDevice = _torchGpuAvailable ? TorchSharp.torch.CUDA : TorchSharp.torch.CPU;
            _torchA = TorchSharp.torch.tensor(d1, new long[] { 1000, 1000 }).to(_torchDevice);
            _torchB = TorchSharp.torch.tensor(d2, new long[] { 1000, 1000 }).to(_torchDevice);
            _torchSoftmax = TorchSharp.torch.tensor(d1, new long[] { 1000, 1000 }).to(_torchDevice);

            float[] matData = new float[512 * 512];
            float[] matData2 = new float[512 * 512];
            for (int i = 0; i < 512; i++)
                for (int j = 0; j < 512; j++)
                { matData[i * 512 + j] = _matA[i, j]; matData2[i * 512 + j] = _matB[i, j]; }
            _torchMatA = TorchSharp.torch.tensor(matData, new long[] { 512, 512 }).to(_torchDevice);
            _torchMatB = TorchSharp.torch.tensor(matData2, new long[] { 512, 512 }).to(_torchDevice);
        }
        catch
        {
            _torchGpuAvailable = false;
        }
    }

    // ==================== Element-wise ====================

    [Benchmark(Description = "Add 1M (AiDotNet GPU)")]
    public Tensor<float>? AiDotNet_Add() => _gpu?.TensorAdd(_input1M, _input2_1M);

    [Benchmark(Description = "Add 1M (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_Add() => _torchA?.add(_torchB);

    [Benchmark(Description = "Multiply 1M (AiDotNet GPU)")]
    public Tensor<float>? AiDotNet_Mul() => _gpu?.TensorMultiply(_input1M, _input2_1M);

    [Benchmark(Description = "Multiply 1M (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_Mul() => _torchA?.mul(_torchB);

    // ==================== Activations ====================

    [Benchmark(Description = "Sigmoid 1M (AiDotNet GPU)")]
    public Tensor<float>? AiDotNet_Sigmoid() => _gpu?.TensorSigmoid(_input1M);

    [Benchmark(Description = "Sigmoid 1M (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_Sigmoid() => _torchA?.sigmoid();

    [Benchmark(Description = "ReLU 1M (AiDotNet GPU)")]
    public Tensor<float>? AiDotNet_ReLU() => _gpu?.TensorReLU(_input1M);

    [Benchmark(Description = "ReLU 1M (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_ReLU() => _torchA?.relu();

    // ==================== Softmax ====================

    [Benchmark(Description = "Softmax 1Kx1K (AiDotNet GPU)")]
    public Tensor<float>? AiDotNet_Softmax() => _gpu?.Softmax(_softmaxInput, -1);

    [Benchmark(Description = "Softmax 1Kx1K (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_Softmax() => TorchSharp.torch.nn.functional.softmax(_torchSoftmax, -1);

    // ==================== MatMul ====================

    [Benchmark(Description = "MatMul 512x512 (AiDotNet GPU)")]
    public Matrix<float>? AiDotNet_MatMul() => _gpu is not null ? ((IEngine)_gpu).MatrixMultiply(_matA, _matB) : null;

    [Benchmark(Description = "MatMul 512x512 (TorchSharp GPU)")]
    public TorchSharp.torch.Tensor? Torch_MatMul() => _torchMatA?.mm(_torchMatB);

    // ==================== Reductions ====================

    [Benchmark(Description = "Sum 1M (AiDotNet GPU)")]
    public float AiDotNet_Sum() => _gpu?.TensorSum(_input1M) ?? 0;

    [Benchmark(Description = "Sum 1M (TorchSharp GPU)")]
    public float Torch_Sum() => _torchA?.sum().item<float>() ?? 0;

    [GlobalCleanup]
    public void Cleanup()
    {
        _torchA?.Dispose();
        _torchB?.Dispose();
        _torchSoftmax?.Dispose();
        _torchMatA?.Dispose();
        _torchMatB?.Dispose();
    }
}
