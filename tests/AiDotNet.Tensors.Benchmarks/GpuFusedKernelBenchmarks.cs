using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks for GPU fused kernels vs CPU, measuring throughput and speedup.
/// Covers reductions, element-wise, activations, gated activations, softmax, and linear algebra.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
public class GpuFusedKernelBenchmarks
{
    private CpuEngine _cpu = null!;
    private DirectGpuTensorEngine? _gpu;
    private Tensor<float> _input1M = null!;
    private Tensor<float> _input2_1M = null!;
    private Tensor<float> _input100K = null!;
    private Tensor<float> _gluInput = null!; // [batch, 2*dim]
    private Matrix<float> _matA = null!;
    private Matrix<float> _matB = null!;
    private Vector<float> _vecA = null!;
    private Vector<float> _vecB = null!;

    [GlobalSetup]
    public void Setup()
    {
        _cpu = new CpuEngine();
        try { _gpu = new DirectGpuTensorEngine(); if (!_gpu.IsGpuAvailable) _gpu = null; }
        catch { _gpu = null; }

        var rng = new Random(42);
        float[] data1M = new float[1_000_000];
        float[] data2_1M = new float[1_000_000];
        float[] data100K = new float[100_000];
        float[] gluData = new float[4 * 512]; // batch=4, 2*dim=512 -> dim=256
        for (int i = 0; i < data1M.Length; i++) { data1M[i] = (float)(rng.NextDouble() * 2 - 1); data2_1M[i] = (float)(rng.NextDouble() * 2 - 1); }
        for (int i = 0; i < data100K.Length; i++) data100K[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < gluData.Length; i++) gluData[i] = (float)(rng.NextDouble() * 2 - 1);

        _input1M = new Tensor<float>(data1M, new[] { 1000, 1000 });
        _input2_1M = new Tensor<float>(data2_1M, new[] { 1000, 1000 });
        _input100K = new Tensor<float>(data100K, new[] { 100, 1000 });
        _gluInput = new Tensor<float>(gluData, new[] { 4, 512 });

        _matA = new Matrix<float>(256, 256);
        _matB = new Matrix<float>(256, 256);
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < 256; j++)
            { _matA[i, j] = (float)(rng.NextDouble() * 2 - 1); _matB[i, j] = (float)(rng.NextDouble() * 2 - 1); }

        _vecA = new Vector<float>(1024);
        _vecB = new Vector<float>(1024);
        for (int i = 0; i < 1024; i++) { _vecA[i] = (float)(rng.NextDouble() * 2 - 1); _vecB[i] = (float)(rng.NextDouble() * 2 - 1); }
    }

    // ==================== Element-wise (1M elements) ====================

    [Benchmark(Description = "TensorAdd 1M (CPU)")] public Tensor<float> Add_CPU() => _cpu.TensorAdd(_input1M, _input2_1M);
    [Benchmark(Description = "TensorAdd 1M (GPU)")] public Tensor<float>? Add_GPU() => _gpu?.TensorAdd(_input1M, _input2_1M);

    [Benchmark(Description = "TensorMultiply 1M (CPU)")] public Tensor<float> Mul_CPU() => _cpu.TensorMultiply(_input1M, _input2_1M);
    [Benchmark(Description = "TensorMultiply 1M (GPU)")] public Tensor<float>? Mul_GPU() => _gpu?.TensorMultiply(_input1M, _input2_1M);

    // ==================== Reductions (1M elements) ====================

    [Benchmark(Description = "TensorSum 1M (CPU)")] public float Sum_CPU() => _cpu.TensorSum(_input1M);
    [Benchmark(Description = "TensorSum 1M (GPU)")] public float Sum_GPU() => _gpu?.TensorSum(_input1M) ?? 0;

    [Benchmark(Description = "TensorMean 1M (CPU)")] public float Mean_CPU() => _cpu.TensorMean(_input1M);
    [Benchmark(Description = "TensorMean 1M (GPU)")] public float Mean_GPU() => _gpu?.TensorMean(_input1M) ?? 0;

    // ==================== Activations (1M elements) ====================

    [Benchmark(Description = "Sigmoid 1M (CPU)")] public Tensor<float> Sigmoid_CPU() => _cpu.TensorSigmoid(_input1M);
    [Benchmark(Description = "Sigmoid 1M (GPU)")] public Tensor<float>? Sigmoid_GPU() => _gpu?.TensorSigmoid(_input1M);

    [Benchmark(Description = "ReLU 1M (CPU)")] public Tensor<float> ReLU_CPU() => _cpu.TensorReLU(_input1M);
    [Benchmark(Description = "ReLU 1M (GPU)")] public Tensor<float>? ReLU_GPU() => _gpu?.TensorReLU(_input1M);

    [Benchmark(Description = "GELU 1M (CPU)")] public Tensor<float> GELU_CPU() => _cpu.TensorGELU(_input1M);
    [Benchmark(Description = "GELU 1M (GPU)")] public Tensor<float>? GELU_GPU() => _gpu?.TensorGELU(_input1M);

    [Benchmark(Description = "Tanh 1M (CPU)")] public Tensor<float> Tanh_CPU() => _cpu.TensorTanh(_input1M);
    [Benchmark(Description = "Tanh 1M (GPU)")] public Tensor<float>? Tanh_GPU() => _gpu?.TensorTanh(_input1M);

    // ==================== Fused kernels (100K elements) ====================

    [Benchmark(Description = "TensorClip 100K (CPU)")] public Tensor<float> Clip_CPU() => _cpu.TensorClip(_input100K, -0.5f, 0.5f);
    [Benchmark(Description = "TensorClip 100K (GPU)")] public Tensor<float>? Clip_GPU() => _gpu?.TensorClip(_input100K, -0.5f, 0.5f);

    [Benchmark(Description = "TensorFrac 100K (CPU)")] public Tensor<float> Frac_CPU() => _cpu.TensorFrac(_input100K);
    [Benchmark(Description = "TensorFrac 100K (GPU)")] public Tensor<float>? Frac_GPU() => _gpu?.TensorFrac(_input100K);

    // ==================== Gated activations ====================

    [Benchmark(Description = "GLU 4x512 (CPU)")] public Tensor<float> GLU_CPU() => _cpu.GLU(_gluInput, -1);
    [Benchmark(Description = "GLU 4x512 (GPU)")] public Tensor<float>? GLU_GPU() => _gpu?.GLU(_gluInput, -1);

    [Benchmark(Description = "GeGLU 4x512 (CPU)")] public Tensor<float> GeGLU_CPU() => _cpu.GeGLU(_gluInput, -1);
    [Benchmark(Description = "GeGLU 4x512 (GPU)")] public Tensor<float>? GeGLU_GPU() => _gpu?.GeGLU(_gluInput, -1);

    [Benchmark(Description = "SwiGLU 4x512 (CPU)")] public Tensor<float> SwiGLU_CPU() => _cpu.SwiGLU(_gluInput, -1);
    [Benchmark(Description = "SwiGLU 4x512 (GPU)")] public Tensor<float>? SwiGLU_GPU() => _gpu?.SwiGLU(_gluInput, -1);

    // ==================== Softmax ====================

    [Benchmark(Description = "Softmax 1Kx1K (CPU)")] public Tensor<float> Softmax_CPU() => _cpu.Softmax(_input1M, -1);
    [Benchmark(Description = "Softmax 1Kx1K (GPU)")] public Tensor<float>? Softmax_GPU() => _gpu?.Softmax(_input1M, -1);

    // ==================== Linear algebra ====================

    [Benchmark(Description = "MatMul 256x256 (CPU)")] public Matrix<float> MatMul_CPU() => _cpu.MatrixMultiply(_matA, _matB);
    [Benchmark(Description = "MatMul 256x256 (GPU)")] public Matrix<float>? MatMul_GPU() => _gpu is not null ? ((IEngine)_gpu).MatrixMultiply(_matA, _matB) : null;

    [Benchmark(Description = "DotProduct 1024 (CPU)")] public float Dot_CPU() => _cpu.DotProduct(_vecA, _vecB);
    [Benchmark(Description = "DotProduct 1024 (GPU)")] public float Dot_GPU() => _gpu?.DotProduct(_vecA, _vecB) ?? 0;

    // ==================== Eye/OneHot ====================

    [Benchmark(Description = "TensorEye 64 (CPU)")] public Tensor<float> Eye_CPU() => _cpu.TensorEye<float>(64);
    [Benchmark(Description = "TensorEye 64 (GPU)")] public Tensor<float>? Eye_GPU() => _gpu?.TensorEye<float>(64);

    [GlobalCleanup]
    public void Cleanup()
    {
        (_gpu as IDisposable)?.Dispose();
    }
}
