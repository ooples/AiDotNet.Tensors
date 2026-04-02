using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmarks for hyperbolic geometry and octonion algebra GPU kernels.
/// Measures: GPU vs CPU throughput, fused vs non-fused kernel performance.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
public class HyperbolicOctonionGpuBenchmarks
{
    private CpuEngine _cpu = null!;
    private VulkanBackend? _vulkan;

    // Poincaré ball test data
    private float[] _pointsX = null!;
    private float[] _pointsY = null!;
    private float[] _tangentVecs = null!;
    private IGpuBuffer? _gpuPointsX, _gpuPointsY, _gpuTangent, _gpuOutput, _gpuDistOutput;

    // Octonion test data
    private float[] _octA = null!;
    private float[] _octB = null!;
    private float[] _octInput = null!;
    private float[] _octWeights = null!;
    private float[] _octBiases = null!;
    private IGpuBuffer? _gpuOctA, _gpuOctB, _gpuOctOut;
    private IGpuBuffer? _gpuOctInput, _gpuOctWeights, _gpuOctBiases, _gpuOctForwardOut, _gpuOctFusedOut;

    // Hyperbolic linear test data
    private float[] _hypInput = null!;
    private float[] _hypWeights = null!;
    private float[] _hypBiases = null!;
    private IGpuBuffer? _gpuHypInput, _gpuHypWeights, _gpuHypBiases, _gpuHypOutput, _gpuHypFusedOutput;

    // CpuEngine tensors for comparison
    private Tensor<float> _cpuOctInputT = null!;
    private Tensor<float> _cpuOctWeightsT = null!;

    [Params(32, 128, 512)]
    public int BatchSize { get; set; }

    private const int Dim = 64;
    private const int OctCount = 256;
    private const int OctInputFeatures = 16;
    private const int OctOutputFeatures = 8;
    private const int HypInputFeatures = 64;
    private const int HypOutputFeatures = 32;
    private const float Curvature = 1.0f;

    [GlobalSetup]
    public void Setup()
    {
        _cpu = new CpuEngine();
        try
        {
            _vulkan = VulkanBackend.Instance;
            if (!_vulkan.Initialize()) _vulkan = null;
        }
        catch { _vulkan = null; }

        var rng = new Random(42);
        float RandF() => (float)(rng.NextDouble() * 2 - 1);
        float RandSmall() => (float)(rng.NextDouble() * 0.8 - 0.4); // Inside Poincaré ball

        // Poincaré data
        _pointsX = Enumerable.Range(0, BatchSize * Dim).Select(_ => RandSmall()).ToArray();
        _pointsY = Enumerable.Range(0, BatchSize * Dim).Select(_ => RandSmall()).ToArray();
        _tangentVecs = Enumerable.Range(0, BatchSize * Dim).Select(_ => RandF() * 0.3f).ToArray();

        // Octonion data
        _octA = Enumerable.Range(0, OctCount * 8).Select(_ => RandF()).ToArray();
        _octB = Enumerable.Range(0, OctCount * 8).Select(_ => RandF()).ToArray();
        _octInput = Enumerable.Range(0, BatchSize * OctInputFeatures * 8).Select(_ => RandF() * 0.3f).ToArray();
        _octWeights = Enumerable.Range(0, OctOutputFeatures * OctInputFeatures * 8).Select(_ => RandF() * 0.1f).ToArray();
        _octBiases = Enumerable.Range(0, OctOutputFeatures * 8).Select(_ => RandF() * 0.1f).ToArray();

        // Hyperbolic linear data
        _hypInput = Enumerable.Range(0, BatchSize * HypInputFeatures).Select(_ => RandF()).ToArray();
        _hypWeights = Enumerable.Range(0, HypOutputFeatures * HypInputFeatures).Select(_ => RandF() * 0.1f).ToArray();
        _hypBiases = Enumerable.Range(0, HypOutputFeatures).Select(_ => RandF() * 0.1f).ToArray();

        // CPU tensors for OctonionMatMulTensor
        _cpuOctInputT = new Tensor<float>(_octInput, new[] { BatchSize, OctInputFeatures, 8 });
        _cpuOctWeightsT = new Tensor<float>(_octWeights, new[] { OctOutputFeatures, OctInputFeatures, 8 });

        if (_vulkan is not null)
        {
            _gpuPointsX = _vulkan.AllocateBuffer(_pointsX);
            _gpuPointsY = _vulkan.AllocateBuffer(_pointsY);
            _gpuTangent = _vulkan.AllocateBuffer(_tangentVecs);
            _gpuOutput = _vulkan.AllocateBuffer(BatchSize * Dim);
            _gpuDistOutput = _vulkan.AllocateBuffer(BatchSize);

            _gpuOctA = _vulkan.AllocateBuffer(_octA);
            _gpuOctB = _vulkan.AllocateBuffer(_octB);
            _gpuOctOut = _vulkan.AllocateBuffer(OctCount * 8);

            _gpuOctInput = _vulkan.AllocateBuffer(_octInput);
            _gpuOctWeights = _vulkan.AllocateBuffer(_octWeights);
            _gpuOctBiases = _vulkan.AllocateBuffer(_octBiases);
            _gpuOctForwardOut = _vulkan.AllocateBuffer(BatchSize * OctOutputFeatures * 8);
            _gpuOctFusedOut = _vulkan.AllocateBuffer(BatchSize * OctOutputFeatures * 8);

            _gpuHypInput = _vulkan.AllocateBuffer(_hypInput);
            _gpuHypWeights = _vulkan.AllocateBuffer(_hypWeights);
            _gpuHypBiases = _vulkan.AllocateBuffer(_hypBiases);
            _gpuHypOutput = _vulkan.AllocateBuffer(BatchSize * HypOutputFeatures);
            _gpuHypFusedOutput = _vulkan.AllocateBuffer(BatchSize * HypOutputFeatures);
        }
    }

    // ==================== Poincaré Ball Operations ====================

    [Benchmark(Description = "MobiusAdd CPU")]
    public void MobiusAdd_Cpu()
    {
        // CPU reference: manual Möbius addition
        var result = new float[BatchSize * Dim];
        for (int b = 0; b < BatchSize; b++)
        {
            int off = b * Dim;
            float xSq = 0, ySq = 0, xy = 0;
            for (int d = 0; d < Dim; d++)
            {
                xSq += _pointsX[off + d] * _pointsX[off + d];
                ySq += _pointsY[off + d] * _pointsY[off + d];
                xy += _pointsX[off + d] * _pointsY[off + d];
            }
            float denom = 1f + 2f * Curvature * xy + Curvature * Curvature * xSq * ySq;
            float numX = 1f + 2f * Curvature * xy + Curvature * ySq;
            float numY = 1f - Curvature * xSq;
            for (int d = 0; d < Dim; d++)
                result[off + d] = (numX * _pointsX[off + d] + numY * _pointsY[off + d]) / denom;
        }
    }

    [Benchmark(Description = "MobiusAdd GPU")]
    public void MobiusAdd_Gpu()
    {
        if (_vulkan is null) return;
        _vulkan.MobiusAdd(_gpuPointsX!, _gpuPointsY!, _gpuOutput!, BatchSize, Dim, Curvature);
    }

    [Benchmark(Description = "PoincareExpMap CPU")]
    public void PoincareExpMap_Cpu()
    {
        var result = new float[BatchSize * Dim];
        for (int b = 0; b < BatchSize; b++)
        {
            int off = b * Dim;
            float bpSq = 0, tvNorm = 0;
            for (int d = 0; d < Dim; d++)
            {
                bpSq += _pointsX[off + d] * _pointsX[off + d];
                tvNorm += _tangentVecs[off + d] * _tangentVecs[off + d];
            }
            tvNorm = MathF.Sqrt(tvNorm);
            float lambda = 2f / (1f - Curvature * bpSq);
            float t = MathF.Tanh(MathF.Sqrt(Curvature) * lambda * tvNorm / 2f) / (MathF.Sqrt(Curvature) * tvNorm + 1e-10f);
            for (int d = 0; d < Dim; d++) result[off + d] = _pointsX[off + d] + t * _tangentVecs[off + d];
        }
    }

    [Benchmark(Description = "PoincareExpMap GPU")]
    public void PoincareExpMap_Gpu()
    {
        if (_vulkan is null) return;
        _vulkan.PoincareExpMap(_gpuPointsX!, _gpuTangent!, _gpuOutput!, BatchSize, Dim, Curvature);
    }

    [Benchmark(Description = "PoincareDistance CPU")]
    public void PoincareDistance_Cpu()
    {
        var result = new float[BatchSize];
        for (int b = 0; b < BatchSize; b++)
        {
            int off = b * Dim;
            float diffSq = 0, xSq = 0, ySq = 0;
            for (int d = 0; d < Dim; d++)
            {
                float diff = _pointsX[off + d] - _pointsY[off + d];
                diffSq += diff * diff;
                xSq += _pointsX[off + d] * _pointsX[off + d];
                ySq += _pointsY[off + d] * _pointsY[off + d];
            }
            float arg = 1f + 2f * Curvature * diffSq / ((1f - Curvature * xSq) * (1f - Curvature * ySq));
            result[b] = MathF.Log(arg + MathF.Sqrt(arg * arg - 1f));
        }
    }

    [Benchmark(Description = "PoincareDistance GPU")]
    public void PoincareDistance_Gpu()
    {
        if (_vulkan is null) return;
        _vulkan.PoincareDistance(_gpuPointsX!, _gpuPointsY!, _gpuDistOutput!, BatchSize, Dim, Curvature);
    }

    // ==================== Octonion Operations ====================

    [Benchmark(Description = "OctonionMultiply CPU")]
    public void OctonionMultiply_Cpu()
    {
        var result = new float[OctCount * 8];
        for (int n = 0; n < OctCount; n++)
        {
            int o = n * 8;
            float a0=_octA[o],a1=_octA[o+1],a2=_octA[o+2],a3=_octA[o+3];
            float a4=_octA[o+4],a5=_octA[o+5],a6=_octA[o+6],a7=_octA[o+7];
            float b0=_octB[o],b1=_octB[o+1],b2=_octB[o+2],b3=_octB[o+3];
            float b4=_octB[o+4],b5=_octB[o+5],b6=_octB[o+6],b7=_octB[o+7];
            result[o] = a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7;
            // ... (8 components)
        }
    }

    [Benchmark(Description = "OctonionMultiply GPU")]
    public void OctonionMultiply_Gpu()
    {
        if (_vulkan is null) return;
        _vulkan.OctonionMultiply(_gpuOctA!, _gpuOctB!, _gpuOctOut!, OctCount);
    }

    // ==================== Octonion Linear (Fused vs Non-Fused) ====================

    [Benchmark(Description = "OctonionLinear CPU (CpuEngine)")]
    public Tensor<float> OctonionLinear_CpuEngine()
    {
        return _cpu.OctonionMatMulTensor(_cpuOctInputT, _cpuOctWeightsT);
    }

    [Benchmark(Description = "OctonionLinear GPU (non-fused: forward only)")]
    public void OctonionLinear_Gpu_NonFused()
    {
        if (_vulkan is null) return;
        _vulkan.OctonionLinearForward(_gpuOctInput!, _gpuOctWeights!, _gpuOctBiases!, _gpuOctForwardOut!,
            BatchSize, OctInputFeatures, OctOutputFeatures);
    }

    [Benchmark(Description = "OctonionLinear GPU (fused: forward + ReLU)")]
    public void OctonionLinear_Gpu_FusedReLU()
    {
        if (_vulkan is null) return;
        _vulkan.OctonionLinearForwardFusedReLU(_gpuOctInput!, _gpuOctWeights!, _gpuOctBiases!, _gpuOctFusedOut!,
            BatchSize, OctInputFeatures, OctOutputFeatures);
    }

    // ==================== Hyperbolic Linear (Fused vs Non-Fused) ====================

    [Benchmark(Description = "HyperbolicLinear GPU (non-fused: matmul+bias+project)")]
    public void HyperbolicLinear_Gpu_NonFused()
    {
        if (_vulkan is null) return;
        _vulkan.HyperbolicLinearForward(_gpuHypInput!, _gpuHypWeights!, _gpuHypBiases!, _gpuHypOutput!,
            BatchSize, HypInputFeatures, HypOutputFeatures, Curvature, 1e-5f);
    }

    [Benchmark(Description = "HyperbolicLinear GPU (fused)")]
    public void HyperbolicLinear_Gpu_Fused()
    {
        if (_vulkan is null) return;
        _vulkan.HyperbolicLinearForwardFused(_gpuHypInput!, _gpuHypWeights!, _gpuHypBiases!, _gpuHypFusedOutput!,
            BatchSize, HypInputFeatures, HypOutputFeatures, Curvature, 1e-5f);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _gpuPointsX?.Dispose(); _gpuPointsY?.Dispose(); _gpuTangent?.Dispose();
        _gpuOutput?.Dispose(); _gpuDistOutput?.Dispose();
        _gpuOctA?.Dispose(); _gpuOctB?.Dispose(); _gpuOctOut?.Dispose();
        _gpuOctInput?.Dispose(); _gpuOctWeights?.Dispose(); _gpuOctBiases?.Dispose();
        _gpuOctForwardOut?.Dispose(); _gpuOctFusedOut?.Dispose();
        _gpuHypInput?.Dispose(); _gpuHypWeights?.Dispose(); _gpuHypBiases?.Dispose();
        _gpuHypOutput?.Dispose(); _gpuHypFusedOutput?.Dispose();
    }
}
