// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class FusedIssue301Benchmarks
{
    private const int Batch = 32;
    private const int InputFeatures = 256;
    private const int Rank = 8;
    private const int OutputFeatures = 128;
    private const float LoRAScaling = 0.5f;

    private CpuEngine _engine = null!;

    private Tensor<float> _input = null!;
    private Tensor<float> _baseOutput = null!;
    private Tensor<float> _loraA = null!;
    private Tensor<float> _loraB = null!;
    private Tensor<float> _loraOutput = null!;

    private Tensor<float> _xT = null!;
    private Tensor<float> _epsilon = null!;
    private Tensor<float> _ddimOutput = null!;

    private Tensor<float> _sparseInput = null!;
    private Tensor<float> _sparseDenseWeight = null!;
    private Tensor<float> _sparseValues = null!;
    private Tensor<float> _sparseBias = null!;
    private Tensor<float> _sparseOutput = null!;
    private int[] _sparseRowOffsets = null!;
    private int[] _sparseColIndices = null!;

    private TorchTensor _torchInput = null!;
    private TorchTensor _torchBaseOutput = null!;
    private TorchTensor _torchLoraA = null!;
    private TorchTensor _torchLoraB = null!;
    private TorchTensor _torchXT = null!;
    private TorchTensor _torchEpsilon = null!;
    private TorchTensor _torchSparseInput = null!;
    private TorchTensor _torchSparseDenseWeight = null!;
    private TorchTensor _torchSparseBias = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();

        _input = Tensor<float>.CreateRandom(new[] { Batch, InputFeatures });
        _baseOutput = Tensor<float>.CreateRandom(new[] { Batch, OutputFeatures });
        _loraA = Tensor<float>.CreateRandom(new[] { InputFeatures, Rank });
        _loraB = Tensor<float>.CreateRandom(new[] { Rank, OutputFeatures });
        _loraOutput = new Tensor<float>(new[] { Batch, OutputFeatures });

        _xT = Tensor<float>.CreateRandom(new[] { Batch * OutputFeatures });
        _epsilon = Tensor<float>.CreateRandom(new[] { Batch * OutputFeatures });
        _ddimOutput = new Tensor<float>(new[] { Batch * OutputFeatures });

        BuildSparseLinearInputs();

        _torchInput = torch.randn([Batch, InputFeatures]);
        _torchBaseOutput = torch.randn([Batch, OutputFeatures]);
        _torchLoraA = torch.randn([InputFeatures, Rank]);
        _torchLoraB = torch.randn([Rank, OutputFeatures]);
        _torchXT = torch.randn([Batch * OutputFeatures]);
        _torchEpsilon = torch.randn([Batch * OutputFeatures]);
        _torchSparseInput = torch.randn([Batch, InputFeatures]);
        _torchSparseDenseWeight = torch.tensor(_sparseDenseWeight.AsSpan().ToArray(), new long[] { InputFeatures, OutputFeatures });
        _torchSparseBias = torch.randn([OutputFeatures]);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _torchInput?.Dispose();
        _torchBaseOutput?.Dispose();
        _torchLoraA?.Dispose();
        _torchLoraB?.Dispose();
        _torchXT?.Dispose();
        _torchEpsilon?.Dispose();
        _torchSparseInput?.Dispose();
        _torchSparseDenseWeight?.Dispose();
        _torchSparseBias?.Dispose();
    }

    [Benchmark(Description = "AiDotNet Decomposed: LoRA forward")]
    public Tensor<float> AiDotNet_LoRA_Decomposed()
    {
        var hidden = _engine.TensorMatMul(_input, _loraA);
        var delta = _engine.TensorMatMul(hidden, _loraB);
        var scaled = _engine.TensorMultiplyScalar(delta, LoRAScaling);
        return _engine.TensorAdd(_baseOutput, scaled);
    }

    [Benchmark(Description = "AiDotNet Fused: LoRA forward")]
    public Tensor<float> AiDotNet_LoRA_Fused()
    {
        CpuFusedOperations.FusedLoRAForward(_input, _baseOutput, _loraA, _loraB, LoRAScaling, _loraOutput);
        return _loraOutput;
    }

    [Benchmark(Description = "TorchSharp/PyTorch: LoRA forward")]
    public TorchTensor PyTorch_LoRA()
    {
        var hidden = torch.matmul(_torchInput, _torchLoraA);
        var delta = torch.matmul(hidden, _torchLoraB);
        var scaled = delta * LoRAScaling;
        return _torchBaseOutput + scaled;
    }

    [Benchmark(Description = "AiDotNet Decomposed: DDIM step")]
    public Tensor<float> AiDotNet_DDIM_Decomposed()
    {
        const float alphaBarT = 0.64f;
        const float alphaBarTMinus1 = 0.81f;
        var noise = _engine.TensorMultiplyScalar(_epsilon, MathF.Sqrt(1f - alphaBarT));
        var x0Numerator = _engine.TensorSubtract(_xT, noise);
        var x0Pred = _engine.TensorDivideScalar(x0Numerator, MathF.Sqrt(alphaBarT));
        var prevX0 = _engine.TensorMultiplyScalar(x0Pred, MathF.Sqrt(alphaBarTMinus1));
        var prevNoise = _engine.TensorMultiplyScalar(_epsilon, MathF.Sqrt(1f - alphaBarTMinus1));
        return _engine.TensorAdd(prevX0, prevNoise);
    }

    [Benchmark(Description = "AiDotNet Fused: DDIM step")]
    public Tensor<float> AiDotNet_DDIM_Fused()
    {
        CpuFusedOperations.FusedDDIMStep(_xT, _epsilon, 0.64f, 0.81f, _ddimOutput);
        return _ddimOutput;
    }

    [Benchmark(Description = "TorchSharp/PyTorch: DDIM step")]
    public TorchTensor PyTorch_DDIM()
    {
        const float alphaBarT = 0.64f;
        const float alphaBarTMinus1 = 0.81f;
        var x0Pred = (_torchXT - MathF.Sqrt(1f - alphaBarT) * _torchEpsilon) / MathF.Sqrt(alphaBarT);
        return MathF.Sqrt(alphaBarTMinus1) * x0Pred + MathF.Sqrt(1f - alphaBarTMinus1) * _torchEpsilon;
    }

    [Benchmark(Description = "AiDotNet Decomposed: sparse linear as dense matmul+bias+ReLU")]
    public Tensor<float> AiDotNet_SparseLinear_DecomposedDense()
    {
        var linear = _engine.TensorMatMul(_sparseInput, _sparseDenseWeight);
        var biased = _engine.TensorBroadcastAdd(linear, _sparseBias);
        return _engine.ReLU(biased);
    }

    [Benchmark(Description = "AiDotNet Fused: CSR sparse linear+bias+ReLU")]
    public Tensor<float> AiDotNet_SparseLinear_Fused()
    {
        CpuFusedOperations.FusedSparseLinear(
            _sparseInput,
            _sparseRowOffsets,
            _sparseColIndices,
            _sparseValues,
            _sparseBias,
            FusedActivationType.ReLU,
            _sparseOutput);
        return _sparseOutput;
    }

    [Benchmark(Description = "TorchSharp/PyTorch: dense sparse-linear equivalent")]
    public TorchTensor PyTorch_SparseLinear_Dense()
    {
        var linear = torch.matmul(_torchSparseInput, _torchSparseDenseWeight);
        return torch.nn.functional.relu(linear + _torchSparseBias);
    }

    private void BuildSparseLinearInputs()
    {
        _sparseInput = Tensor<float>.CreateRandom(new[] { Batch, InputFeatures });
        _sparseBias = Tensor<float>.CreateRandom(new[] { OutputFeatures });
        _sparseOutput = new Tensor<float>(new[] { Batch, OutputFeatures });

        var rng = new Random(123);
        var dense = new float[InputFeatures * OutputFeatures];
        var rowOffsets = new int[OutputFeatures + 1];
        var cols = new List<int>(InputFeatures * OutputFeatures / 10);
        var values = new List<float>(InputFeatures * OutputFeatures / 10);

        for (int j = 0; j < OutputFeatures; j++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                if (rng.NextDouble() < 0.9)
                    continue;

                float value = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.1);
                dense[i * OutputFeatures + j] = value;
                cols.Add(i);
                values.Add(value);
            }
            rowOffsets[j + 1] = cols.Count;
        }

        _sparseDenseWeight = new Tensor<float>(dense, new[] { InputFeatures, OutputFeatures });
        _sparseRowOffsets = rowOffsets;
        _sparseColIndices = cols.ToArray();
        _sparseValues = new Tensor<float>(values.ToArray(), new[] { values.Count });
    }
}
