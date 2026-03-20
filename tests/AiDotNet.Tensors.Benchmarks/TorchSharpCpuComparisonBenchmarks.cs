#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class TorchSharpCpuComparisonBenchmarks
{
    private static readonly int[] MatrixSizes = [256, 512];
    private static readonly int[] VectorSizes = [100_000, 1_000_000];
    private const int LargeSize = 1_000_000;
    private const int SmallSize = 100_000;

    private CpuEngine _cpuEngine = null!;
    private readonly Dictionary<int, Tensor<float>> _aiMatricesA = new();
    private readonly Dictionary<int, Tensor<float>> _aiMatricesB = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsA = new();
    private readonly Dictionary<int, Tensor<float>> _aiVectorsB = new();

    // Raw arrays for direct TensorPrimitives comparison
    private readonly Dictionary<int, float[]> _rawArraysA = new();
    private readonly Dictionary<int, float[]> _rawArraysB = new();
    private readonly Dictionary<int, float[]> _rawDestination = new();

    private readonly Dictionary<int, TorchTensor> _torchMatricesA = new();
    private readonly Dictionary<int, TorchTensor> _torchMatricesB = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsA = new();
    private readonly Dictionary<int, TorchTensor> _torchVectorsB = new();

    private Tensor<float>? _aiConvInput;
    private Tensor<float>? _aiConvKernel;
    private Tensor<float>? _aiConvOutput;
    private TorchTensor? _torchConvInput;
    private TorchTensor? _torchConvKernel;

    // Normalization data
    private Tensor<float>? _aiNormInput;
    private Tensor<float>? _aiNormGamma;
    private Tensor<float>? _aiNormBeta;
    private TorchTensor? _torchNormInput;
    private TorchTensor? _torchNormWeight;
    private TorchTensor? _torchNormBias;

    // Pooling data
    private Tensor<float>? _aiPoolInput;
    private TorchTensor? _torchPoolInput;

    // Softmax data (2D: batch x features)
    private Tensor<float>? _aiSoftmaxInput;
    private TorchTensor? _torchSoftmaxInput;

    // Backward pass data
    private Tensor<float>? _aiSigmoidOutput;
    private Tensor<float>? _aiGradOutput;
    private Tensor<float>? _aiTanhOutput;
    private TorchTensor? _torchSigmoidOutput;
    private TorchTensor? _torchGradOutput;
    private TorchTensor? _torchTanhOutput;

    // Attention data
    private Tensor<float>? _aiQueryMatrix;
    private Tensor<float>? _aiKeyMatrix;
    private TorchTensor? _torchQueryMatrix;
    private TorchTensor? _torchKeyMatrix;

    // Double precision data
    private Tensor<double>? _aiDoubleVectorA;
    private Tensor<double>? _aiDoubleVectorB;
    private Tensor<double>? _aiDoubleMatA;
    private Tensor<double>? _aiDoubleMatB;
    private Tensor<double>? _aiDoubleSoftmaxInput;
    private Tensor<double>? _aiDoubleConvInput;
    private Tensor<double>? _aiDoubleConvKernel;
    private TorchTensor? _torchDoubleVectorA;
    private TorchTensor? _torchDoubleVectorB;
    private TorchTensor? _torchDoubleMatA;
    private TorchTensor? _torchDoubleMatB;
    private TorchTensor? _torchDoubleSoftmaxInput;
    private TorchTensor? _torchDoubleConvInput;
    private TorchTensor? _torchDoubleConvKernel;

    private int _convStride;
    private int _convPadding;
    private int _convDilation;
    private long[]? _torchConvStride;
    private long[]? _torchConvPadding;
    private long[]? _torchConvDilation;

    private torch.Device _torchDevice = null!;
    private readonly Consumer _consumer = new();

    [GlobalSetup]
    public void Setup()
    {
        _cpuEngine = new CpuEngine();
        AiDotNetEngine.Current = _cpuEngine;

        torch.set_grad_enabled(false);
        _torchDevice = torch.CPU;
        Console.WriteLine($"TorchSharp device: CPU (forced), threads: {torch.get_num_threads()}");
        Console.WriteLine("AiDotNet BLAS: OpenBLAS package referenced (runtime load required)");

        foreach (var size in MatrixSizes)
        {
            var dataA = CreateData(size * size, seedOffset: size);
            var dataB = CreateData(size * size, seedOffset: size + 13_337);

            _aiMatricesA[size] = new Tensor<float>(dataA, new[] { size, size });
            _aiMatricesB[size] = new Tensor<float>(dataB, new[] { size, size });

            _torchMatricesA[size] = torch.tensor(dataA, new long[] { size, size }, device: _torchDevice);
            _torchMatricesB[size] = torch.tensor(dataB, new long[] { size, size }, device: _torchDevice);
        }

        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            var dataB = CreateData(size, seedOffset: size + 99_999);

            _aiVectorsA[size] = new Tensor<float>(dataA, new[] { size });
            _aiVectorsB[size] = new Tensor<float>(dataB, new[] { size });

            _rawArraysA[size] = (float[])dataA.Clone();
            _rawArraysB[size] = (float[])dataB.Clone();
            _rawDestination[size] = new float[size];

            _torchVectorsA[size] = torch.tensor(dataA, new long[] { size }, device: _torchDevice);
            _torchVectorsB[size] = torch.tensor(dataB, new long[] { size }, device: _torchDevice);
        }

        InitializeConv2D();
        InitializeNormalization();
        InitializePooling();
        InitializeSoftmax();
        InitializeBackwardData();
        InitializeAttention();
        InitializeDoublePrecision();
        Warmup();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (var tensor in _torchMatricesA.Values) tensor.Dispose();
        foreach (var tensor in _torchMatricesB.Values) tensor.Dispose();
        foreach (var tensor in _torchVectorsA.Values) tensor.Dispose();
        foreach (var tensor in _torchVectorsB.Values) tensor.Dispose();

        _torchConvInput?.Dispose();
        _torchConvKernel?.Dispose();
        _torchNormInput?.Dispose();
        _torchNormWeight?.Dispose();
        _torchNormBias?.Dispose();
        _torchPoolInput?.Dispose();
        _torchSoftmaxInput?.Dispose();
        _torchSigmoidOutput?.Dispose();
        _torchGradOutput?.Dispose();
        _torchTanhOutput?.Dispose();
        _torchQueryMatrix?.Dispose();
        _torchKeyMatrix?.Dispose();
        _torchDoubleVectorA?.Dispose();
        _torchDoubleVectorB?.Dispose();
        _torchDoubleMatA?.Dispose();
        _torchDoubleMatB?.Dispose();
        _torchDoubleSoftmaxInput?.Dispose();
        _torchDoubleConvInput?.Dispose();
        _torchDoubleConvKernel?.Dispose();
    }

    [IterationSetup]
    public void IterationSetup()
    {
        foreach (var size in VectorSizes)
        {
            var dataA = CreateData(size, seedOffset: size);
            Array.Copy(dataA, _aiVectorsA[size].GetDataArray(), size);

            _torchVectorsA[size].Dispose();
            _torchVectorsA[size] = torch.tensor(dataA, new long[] { size }, device: _torchDevice);
        }
    }

    #region Setup Helpers

    private void Warmup()
    {
        _ = AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[MatrixSizes[0]], _aiMatricesB[MatrixSizes[0]]);
        using (var result = torch.matmul(_torchMatricesA[MatrixSizes[0]], _torchMatricesB[MatrixSizes[0]]))
            ConsumeTorchResult(result);

        _cpuEngine.TensorAddInPlace(_aiVectorsA[VectorSizes[0]], _aiVectorsB[VectorSizes[0]]);
        torch.add_(_torchVectorsA[VectorSizes[0]], _torchVectorsB[VectorSizes[0]]);

        _ = AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);
        using (var result = torch.nn.functional.conv2d(_torchConvInput!, _torchConvKernel!,
            strides: _torchConvStride!, padding: _torchConvPadding!, dilation: _torchConvDilation!))
            ConsumeTorchResult(result);
    }

    private void InitializeConv2D()
    {
        const int batch = 1, inChannels = 16, height = 64, width = 64, outChannels = 32, kernelSize = 3;
        _convStride = 1;
        _convPadding = 1;
        _convDilation = 1;

        var inputData = CreateData(batch * inChannels * height * width, seedOffset: 7_777);
        var kernelData = CreateData(outChannels * inChannels * kernelSize * kernelSize, seedOffset: 9_999);

        _aiConvInput = new Tensor<float>(inputData, new[] { batch, inChannels, height, width });
        _aiConvKernel = new Tensor<float>(kernelData, new[] { outChannels, inChannels, kernelSize, kernelSize });
        _aiConvOutput = new Tensor<float>(new[] { batch, outChannels, height, width });

        _torchConvInput = torch.tensor(inputData, new long[] { batch, inChannels, height, width }, device: _torchDevice);
        _torchConvKernel = torch.tensor(kernelData, new long[] { outChannels, inChannels, kernelSize, kernelSize }, device: _torchDevice);
        _torchConvStride = new[] { (long)_convStride, _convStride };
        _torchConvPadding = new[] { (long)_convPadding, _convPadding };
        _torchConvDilation = new[] { (long)_convDilation, _convDilation };
    }

    private void InitializeNormalization()
    {
        const int batch = 32, channels = 64, height = 32, width = 32;
        var inputData = CreateData(batch * channels * height * width, seedOffset: 11_111);
        var gammaData = CreateData(channels, seedOffset: 22_222);
        var betaData = CreateData(channels, seedOffset: 33_333);

        _aiNormInput = new Tensor<float>(inputData, new[] { batch, channels, height, width });
        _aiNormGamma = new Tensor<float>(gammaData, new[] { channels });
        _aiNormBeta = new Tensor<float>(betaData, new[] { channels });

        _torchNormInput = torch.tensor(inputData, new long[] { batch, channels, height, width }, device: _torchDevice);
        _torchNormWeight = torch.tensor(gammaData, new long[] { channels }, device: _torchDevice);
        _torchNormBias = torch.tensor(betaData, new long[] { channels }, device: _torchDevice);
    }

    private void InitializePooling()
    {
        const int batch = 1, channels = 32, height = 64, width = 64;
        var poolData = CreateData(batch * channels * height * width, seedOffset: 44_444);

        _aiPoolInput = new Tensor<float>(poolData, new[] { batch, channels, height, width });
        _torchPoolInput = torch.tensor(poolData, new long[] { batch, channels, height, width }, device: _torchDevice);
    }

    private void InitializeSoftmax()
    {
        const int batch = 512, features = 1024;
        var softmaxData = CreateData(batch * features, seedOffset: 55_555);

        _aiSoftmaxInput = new Tensor<float>(softmaxData, new[] { batch, features });
        _torchSoftmaxInput = torch.tensor(softmaxData, new long[] { batch, features }, device: _torchDevice);
    }

    private void InitializeBackwardData()
    {
        var sigmoidData = CreateData(LargeSize, seedOffset: 66_666);
        var gradData = CreateData(LargeSize, seedOffset: 77_777);

        // Sigmoid output values should be in (0, 1)
        for (int i = 0; i < sigmoidData.Length; i++)
            sigmoidData[i] = sigmoidData[i] * 0.8f + 0.1f;

        _aiSigmoidOutput = new Tensor<float>(sigmoidData, new[] { LargeSize });
        _aiGradOutput = new Tensor<float>(gradData, new[] { LargeSize });

        // Tanh output values should be in (-1, 1)
        var tanhData = CreateData(LargeSize, seedOffset: 88_888);
        for (int i = 0; i < tanhData.Length; i++)
            tanhData[i] = tanhData[i] * 1.8f - 0.9f;
        _aiTanhOutput = new Tensor<float>(tanhData, new[] { LargeSize });

        _torchSigmoidOutput = torch.tensor(sigmoidData, new long[] { LargeSize }, device: _torchDevice);
        _torchGradOutput = torch.tensor(gradData, new long[] { LargeSize }, device: _torchDevice);
        _torchTanhOutput = torch.tensor(tanhData, new long[] { LargeSize }, device: _torchDevice);
    }

    private void InitializeAttention()
    {
        const int seqLen = 512, headDim = 64;
        var queryData = CreateData(seqLen * headDim, seedOffset: 99_999);
        var keyData = CreateData(seqLen * headDim, seedOffset: 111_111);

        _aiQueryMatrix = new Tensor<float>(queryData, new[] { seqLen, headDim });
        _aiKeyMatrix = new Tensor<float>(keyData, new[] { seqLen, headDim });

        _torchQueryMatrix = torch.tensor(queryData, new long[] { seqLen, headDim }, device: _torchDevice);
        _torchKeyMatrix = torch.tensor(keyData, new long[] { seqLen, headDim }, device: _torchDevice);
    }

    private void InitializeDoublePrecision()
    {
        const int size = LargeSize;
        const int matSize = 256;

        var dataA = new double[size];
        var dataB = new double[size];
        for (int i = 0; i < size; i++)
        {
            dataA[i] = DeterministicValue(i + 200_000);
            dataB[i] = DeterministicValue(i + 300_000);
        }

        _aiDoubleVectorA = new Tensor<double>(dataA, new[] { size });
        _aiDoubleVectorB = new Tensor<double>(dataB, new[] { size });

        _torchDoubleVectorA = torch.tensor(dataA, new long[] { size }, device: _torchDevice);
        _torchDoubleVectorB = torch.tensor(dataB, new long[] { size }, device: _torchDevice);

        var matDataA = new double[matSize * matSize];
        var matDataB = new double[matSize * matSize];
        for (int i = 0; i < matDataA.Length; i++)
        {
            matDataA[i] = DeterministicValue(i + 400_000);
            matDataB[i] = DeterministicValue(i + 500_000);
        }

        _aiDoubleMatA = new Tensor<double>(matDataA, new[] { matSize, matSize });
        _aiDoubleMatB = new Tensor<double>(matDataB, new[] { matSize, matSize });

        _torchDoubleMatA = torch.tensor(matDataA, new long[] { matSize, matSize }, device: _torchDevice);
        _torchDoubleMatB = torch.tensor(matDataB, new long[] { matSize, matSize }, device: _torchDevice);

        // Double softmax data
        const int dBatch = 512, dFeatures = 1024;
        var dSoftmaxData = new double[dBatch * dFeatures];
        for (int i = 0; i < dSoftmaxData.Length; i++)
            dSoftmaxData[i] = DeterministicValue(i + 600_000);
        _aiDoubleSoftmaxInput = new Tensor<double>(dSoftmaxData, new[] { dBatch, dFeatures });
        _torchDoubleSoftmaxInput = torch.tensor(dSoftmaxData, new long[] { dBatch, dFeatures }, device: _torchDevice);

        // Double conv2d data
        const int dConvBatch = 1, dInCh = 3, dOutCh = 16, dH = 32, dW = 32, dKs = 3;
        var dConvInputData = new double[dConvBatch * dInCh * dH * dW];
        var dConvKernelData = new double[dOutCh * dInCh * dKs * dKs];
        for (int i = 0; i < dConvInputData.Length; i++)
            dConvInputData[i] = DeterministicValue(i + 700_000);
        for (int i = 0; i < dConvKernelData.Length; i++)
            dConvKernelData[i] = DeterministicValue(i + 800_000);
        _aiDoubleConvInput = new Tensor<double>(dConvInputData, new[] { dConvBatch, dInCh, dH, dW });
        _aiDoubleConvKernel = new Tensor<double>(dConvKernelData, new[] { dOutCh, dInCh, dKs, dKs });
        _torchDoubleConvInput = torch.tensor(dConvInputData, new long[] { dConvBatch, dInCh, dH, dW }, device: _torchDevice);
        _torchDoubleConvKernel = torch.tensor(dConvKernelData, new long[] { dOutCh, dInCh, dKs, dKs }, device: _torchDevice);
    }

    private void ConsumeTorchResult(TorchTensor result) => _consumer.Consume(result);

    private static float[] CreateData(int length, int seedOffset)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = DeterministicValue(i + seedOffset);
        return data;
    }

    private static float DeterministicValue(int i)
    {
        unchecked
        {
            uint x = (uint)(i * 1664525 + 1013904223);
            return (x & 0x00FFFFFF) / 16777216f;
        }
    }

    #endregion

    #region MatMul

    [Benchmark]
    [Arguments(256)]
    [Arguments(512)]
    public Tensor<float> AiDotNet_TensorMatMul(int size)
        => AiDotNetEngine.Current.TensorMatMul(_aiMatricesA[size], _aiMatricesB[size]);

    [Benchmark]
    [Arguments(256)]
    [Arguments(512)]
    public void TorchSharp_MatMul(int size)
    {
        using var result = torch.matmul(_torchMatricesA[size], _torchMatricesB[size]);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Element-wise Arithmetic

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorAdd(int size)
    {
        _cpuEngine.TensorAddInPlace(_aiVectorsA[size], _aiVectorsB[size]);
        return _aiVectorsA[size];
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void RawTensorPrimitives_Add(int size)
    {
        System.Numerics.Tensors.TensorPrimitives.Add(
            _rawArraysA[size].AsSpan(),
            _rawArraysB[size].AsSpan(),
            _rawDestination[size].AsSpan());
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Add(int size)
    {
        torch.add_(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(_torchVectorsA[size]);
    }

    [Benchmark]
    [Arguments(1_000_000)]
    public void TorchSharp_Add_1Thread(int size)
    {
        int prev = (int)torch.get_num_threads();
        torch.set_num_threads(1);
        try { torch.add_(_torchVectorsA[size], _torchVectorsB[size]); }
        finally { torch.set_num_threads(prev); }
        ConsumeTorchResult(_torchVectorsA[size]);
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public Tensor<float> AiDotNet_TensorMultiply(int size)
    {
        _cpuEngine.TensorMultiplyInPlace(_aiVectorsA[size], _aiVectorsB[size]);
        return _aiVectorsA[size];
    }

    [Benchmark]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public void TorchSharp_Multiply(int size)
    {
        torch.mul_(_torchVectorsA[size], _torchVectorsB[size]);
        ConsumeTorchResult(_torchVectorsA[size]);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TensorSubtract()
        => _cpuEngine.TensorSubtract(_aiVectorsA[LargeSize], _aiVectorsB[LargeSize]);

    [Benchmark]
    public void TorchSharp_Subtract()
    {
        using var result = torch.sub(_torchVectorsA[LargeSize], _torchVectorsB[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TensorDivide()
        => _cpuEngine.TensorDivide(_aiVectorsA[LargeSize], _aiVectorsB[LargeSize]);

    [Benchmark]
    public void TorchSharp_Divide()
    {
        using var result = torch.div(_torchVectorsA[LargeSize], _torchVectorsB[LargeSize]);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Unary Element-wise

    [Benchmark]
    public Tensor<float> AiDotNet_TensorExp()
        => _cpuEngine.TensorExp(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Exp()
    {
        using var result = torch.exp(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TensorLog()
        => _cpuEngine.TensorLog(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Log()
    {
        using var result = torch.log(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TensorSqrt()
        => _cpuEngine.TensorSqrt(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Sqrt()
    {
        using var result = torch.sqrt(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TensorAbs()
        => _cpuEngine.TensorAbs(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Abs()
    {
        using var result = torch.abs(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Activations

    [Benchmark]
    public Tensor<float> AiDotNet_ReLU()
    {
        _cpuEngine.ReLUInPlace(_aiVectorsA[LargeSize]);
        return _aiVectorsA[LargeSize];
    }

    [Benchmark]
    public void TorchSharp_ReLU()
    {
        _torchVectorsA[LargeSize].relu_();
        ConsumeTorchResult(_torchVectorsA[LargeSize]);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Sigmoid()
    {
        _cpuEngine.SigmoidInPlace(_aiVectorsA[LargeSize]);
        return _aiVectorsA[LargeSize];
    }

    [Benchmark]
    public void RawTensorPrimitives_Sigmoid()
    {
        System.Numerics.Tensors.TensorPrimitives.Sigmoid(
            _rawArraysA[LargeSize].AsSpan(),
            _rawDestination[LargeSize].AsSpan());
    }

    [Benchmark]
    public void TorchSharp_Sigmoid()
    {
        _torchVectorsA[LargeSize].sigmoid_();
        ConsumeTorchResult(_torchVectorsA[LargeSize]);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Tanh()
        => _cpuEngine.TensorTanh(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Tanh()
    {
        using var result = torch.tanh(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_GELU()
        => _cpuEngine.TensorGELU(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_GELU()
    {
        using var result = torch.nn.functional.gelu(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_Mish()
        => _cpuEngine.TensorMish(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Mish()
    {
        using var result = torch.nn.functional.mish(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_LeakyReLU()
        => _cpuEngine.TensorLeakyReLU(_aiVectorsA[LargeSize], 0.01f);

    [Benchmark]
    public void TorchSharp_LeakyReLU()
    {
        using var result = torch.nn.functional.leaky_relu(_torchVectorsA[LargeSize], 0.01);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Reductions

    [Benchmark]
    public float AiDotNet_TensorSum()
        => AiDotNetEngine.Current.TensorSum(_aiVectorsA[LargeSize]);

    [Benchmark]
    public float RawTensorPrimitives_Sum()
        => System.Numerics.Tensors.TensorPrimitives.Sum(_rawArraysA[LargeSize].AsSpan());

    [Benchmark]
    public void TorchSharp_Sum()
    {
        using var result = torch.sum(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public float AiDotNet_TensorMean()
        => AiDotNetEngine.Current.TensorMean(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Mean()
    {
        using var result = torch.mean(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public float AiDotNet_TensorMaxValue()
        => _cpuEngine.TensorMaxValue(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Max()
    {
        using var result = torch.max(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public float AiDotNet_TensorMinValue()
        => _cpuEngine.TensorMinValue(_aiVectorsA[LargeSize]);

    [Benchmark]
    public void TorchSharp_Min()
    {
        using var result = torch.min(_torchVectorsA[LargeSize]);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Softmax

    [Benchmark]
    public Tensor<float> AiDotNet_Softmax()
        => _cpuEngine.TensorSoftmax(_aiSoftmaxInput!, axis: 1);

    [Benchmark]
    public void TorchSharp_Softmax()
    {
        using var result = torch.nn.functional.softmax(_torchSoftmaxInput!, dim: 1);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_LogSoftmax()
        => _cpuEngine.TensorLogSoftmax(_aiSoftmaxInput!, axis: 1);

    [Benchmark]
    public void TorchSharp_LogSoftmax()
    {
        using var result = torch.nn.functional.log_softmax(_torchSoftmaxInput!, dim: 1);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Conv2D

    [Benchmark]
    public Tensor<float> AiDotNet_Conv2D()
        => AiDotNetEngine.Current.Conv2D(_aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);

    [Benchmark]
    public void AiDotNet_Conv2D_ZeroAlloc()
        => _cpuEngine.Conv2DInto(_aiConvOutput!, _aiConvInput!, _aiConvKernel!, _convStride, _convPadding, _convDilation);

    [Benchmark]
    public void TorchSharp_Conv2D()
    {
        using var result = torch.nn.functional.conv2d(_torchConvInput!, _torchConvKernel!,
            strides: _torchConvStride!, padding: _torchConvPadding!, dilation: _torchConvDilation!);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Normalization

    [Benchmark]
    public Tensor<float> AiDotNet_BatchNorm()
        => _cpuEngine.BatchNorm(_aiNormInput!, _aiNormGamma!, _aiNormBeta!, 1e-5, out _, out _);

    [Benchmark]
    public void TorchSharp_BatchNorm()
    {
        using var result = torch.nn.functional.batch_norm(
            _torchNormInput!, null, null, _torchNormWeight, _torchNormBias, training: true, eps: 1e-5);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_LayerNorm()
        => _cpuEngine.LayerNorm(_aiNormInput!, _aiNormGamma!, _aiNormBeta!, 1e-5, out _, out _);

    [Benchmark]
    public void TorchSharp_LayerNorm()
    {
        using var result = torch.nn.functional.layer_norm(
            _torchNormInput!, new long[] { 64, 32, 32 }, _torchNormWeight, _torchNormBias, eps: 1e-5);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_GroupNorm()
    {
        if (_aiNormInput is null || _aiNormGamma is null || _aiNormBeta is null)
            throw new InvalidOperationException("Setup not called");
        return _cpuEngine.GroupNorm(_aiNormInput, 32, _aiNormGamma, _aiNormBeta, 1e-5, out _, out _);
    }

    [Benchmark]
    public void TorchSharp_GroupNorm()
    {
        using var result = torch.nn.functional.group_norm(
            _torchNormInput, 32, _torchNormWeight, _torchNormBias, eps: 1e-5);
        if (result is not null) ConsumeTorchResult(result);
    }

    [Benchmark]
    public void AiDotNet_GroupNormSwish()
    {
        if (_aiNormInput is null || _aiNormGamma is null || _aiNormBeta is null)
            throw new InvalidOperationException("Setup not called");
        var output = TensorAllocator.Rent<float>(_aiNormInput.Shape);
        _cpuEngine.GroupNormSwishInto(output, _aiNormInput, 32, _aiNormGamma, _aiNormBeta, 1e-5);
        TensorAllocator.Return(output);
    }

    #endregion

    #region Pooling

    [Benchmark]
    public Tensor<float> AiDotNet_MaxPool2D()
        => _cpuEngine.TensorMaxPool2D(_aiPoolInput!, poolSize: 3, stride: 2, padding: 1);

    [Benchmark]
    public void TorchSharp_MaxPool2D()
    {
        using var result = torch.nn.functional.max_pool2d(_torchPoolInput!, kernel_size: 3, stride: 2, padding: 1);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Backward Passes

    [Benchmark]
    public Tensor<float> AiDotNet_SigmoidBackward()
        => _cpuEngine.SigmoidBackward(_aiGradOutput!, _aiSigmoidOutput!);

    [Benchmark]
    public void TorchSharp_SigmoidBackward()
    {
        // sigmoid_backward: grad * output * (1 - output)
        using var oneMinusOut = torch.sub(1.0f, _torchSigmoidOutput!);
        using var gradTimesOut = torch.mul(_torchGradOutput!, _torchSigmoidOutput);
        using var result = torch.mul(gradTimesOut, oneMinusOut);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<float> AiDotNet_TanhBackward()
        => _cpuEngine.TanhBackward(_aiGradOutput!, _aiTanhOutput!);

    [Benchmark]
    public void TorchSharp_TanhBackward()
    {
        // tanh_backward: grad * (1 - output^2)
        using var outSq = torch.mul(_torchTanhOutput!, _torchTanhOutput);
        using var oneMinusSq = torch.sub(1.0f, outSq);
        using var result = torch.mul(_torchGradOutput!, oneMinusSq);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Attention Q@K^T

    [Benchmark]
    public Tensor<float> AiDotNet_AttentionQKT()
    {
        // Q @ K^T for attention scores (512x64 @ 64x512 = 512x512)
        var keyT = _cpuEngine.TensorTranspose(_aiKeyMatrix!);
        return _cpuEngine.TensorMatMul(_aiQueryMatrix!, keyT);
    }

    [Benchmark]
    public void TorchSharp_AttentionQKT()
    {
        using var keyT = _torchKeyMatrix!.t();
        using var result = torch.matmul(_torchQueryMatrix!, keyT);
        ConsumeTorchResult(result);
    }

    #endregion

    #region Double Precision

    [Benchmark]
    public Tensor<double> AiDotNet_TensorAdd_Double()
    {
        var result = _cpuEngine.TensorAdd(_aiDoubleVectorA!, _aiDoubleVectorB!);
        return result;
    }

    [Benchmark]
    public void TorchSharp_Add_Double()
    {
        using var result = torch.add(_torchDoubleVectorA!, _torchDoubleVectorB!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_MatMul_Double()
        => _cpuEngine.TensorMatMul(_aiDoubleMatA!, _aiDoubleMatB!);

    [Benchmark]
    public void TorchSharp_MatMul_Double()
    {
        using var result = torch.matmul(_torchDoubleMatA!, _torchDoubleMatB!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Sigmoid_Double()
        => _cpuEngine.Sigmoid(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_Sigmoid_Double()
    {
        using var result = torch.sigmoid(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Exp_Double()
        => _cpuEngine.TensorExp(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_Exp_Double()
    {
        using var result = torch.exp(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Log_Double()
        => _cpuEngine.TensorLog(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_Log_Double()
    {
        using var result = torch.log(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Tanh_Double()
        => _cpuEngine.TensorTanh(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_Tanh_Double()
    {
        using var result = torch.tanh(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_GELU_Double()
        => _cpuEngine.TensorGELU(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_GELU_Double()
    {
        using var result = torch.nn.functional.gelu(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Mish_Double()
        => _cpuEngine.TensorMish(_aiDoubleVectorA!);

    [Benchmark]
    public void TorchSharp_Mish_Double()
    {
        using var result = torch.nn.functional.mish(_torchDoubleVectorA!);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Softmax_Double()
        => _cpuEngine.TensorSoftmax(_aiDoubleSoftmaxInput!, axis: 1);

    [Benchmark]
    public void TorchSharp_Softmax_Double()
    {
        using var result = torch.nn.functional.softmax(_torchDoubleSoftmaxInput!, dim: 1);
        ConsumeTorchResult(result);
    }

    [Benchmark]
    public Tensor<double> AiDotNet_Conv2D_Double()
        => _cpuEngine.Conv2D(_aiDoubleConvInput!, _aiDoubleConvKernel!, _convStride, _convPadding, _convDilation);

    [Benchmark]
    public void TorchSharp_Conv2D_Double()
    {
        using var result = torch.nn.functional.conv2d(_torchDoubleConvInput!, _torchDoubleConvKernel!,
            strides: _torchConvStride!, padding: _torchConvPadding!, dilation: _torchConvDilation!);
        ConsumeTorchResult(result);
    }

    #endregion
}
#endif
