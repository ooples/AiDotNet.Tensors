#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 5, iterationCount: 15)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class Issue305FirstForwardInitBenchmarks
{
    private const float Min = -0.05f;
    private const float Max = 0.05f;
    private const float StdDev = 0.02f;

    private readonly Consumer _consumer = new();
    private CpuEngine _engine = null!;
    private Tensor<float> _destination = null!;
    private TorchTensor _torchDestination = null!;

    [Params(1_000_000, 16_777_216)]
    public int Elements { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();
        _destination = new Tensor<float>([Elements]);

        torch.set_grad_enabled(false);
        torch.set_num_threads(Environment.ProcessorCount);
        _torchDestination = torch.empty([Elements], device: torch.CPU);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _torchDestination.Dispose();
    }

    [Benchmark(Baseline = true)]
    public void AiDotNet_OldRandomUniformRangeThenCopy()
    {
        var random = _engine.TensorRandomUniformRange<float>([Elements], Min, Max);
        random.AsSpan().CopyTo(_destination.AsWritableSpan());
        _consumer.Consume(_destination);
    }

    [Benchmark]
    public void AiDotNet_RandomUniformRangeInto()
    {
        _engine.TensorRandomUniformRangeInto(_destination, Min, Max);
        _consumer.Consume(_destination);
    }

    [Benchmark]
    public void AiDotNet_RandomNormalInto()
    {
        _engine.TensorRandomNormalInto(_destination, 0f, StdDev);
        _consumer.Consume(_destination);
    }

    [Benchmark]
    public void TorchSharp_UniformInPlace()
    {
        _torchDestination.uniform_(Min, Max);
        _consumer.Consume(_torchDestination);
    }

    [Benchmark]
    public void TorchSharp_NormalInPlace()
    {
        _torchDestination.normal_(0.0, StdDev);
        _consumer.Consume(_torchDestination);
    }
}
#endif
