#if NET8_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Conv3D backward acceptance benchmark for video-diffusion temporal blocks.
/// The previous seven-loop implementation made this operation the dominant
/// StableVideoSR training hot path. These shapes retain four-frame temporal
/// depth and the 64/256-channel profiles used by the U-Net while keeping the
/// benchmark practical on a developer workstation.
///
/// Run:
///   dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- \
///     --conv3d-backward
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 8)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class Conv3DBackwardBenchmarks
{
    private static readonly int[] Stride = [1, 1, 1];
    private static readonly int[] Padding = [1, 1, 1];
    private static readonly int[] Dilation = [1, 1, 1];

    [Params(64, 256)]
    public int Channels { get; set; }

    private CpuEngine _engine = null!;
    private Tensor<float> _input = null!;
    private Tensor<float> _kernel = null!;
    private Tensor<float> _gradOutput = null!;
    private int[] _inputShape = null!;
    private int[] _kernelShape = null!;

    [GlobalSetup]
    public void Setup()
    {
        _engine = new CpuEngine();
        _inputShape = [1, Channels, 4, 8, 8];
        _kernelShape = [Channels, Channels, 3, 3, 3];
        _input = Tensor<float>.CreateRandom(_inputShape);
        _kernel = Tensor<float>.CreateRandom(_kernelShape);
        _gradOutput = Tensor<float>.CreateRandom(_inputShape);
    }

    [Benchmark(Baseline = true, Description = "Conv3D dInput (video temporal block)")]
    public Tensor<float> InputGradient()
        => _engine.Conv3DBackwardInput(
            _gradOutput, _kernel, _inputShape, Stride, Padding, Dilation);

    [Benchmark(Description = "Conv3D dKernel (video temporal block)")]
    public Tensor<float> KernelGradient()
        => _engine.Conv3DBackwardKernel(
            _gradOutput, _input, _kernelShape, Stride, Padding, Dilation);
}
#endif
