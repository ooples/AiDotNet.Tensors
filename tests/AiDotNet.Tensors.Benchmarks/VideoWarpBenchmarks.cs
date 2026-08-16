using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Measures the production flat-storage, batch-parallel forward splat against the
/// former rank-4-indexer implementation at representative UPRNet feature-map shapes.
/// </summary>
[MemoryDiagnoser]
public class VideoWarpBenchmarks
{
    private readonly CpuEngine _engine = new();
    private Tensor<float> _input = null!;
    private Tensor<float> _flow = null!;

    [Params(1, 4)]
    public int Batch { get; set; }

    [Params(32)]
    public int Channels { get; set; }

    [Params(64)]
    public int SpatialSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(951);
        int pixels = Batch * SpatialSize * SpatialSize;
        _input = new Tensor<float>(
            Enumerable.Range(0, pixels * Channels)
                .Select(_ => random.NextSingle() * 2f - 1f).ToArray(),
            [Batch, Channels, SpatialSize, SpatialSize]);
        _flow = new Tensor<float>(
            Enumerable.Range(0, pixels * 2)
                .Select(_ => random.NextSingle() * 1.5f - 0.75f).ToArray(),
            [Batch, 2, SpatialSize, SpatialSize]);
    }

    [Benchmark(Baseline = true)]
    public Tensor<float> FormerIndexerImplementation()
    {
        int height = SpatialSize;
        int width = SpatialSize;
        var accumulated = new Tensor<float>(_input.Shape.ToArray());
        var weights = new double[Batch * height * width];

        for (int batch = 0; batch < Batch; batch++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            double destinationX = x + _flow[batch, 0, y, x];
            double destinationY = y + _flow[batch, 1, y, x];
            int x0 = (int)Math.Floor(destinationX);
            int y0 = (int)Math.Floor(destinationY);
            double fractionX = destinationX - x0;
            double fractionY = destinationY - y0;
            Accumulate(x0, y0, (1.0 - fractionX) * (1.0 - fractionY));
            Accumulate(x0 + 1, y0, fractionX * (1.0 - fractionY));
            Accumulate(x0, y0 + 1, (1.0 - fractionX) * fractionY);
            Accumulate(x0 + 1, y0 + 1, fractionX * fractionY);

            void Accumulate(int destinationPixelX, int destinationPixelY, double weight)
            {
                if ((uint)destinationPixelX >= (uint)width ||
                    (uint)destinationPixelY >= (uint)height || weight == 0.0)
                    return;

                weights[(batch * height + destinationPixelY) * width + destinationPixelX] += weight;
                float typedWeight = (float)weight;
                for (int channel = 0; channel < Channels; channel++)
                {
                    accumulated[batch, channel, destinationPixelY, destinationPixelX] +=
                        _input[batch, channel, y, x] * typedWeight;
                }
            }
        }

        var result = new Tensor<float>(_input.Shape.ToArray());
        for (int batch = 0; batch < Batch; batch++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            double weight = weights[(batch * height + y) * width + x];
            float denominator = (float)(weight == 0.0 ? 1.0 : weight);
            for (int channel = 0; channel < Channels; channel++)
            {
                result[batch, channel, y, x] =
                    accumulated[batch, channel, y, x] / denominator;
            }
        }

        return result;
    }

    [Benchmark]
    public Tensor<float> FlatStorageBatchParallel()
        => _engine.ForwardSplat(_input, _flow, normalize: true);
}
