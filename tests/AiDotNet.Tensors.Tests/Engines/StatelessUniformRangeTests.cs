using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class StatelessUniformRangeTests
{
    [Fact]
    public void SeededUnitRange_IsExactAcrossFloatDoubleAndParallelThreshold()
    {
        var engine = new CpuEngine();
        float[] small = engine.TensorRandomUniformRange<float>([257], 0f, 1f, 1729).ToArray();
        float[] large = engine.TensorRandomUniformRange<float>([20000], 0f, 1f, 1729).ToArray();
        double[] oracle = engine.TensorRandomUniformRange<double>([20000], 0.0, 1.0, 1729).ToArray();

        for (int i = 0; i < large.Length; i++)
        {
            Assert.Equal(large[i], (float)oracle[i]);
            Assert.InRange(large[i], 0f, float.BitDecrement(1f));
            if (i < small.Length)
                Assert.Equal(small[i], large[i]);
        }
    }

    [Fact]
    public void SeededRange_IsReproducibleAndSeedSensitive()
    {
        var engine = new CpuEngine();
        float[] first = engine.TensorRandomUniformRange<float>([1024], -3.25f, 7.5f, 17).ToArray();
        float[] repeat = engine.TensorRandomUniformRange<float>([1024], -3.25f, 7.5f, 17).ToArray();
        float[] different = engine.TensorRandomUniformRange<float>([1024], -3.25f, 7.5f, 18).ToArray();

        Assert.Equal(first, repeat);
        Assert.NotEqual(first, different);
        Assert.All(first, value => Assert.InRange(value, -3.25f, float.BitDecrement(7.5f)));
    }

    [Fact]
    public void ZeroSeed_IsAStableExplicitSeed()
    {
        var engine = new CpuEngine();
        float[] first = engine.TensorRandomUniformRange<float>([4096], -2f, 2f, 0).ToArray();
        float[] repeat = engine.TensorRandomUniformRange<float>([4096], -2f, 2f, 0).ToArray();

        Assert.Equal(first, repeat);
        Assert.Contains(first, value => value != 0f);
    }
}
