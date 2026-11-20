using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class ShortWaveInputGradientTests
{
    [Theory]
    [InlineData(256, 33, false)]
    [InlineData(256, 65, true)]
    [InlineData(256, 128, false)]
    [InlineData(1024, 33, true)]
    public void DirectTransposedProductMatchesIndependentDoubleDerivativeAndLiveOffsetWeights(int channels, int width, bool add)
    {
        const int taps = 5, padding = 2;
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 8;
            var engine = new CpuEngine();
            var weightBacking = Enumerable.Repeat(-7f, channels * channels * taps + 2).ToArray();
            var upstreamBacking = Enumerable.Repeat(-9f, channels * width + 2).ToArray();
            var targetBacking = Enumerable.Repeat(.25f, channels * width + 2).ToArray();
            using var kernel = Tensor<float>.FromMemory(weightBacking.AsMemory(1, weightBacking.Length - 2), new[] { channels, channels, 1, taps });
            using var upstream = Tensor<float>.FromMemory(upstreamBacking.AsMemory(1, upstreamBacking.Length - 2), new[] { 1, channels, 1, width });
            using var target = Tensor<float>.FromMemory(targetBacking.AsMemory(1, targetBacking.Length - 2), new[] { 1, channels, 1, width });
            for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(.05 * Math.Cos(i * .013));
            for (int i = 0; i < upstream.Length; i++) upstream[i] = (float)(.2 * Math.Sin(i * .037));
            for (int replay = 0; replay < 2; replay++)
            {
                if (replay != 0) kernel[0] += .01f;
                target.Fill(.25f);
                int before = CpuEngine.ShortWaveInputGradientRunsOnThisThread;
                engine.Conv2DBackwardInputInto(target, upstream, kernel, target.Shape.ToArray(),
                    new[] { 1, 1 }, new[] { 0, padding }, new[] { 1, 1 }, add);
                Assert.Equal(before + 1, CpuEngine.ShortWaveInputGradientRunsOnThisThread);
                foreach (int ic in new[] { 0, channels / 2, channels - 1 })
                    foreach (int time in new[] { 0, width / 2, width - 1 })
                    {
                        double expected = add ? .25 : 0;
                        for (int oc = 0; oc < channels; oc++) for (int tap = 0; tap < taps; tap++)
                        {
                            int from = time + padding - tap;
                            if ((uint)from < (uint)width)
                                expected += (double)kernel[(oc * channels + ic) * taps + tap] * upstream[oc * width + from];
                        }
                        Assert.InRange(Math.Abs(target[ic * width + time] - expected), 0, 2e-5 * Math.Max(1, Math.Abs(expected)));
                    }
                Assert.Equal(-7f, weightBacking[0]); Assert.Equal(-7f, weightBacking[^1]);
                Assert.Equal(-9f, upstreamBacking[0]); Assert.Equal(-9f, upstreamBacking[^1]);
                Assert.Equal(.25f, targetBacking[0]); Assert.Equal(.25f, targetBacking[^1]);
            }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
    }
}
