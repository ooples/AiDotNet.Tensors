using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Stats-Serial")]
public sealed class Conv1DTaskGridTests
{
    [Theory]
    [InlineData(1, 1024, 1024, 128, 5)]
    [InlineData(1, 1024, 512, 128, 5)]
    [InlineData(1, 256, 256, 256, 11)]
    [InlineData(1, 1024, 1024, 65, 5)]
    [InlineData(1, 1024, 1024, 33, 5)]
    [InlineData(7, 1024, 1024, 15, 5)]
    [InlineData(1, 128, 257, 257, 3)]
    public void OutputRowTasksMatchOriginalGemmGroupingBitwise(int batch, int channels, int outputs, int width, int taps)
    {
        int previous = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 8;
            int inputLength = batch * channels * width, kernelLength = outputs * channels * taps;
            int outputLength = batch * outputs * width;
            var input = new float[inputLength + 2]; var kernel = new float[kernelLength + 2];
            var expected = Enumerable.Repeat(-7f, outputLength + 2).ToArray();
            var actual = expected.ToArray();
            for (int i = 0; i < inputLength; i++) input[i + 1] = (float)(.13 * Math.Sin(i * .037));
            for (int i = 0; i < kernelLength; i++) kernel[i + 1] = (float)(.07 * Math.Cos(i * .019));
            Assert.True(CpuEngine.TryConv1DChannelBlocks(input, 1, kernel, 1, expected, 1,
                batch, channels, width, outputs, taps, 1, taps / 2, 1, width, splitChannels: false));
            Assert.True(CpuEngine.TryConv1DChannelBlocks(input, 1, kernel, 1, actual, 1,
                batch, channels, width, outputs, taps, 1, taps / 2, 1, width, splitChannels: true));
            for (int i = 0; i < actual.Length; i++)
                Assert.Equal(MathCompat.SingleToInt32Bits(expected[i]), MathCompat.SingleToInt32Bits(actual[i]));
            // Changing live weights must not reuse an earlier row-pack.
            kernel[1] += .01f;
            Assert.True(CpuEngine.TryConv1DChannelBlocks(input, 1, kernel, 1, expected, 1,
                batch, channels, width, outputs, taps, 1, taps / 2, 1, width, splitChannels: false));
            Assert.True(CpuEngine.TryConv1DChannelBlocks(input, 1, kernel, 1, actual, 1,
                batch, channels, width, outputs, taps, 1, taps / 2, 1, width, splitChannels: true));
            Assert.Equal(expected, actual);
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = previous; }
    }
}
