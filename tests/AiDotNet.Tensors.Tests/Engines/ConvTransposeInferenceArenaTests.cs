using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("BlasManaged-Perf-Serial")]
public sealed class ConvTransposeInferenceArenaTests
{
    [Theory]
    [InlineData(false, 2, 4, 0)]
    [InlineData(false, 8, 16, 0)]
    [InlineData(false, 2, 5, 1)]
    [InlineData(true, 2, 4, 0)]
    [InlineData(true, 8, 16, 0)]
    [InlineData(true, 2, 5, 1)]
    public void ArenaForwardMatchesUnpooledAndKeepsEarlierOutputs(bool useDouble, int rate, int kernel, int outputPadding)
    {
        if (useDouble) Check<double>(rate, kernel, outputPadding);
        else Check<float>(rate, kernel, outputPadding);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ArenaSpatialForwardMatchesUnpooled(bool useDouble)
    {
        if (useDouble) Check<double>(2, 4, 1, spatial2D: true);
        else Check<float>(2, 4, 1, spatial2D: true);
    }

    private static void Check<T>(int rate, int kernelWidth, int outputPadding, bool spatial2D = false)
    {
        var engine = new CpuEngine();
        var operations = MathHelper.GetNumericOperations<T>();
        int height = spatial2D ? 3 : 1, kernelHeight = spatial2D ? 3 : 1;
        var input = new Tensor<T>(Enumerable.Range(0, 2 * 2 * height * 7)
            .Select(i => operations.FromDouble(.2 * Math.Sin(i * .17))).ToArray(), new[] { 2, 2, height, 7 });
        var kernel = new Tensor<T>(Enumerable.Range(0, 2 * 3 * kernelHeight * kernelWidth)
            .Select(i => operations.FromDouble(.1 * Math.Cos(i * .23))).ToArray(), new[] { 2, 3, kernelHeight, kernelWidth });
        var originalInput = input.ToArray();
        var originalKernel = kernel.ToArray();
        int[] stride = { spatial2D ? 2 : 1, rate }, padding = { spatial2D ? 1 : 0, (kernelWidth - rate) / 2 }, extra = { spatial2D ? 1 : 0, outputPadding };
        T[] expected;
        using (new NoGradScope<T>()) expected = engine.ConvTranspose2D(input, kernel, stride, padding, extra).ToArray();
        using var arena = TensorArena.Create();
        var sentinel = TensorAllocator.Rent<T>(new[] { 31 });
        sentinel.Fill(operations.FromDouble(91));
        using var noGrad = new NoGradScope<T>();
        var first = engine.ConvTranspose2D(input, kernel, stride, padding, extra);
        AssertBits(expected, first.ToArray());
        for (int i = 0; i < 3; i++)
        {
            AssertBits(expected, engine.ConvTranspose2D(input, kernel, stride, padding, extra).ToArray());
            AssertBits(expected, first.ToArray());
            Assert.All(sentinel.ToArray(), value => Assert.Equal(operations.FromDouble(91), value));
        }
        Assert.Equal(originalInput, input.ToArray());
        Assert.Equal(originalKernel, kernel.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IntoWritesLivePaddedBackingAndPreservesPadding(bool useDouble)
    {
        if (useDouble) CheckPadding<double>();
        else CheckPadding<float>();
    }

    private static void CheckPadding<T>()
    {
        var engine = new CpuEngine();
        var operations = MathHelper.GetNumericOperations<T>();
        var input = new Tensor<T>(new[] { operations.FromDouble(1), operations.FromDouble(2), operations.FromDouble(3) }, new[] { 1, 1, 1, 3 });
        var kernel = new Tensor<T>(new[] { operations.FromDouble(2), operations.FromDouble(-1) }, new[] { 1, 1, 1, 2 });
        int[] stride = { 1, 2 }, padding = { 0, 0 }, extra = { 0, 1 };
        using var noGrad = new NoGradScope<T>();
        var expected = engine.ConvTranspose2D(input, kernel, stride, padding, extra).ToArray();
        var backing = Enumerable.Repeat(operations.FromDouble(321), expected.Length + 19).ToArray();
        var destination = Tensor<T>.FromMemory(new Memory<T>(backing, 0, expected.Length), new[] { 1, 1, 1, expected.Length });
        engine.ConvTranspose2DInto(destination, input, kernel, stride, padding, extra);
        AssertBits(expected, destination.ToArray());
        Assert.All(backing.Skip(expected.Length), value => Assert.Equal(operations.FromDouble(321), value));
    }

    [Fact]
    public void IntoPrivatizesCopyOnWriteDestination()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1f, 2f }, new[] { 1, 1, 1, 2 });
        var kernel = new Tensor<float>(new[] { 2f, 3f }, new[] { 1, 1, 1, 2 });
        var peer = new Tensor<float>(new[] { 91f, 91f, 91f }, new[] { 1, 1, 1, 3 });
        var destination = (Tensor<float>)peer.CloneShared();
        Assert.True(destination.IsCowShared);
        using var noGrad = new NoGradScope<float>();
        engine.ConvTranspose2DInto(destination, input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 0, 0 });
        Assert.Equal(new[] { 2f, 7f, 6f }, destination.ToArray());
        Assert.Equal(new[] { 91f, 91f, 91f }, peer.ToArray());
    }

    [Fact]
    public void IntoConsumesDeferredDestinationDownloadBeforeWriting()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 1f, 2f }, new[] { 1, 1, 1, 2 });
        var kernel = new Tensor<float>(new[] { 2f, 3f }, new[] { 1, 1, 1, 2 });
        var backing = new float[32];
        var destination = Tensor<float>.FromMemory(new Memory<float>(backing, 0, 3), new[] { 1, 1, 1, 3 });
        int downloads = 0;
        HostSync.Register(backing, value => { downloads++; Array.Fill((float[])value, 91f); });
        Assert.True(HostSync.IsPending(backing));
        using var noGrad = new NoGradScope<float>();
        engine.ConvTranspose2DInto(destination, input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 0, 0 });
        Assert.Equal(1, downloads);
        Assert.False(HostSync.IsPending(backing));
        Assert.Equal(new[] { 2f, 7f, 6f }, destination.ToArray());
        Assert.Equal(1, downloads);
        Assert.All(backing.Skip(3), value => Assert.Equal(91f, value));
    }

    [Fact]
    public void TapedForwardInsideArenaKeepsAnalyticInputAndKernelGradients()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var input = new Tensor<double>(new[] { 1d, 2d }, new[] { 1, 1, 1, 2 });
            var kernel = new Tensor<double>(new[] { 2d, 3d }, new[] { 1, 1, 1, 2 });
            using var arena = TensorArena.Create();
            using var tape = new GradientTape<double>();
            var output = engine.ConvTranspose2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 0, 0 });
            Assert.Equal(new[] { 2d, 7d, 6d }, output.ToArray());
            var loss = engine.ReduceSum(output, null);
            var gradients = tape.ComputeGradients(loss, new[] { input, kernel });
            Assert.Equal(new[] { 5d, 5d }, gradients[input].ToArray());
            Assert.Equal(new[] { 3d, 3d }, gradients[kernel].ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void WarmArenaAvoidsActivationSizedGcArrays()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(Enumerable.Repeat(.2f, 2 * 32768).ToArray(), new[] { 1, 2, 1, 32768 });
        var kernel = new Tensor<float>(Enumerable.Repeat(.1f, 2 * 4 * 4).ToArray(), new[] { 2, 4, 1, 4 });
        int[] stride = { 1, 2 }, padding = { 0, 1 }, extra = { 0, 0 };
        using var noGrad = new NoGradScope<float>();
        var expectedFirst = engine.ConvTranspose2D(input, kernel, stride, padding, extra)[0];
        using var arena = TensorArena.Create();
        for (int i = 0; i < 3; i++)
        {
            arena.Reset();
            Assert.Equal(expectedFirst, engine.ConvTranspose2D(input, kernel, stride, padding, extra)[0]);
        }
        long start = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < 4; i++)
        {
            arena.Reset();
            Assert.Equal(expectedFirst, engine.ConvTranspose2D(input, kernel, stride, padding, extra)[0]);
        }
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - start;
        Assert.True(allocated < 1024 * 1024,
            $"Four warm calls allocated {allocated} bytes; each result alone is 1 MiB.");
    }

    private static void AssertBits<T>(T[] expected, T[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        if (typeof(T) == typeof(float))
        {
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits((float)(object)expected[i]!), BitConverter.SingleToInt32Bits((float)(object)actual[i]!));
        }
        else
        {
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(BitConverter.DoubleToInt64Bits((double)(object)expected[i]!), BitConverter.DoubleToInt64Bits((double)(object)actual[i]!));
        }
    }
}
