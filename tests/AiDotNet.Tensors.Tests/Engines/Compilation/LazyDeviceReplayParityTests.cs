using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Every op recorded under GraphMode replays one of two ways: a host engine writes straight into the node's output, a
/// device engine computes eagerly and copies the result in (DirectGpuTensorEngine.CopyResultInto). The device branch ran
/// only on GPU hosts, so CI never checked it. An engine reporting SupportsGpu drives it on the CPU: each replayed op must
/// match the same op run eagerly.
/// </summary>
public class LazyDeviceReplayParityTests
{
    private sealed class DeviceLikeEngine : CpuEngine
    {
        public override bool SupportsGpu => true;
    }

    private static Tensor<float> Seq(int[] shape, float scale, float shift)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)Math.Sin(i * 0.37 + shift) * scale + 0.1f * scale;
        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> Positive(int[] shape) => Seq(shape, 1f, 0.3f).Transform((v, _) => Math.Abs(v) + 0.05f);

    public static IEnumerable<object[]> Ops()
    {
        var img = Seq(new[] { 1, 4, 6, 6 }, 1f, 0.1f);
        var kernel = Seq(new[] { 3, 4, 3, 3 }, 0.5f, 0.7f);
        var depthwise = Seq(new[] { 4, 1, 3, 3 }, 0.5f, 0.9f);
        var shuffle = Seq(new[] { 1, 8, 3, 3 }, 1f, 0.2f);
        var vec = Seq(new[] { 3, 5 }, 2f, 0.4f);
        var pos = Positive(new[] { 3, 5 });
        var other = Seq(new[] { 3, 2 }, 1f, 0.5f);
        Func<IEngine, Tensor<float>> f;

        yield return Case("TensorLog", e => e.TensorLog(pos));
        yield return Case("TensorExp", e => e.TensorExp(vec));
        yield return Case("TensorSqrt", e => e.TensorSqrt(pos));
        yield return Case("TensorAbs", e => e.TensorAbs(vec));
        yield return Case("TensorSin", e => e.TensorSin(vec));
        yield return Case("TensorCos", e => e.TensorCos(vec));
        yield return Case("TensorMultiplyScalar", e => e.TensorMultiplyScalar(vec, 1.5f));
        yield return Case("Mish", e => e.Mish(vec));
        yield return Case("ELU", e => e.ELU(vec, 0.7));
        yield return Case("MaxPool2D", e => e.MaxPool2D(img, 2, 2));
        yield return Case("AvgPool2D", e => e.AvgPool2D(img, 2, 2));
        yield return Case("Conv2D", e => e.Conv2D(img, kernel, 1, 1));
        yield return Case("DepthwiseConv2D", e => e.DepthwiseConv2D(img, depthwise, new[] { 1, 1 }, new[] { 1, 1 }));
        yield return Case("Upsample", e => e.Upsample(img, 2, 2));
        yield return Case("PixelShuffle", e => e.PixelShuffle(shuffle, 2));
        yield return Case("Crop", e => e.Crop(img, 1, 1, 3, 4));
        yield return Case("Pad", e => e.Pad(img, 1, 0, 2, 1, 0.25f));
        yield return Case("Concat", e => e.Concat(new[] { vec, other }, 1));
        yield return Case("AdaptiveAvgPool2D", e => e.AdaptiveAvgPool2D(img, 4, 3));
        f = e => e.TensorUpsampleBilinear(img, new[] { 9, 7 });
        yield return Case("TensorUpsampleBilinear", f);
    }

    private static object[] Case(string name, Func<IEngine, Tensor<float>> op) => new object[] { name, op };

    [Theory]
    [MemberData(nameof(Ops))]
    public void DeviceReplay_MatchesEager(string name, Func<IEngine, Tensor<float>> op)
    {
        var expected = op(new CpuEngine()).ToArray();

        var engine = new DeviceLikeEngine();
        Tensor<float> lazy;
        using (GraphMode.Enable())
        {
            lazy = op(engine);
            Assert.True(lazy.LazySource is not null, $"{name} was not recorded under GraphMode.");
        }

        var actual = lazy.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"{name}[{i}]: eager {expected[i]}, device replay {actual[i]}");
    }
}
